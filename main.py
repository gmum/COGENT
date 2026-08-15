"""
Per-step computation::

  render_all_cameras -> grayscale -> stack (1, 1, D, H, W)
  -> resize to (H, W) -> crop/pad depth to 200 -> *255 -> normalize
  -> repeat to 3 channels -> (1, 3, 200, 256, 256)
  -> SybilNet ensemble (raw logits, no sigmoid / calibrate / clip)
  -> loss on logit[target_year]
  -> backward -> PGD step on selected Gaussian attributes

Example::

  python3 -m sybiInterpretability.attack.main \\
    --model_path output/1 \\
    --source_path data/1 \\
    --masks <patient_id>/mask \\
    --output_ply attacked_clf.ply \\
    --sigma 0.25 --remove_top_pct 10.0 \\
    --props f_dc_0 f_dc_1 f_dc_2 \\
    --eps 0.5 --steps 10 \\
    --attack_mode untargeted \\
    --classifier_config_dir /abs/path/to/SybilInference/configs \\
    --classifier_config_name nlst_sybil_ensemble_inference \\
    --target_year 5
"""
import os
import sys

import numpy as np
import torch

from attack.medgs_bootstrap import bootstrap_medgs_path

_MEDGS_ROOT = bootstrap_medgs_path()

try:
    from submodules.MedGS.arguments import ModelParams, PipelineParams
    from submodules.MedGS.gaussian_renderer import render as gs_render
    from submodules.MedGS.scene import Scene
    from submodules.MedGS.scene.gaussian_model import GaussianModel
except ImportError as e:
    print(f"Failed to import MedGS (MEDGS_ROOT={_MEDGS_ROOT}): {e}")
    sys.exit(1)

from attack.camera_loader import load_cameras_with_masks
from attack.classifier_loader import load_sybil_classifier
from attack.cli import parse_args
from attack.mask_filtering import find_gaussians_in_masks, remove_largest_gaussians
from attack.pgd_attack import run_classifier_pgd_attack
from attack.ply_io import detect_sh_degree_from_ply, read_ply_attributes
from attack.scene_setup import build_medgs_args, merge_cfg_args_and_defaults

def _resolve_latest_iteration(model_path):
    pc_root = os.path.join(model_path, "point_cloud")
    iters = [int(d.split("_")[1]) for d in os.listdir(pc_root)
             if d.startswith("iteration_") and d.split("_")[1].isdigit()]
    return max(iters)


def _deduplicate_camera_names(cameras):
    seen = set()
    out = []
    for c in cameras:
        if c["img_name"] in seen:
            continue
        seen.add(c["img_name"])
        out.append(c)
    return out


def _align_hit_mask_length(hit, n_model):
    """Trim or zero-pad the mask so it matches the loaded GaussianModel."""
    if len(hit) == n_model:
        return hit
    print(f"WARNING: PLY({len(hit)}) != model({n_model}) - resizing hit mask.")
    if n_model < len(hit):
        return hit[:n_model]
    return np.concatenate([hit, np.zeros(n_model - len(hit), dtype=bool)])


def main():
    args, extra = parse_args(doc=__doc__)
    if args.alpha is None:
        args.alpha = args.eps / max(1, args.steps)
    if args.cameras_json is None:
        args.cameras_json = os.path.join(args.model_path, "cameras.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"MedGS:  {_MEDGS_ROOT}")

    # --- Step 1: load PLY and matching mask cameras -----------------------
    print("\n" + "=" * 60)
    print("STEP 1: load PLY + mask cameras")
    print("=" * 60)

    if args.iteration < 0:
        args.iteration = _resolve_latest_iteration(args.model_path)
        print(f"  Iteration: {args.iteration}")

    ply_path = os.path.join(args.model_path, "point_cloud",
                            f"iteration_{args.iteration}", "point_cloud.ply")
    xyz, scales, rot_angles = read_ply_attributes(ply_path)
    cameras = load_cameras_with_masks(args.cameras_json, args.masks)

    if not args.use_mirrors_for_filtering:
        cameras = _deduplicate_camera_names(cameras)

    if not cameras:
        print("No cameras with masks - aborting.")
        sys.exit(1)

    # --- Step 2: filter Gaussians by mask projection ----------------------
    print("\n" + "=" * 60)
    print("STEP 2: filter Gaussians by mask projection")
    print("=" * 60)
    hit = find_gaussians_in_masks(
        xyz, scales, rot_angles, cameras,
        white_threshold=args.threshold, sigma_factor=args.sigma,
    )
    if args.remove_top_pct > 0.0:
        hit = remove_largest_gaussians(hit, scales, args.remove_top_pct)

    print(f"\nTo attack: {hit.sum()} / {len(hit)}")
    if hit.sum() == 0:
        print("Empty selection - nothing to attack.")
        sys.exit(0)

    # --- Step 3: load MedGS Scene -----------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: load MedGS Scene")
    print("=" * 60)

    detected_sh = detect_sh_degree_from_ply(ply_path)
    if detected_sh is not None and detected_sh != args.sh_degree:
        print(f"  Detected sh_degree from PLY: {detected_sh} "
              f"(was {args.sh_degree})")
        args.sh_degree = detected_sh

    medgs_args, mp_cls, pp_cls = build_medgs_args(
        args, extra, ModelParams, PipelineParams, args.sh_degree)
    pipe = pp_cls.extract(medgs_args)
    model_params = mp_cls.extract(medgs_args)
    model_params = merge_cfg_args_and_defaults(
        model_params, args.model_path, override_source_path=args.source_path)

    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(model_params, gaussians,
                  load_iteration=args.iteration, shuffle=False)

    hit = _align_hit_mask_length(hit, gaussians._xyz.shape[0])

    bg_color = [1, 1, 1] if getattr(model_params, "white_background", False) else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # --- Step 4: load Sybil classifier ------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: load Sybil classifier")
    print("=" * 60)
    if args.classifier_device == "cuda" and not torch.cuda.is_available():
        print("  CUDA unavailable, falling back to CPU for the classifier.")
        clf_device = torch.device("cpu")
    else:
        clf_device = torch.device(args.classifier_device)
    classifier = load_sybil_classifier(
        args.classifier_config_dir, args.classifier_config_name,
        clf_device, n_models_to_load=args.n_ensemble_load,
    )

    # --- Step 5: PGD attack ------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: PGD attack")
    print("=" * 60)

    run_classifier_pgd_attack(
        gaussians=gaussians, scene=scene, pipe=pipe, background=background,
        subset_mask_np=hit, attacked_props=args.props,
        classifier=classifier, gs_render_fn=gs_render,
        target_year=args.target_year, target_value=args.target_value,
        eps=args.eps, alpha=args.alpha, steps=args.steps,
        attack_mode=args.attack_mode, clip=tuple(args.clip),
        device=device, max_slices=args.max_slices,
        n_ensemble_models=args.n_ensemble_models,
        clf_spatial_size=tuple(args.clf_spatial_size),
        pad_depth=args.pad_depth,
    )

    # --- Step 6: save attacked PLY ----------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: save attacked PLY")
    print("=" * 60)
    output_ply_abs = os.path.abspath(args.output_ply)
    os.makedirs(os.path.dirname(output_ply_abs) or ".", exist_ok=True)
    gaussians.save_ply(output_ply_abs)
    print(f"Saved: {output_ply_abs}")
    print("Done.")


if __name__ == "__main__":
    main()
