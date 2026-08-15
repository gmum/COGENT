"""Command-line argument parser for the adversarial attack pipeline."""
import argparse


def parse_args(doc=None):
    """Parse the CLI for ``python -m attack.main``.

    Returns ``(args, extra)`` where ``extra`` is the list of unknown
    arguments forwarded to MedGS's own argparse-based config.
    """
    p = argparse.ArgumentParser(
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- MedGS / model location -------------------------------------------
    p.add_argument("--medgs_root", default=None,
                   help="Path to the MedGS repository (overrides MEDGS_ROOT).")
    p.add_argument("--model_path", "-m", required=True,
                   help="MedGS output directory containing point_cloud/ and cfg_args.")
    p.add_argument("--source_path", "-s", default="data/lungs",
                   help="MedGS source dataset path (must match training).")
    p.add_argument("--iteration", type=int, default=-1,
                   help="Iteration number to load. -1 = latest.")

    # --- Mask-based filtering ---------------------------------------------
    p.add_argument("--cameras_json", default=None,
                   help="Path to cameras.json. Default: <model_path>/cameras.json")
    p.add_argument("--masks", required=True,
                   help="Directory with per-slice lesion masks.")
    p.add_argument("--threshold", type=int, default=128,
                   help="Pixel value threshold for treating a mask pixel as set.")
    p.add_argument("--sigma", type=float, default=3.0,
                   help="Mahalanobis radius (in stddevs) for mask intersection.")
    p.add_argument("--remove_top_pct", type=float, default=0.0,
                   help="Drop this percent of the largest Gaussians (by area) "
                        "from the hit set before attacking.")
    p.add_argument("--use_mirrors_for_filtering", action="store_true",
                   help="Include mirrored slices when filtering by masks.")

    # --- Attack hyperparameters -------------------------------------------
    p.add_argument("--props", nargs="+", required=True,
                   help="PLY-style attribute names to attack "
                        "(e.g. f_dc_0 f_dc_1 f_dc_2).")
    p.add_argument("--eps", type=float, default=0.5,
                   help="L-infinity radius around the original parameter values.")
    p.add_argument("--alpha", type=float, default=None,
                   help="PGD step size. Default: eps / steps.")
    p.add_argument("--steps", type=int, default=10,
                   help="Number of PGD iterations.")
    p.add_argument("--attack_mode",
                   choices=["untargeted", "targeted"], default="untargeted",
                   help="untargeted = maximize the logit (healthy -> sick); "
                        "targeted = drive the logit toward --target_value.")
    p.add_argument("--clip", nargs=2, type=float, default=(-10.0, 10.0),
                   help="Absolute clip range for attacked attribute values.")

    # --- Sybil classifier --------------------------------------------------
    p.add_argument("--classifier_config_dir", required=True,
                   help="Absolute path to the Hydra config directory "
                        "(e.g. /path/to/SybilInference/configs).")
    p.add_argument("--classifier_config_name", required=True,
                   help="Hydra config name (e.g. nlst_sybil_ensemble_inference).")
    p.add_argument("--target_year", type=int, default=5,
                   help="Logit index: 0=year 1, ..., 5=year 6; -1=mean.")
    p.add_argument("--target_value", type=float, default=0.0,
                   help="Target logit value (only used in targeted mode).")
    p.add_argument("--max_slices", type=int, default=0,
                   help="Cap on slices fed to the classifier (0 = all). "
                        "Lower values save VRAM.")
    p.add_argument("--sliding_window", action="store_true",
                   help="When --max_slices is active, average gradients over "
                        "multiple offset versions of the same slice budget.")
    p.add_argument("--n_ensemble_models", type=int, default=1,
                   help="Ensemble members used per PGD step (0 = all).")
    p.add_argument("--n_ensemble_load", type=int, default=1,
                   help="Ensemble members kept in RAM after loading (0 = all 5). "
                        "Each member is ~130 MB.")
    p.add_argument("--clf_spatial_size", nargs=2, type=int, default=[128, 128],
                   help="(H W) volume size for the classifier. Sybil was "
                        "trained on 256 256; 128 128 uses ~4x less memory "
                        "and is enough for adversarial gradients.")
    p.add_argument("--pad_depth", action="store_true",
                   help="Pad depth to 200 (Sybil's training shape). Off by "
                        "default - uses the actual slice count.")
    p.add_argument("--classifier_device",
                   choices=["cuda", "cpu"], default="cuda",
                   help="Where to place the classifier. cuda = fast but uses "
                        "VRAM; cpu = slow but always works.")

    # --- Output ------------------------------------------------------------
    p.add_argument("--output_ply", required=True,
                   help="Where to write the attacked PLY.")
    p.add_argument("--sh_degree", type=int, default=3,
                   help="Spherical harmonics degree "
                        "(auto-detected from PLY when possible).")

    return p.parse_known_args()
