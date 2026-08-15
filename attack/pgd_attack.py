"""
Projected Gradient Descent attack on MedGS Gaussian parameters using
gradients from the Sybil ensemble's raw logits.

Why raw logits (``out["logit"]``) instead of ``forward_all_years``:
  * ``sigmoid -> calibrate -> clip`` in the public API zeros the gradient
    in saturated regions and through the clip,
  * attacking the raw logit gives a clean, well-conditioned signal.

``target_year`` semantics:
  * ``0`` => year 1, ..., ``5`` => year 6 of the Sybil hazard table,
  * ``-1`` => mean over all six years.

Attack modes:
  * ``"untargeted"`` - maximize the chosen logit (push healthy -> sick),
  * ``"targeted"``   - pull the logit toward ``--target_value`` (MSE loss).
"""
import numpy as np
import torch
import torch.nn.functional as F

from .attribute_mapping import (
    build_attribute_map,
    get_attribute_slice,
    get_gradient_slice,
    set_attribute_slice_,
)
from .volume_rendering import (
    get_sorted_camera_indices,
    render_volume_for_classifier,
)


# Tensors on the MedGS ``GaussianModel`` that we may need to toggle
# ``requires_grad`` on. Order matters only for printing.
ALL_GAUSSIAN_TENSORS = (
    "_xyz", "_features_dc", "_features_rest",
    "_opacity", "_scaling", "_rotation",
)


def _select_ensemble_models(classifier, n_ensemble_models):
    """Pick which ensemble members to run during the forward pass."""
    if hasattr(classifier, "ensemble"):
        all_ensemble = list(classifier.ensemble)
    else:
        single = classifier.model if hasattr(classifier, "model") else classifier
        all_ensemble = [single]

    if 0 < n_ensemble_models < len(all_ensemble):
        return all_ensemble[:n_ensemble_models], len(all_ensemble)
    return all_ensemble, len(all_ensemble)


def _snapshot_originals(gaussians, attr_map, attacked_props):
    """Snapshot the original (pre-attack) values of attacked attribute slices."""
    originals = {}
    for prop in attacked_props:
        attr_name, sub_idx = attr_map[prop]
        with torch.no_grad():
            originals[prop] = get_attribute_slice(
                gaussians, attr_name, sub_idx).detach().clone()
    return originals


def _apply_random_start(gaussians, attr_map, attacked_props,
                        subset_mask, eps, clip):
    """Add uniform noise in ``[-eps, eps]`` to attacked attributes (PGD random start)."""
    with torch.no_grad():
        for prop in attacked_props:
            attr_name, sub_idx = attr_map[prop]
            cur = get_attribute_slice(gaussians, attr_name, sub_idx)
            noise = torch.empty_like(cur).uniform_(-eps, eps)
            noise = torch.where(subset_mask, noise, torch.zeros_like(noise))
            new = torch.clamp(cur + noise, clip[0], clip[1])
            set_attribute_slice_(gaussians, attr_name, sub_idx, new)


def _highlight_top_perturbed(gaussians, attr_map, attacked_props,
                             original_values, subset_mask, device,
                             top_pct=0.05, marker_value=9.0):
    """Set the top ``top_pct`` most-perturbed Gaussians to ``marker_value``.

    Debugging / visualization helper: it makes the strongest adversarial
    Gaussians easy to spot in the saved PLY by clamping their attacked
    attributes to a fixed high value.
    """
    N = gaussians._xyz.shape[0]
    print(f"\nMarking top {top_pct:.1f}% most-perturbed Gaussians with "
          f"{marker_value}...")
    with torch.no_grad():
        change_sq = torch.zeros(N, device=device)
        for prop in attacked_props:
            attr_name, sub_idx = attr_map[prop]
            d = get_attribute_slice(gaussians, attr_name, sub_idx) - original_values[prop]
            change_sq += d * d
        change_mag = torch.sqrt(change_sq)
        subset_vals = change_mag[subset_mask]
        if subset_vals.numel() == 0:
            return

        thr = torch.quantile(subset_vals, 1.0 - top_pct / 100.0).item()
        white = subset_mask & (change_mag >= thr)

        for prop in attacked_props:
            attr_name, sub_idx = attr_map[prop]
            cur = get_attribute_slice(gaussians, attr_name, sub_idx)
            cur = torch.where(white, torch.full_like(cur, marker_value), cur)
            set_attribute_slice_(gaussians, attr_name, sub_idx, cur)

        print(f"  Marked {int(white.sum().item())} Gaussians.")


def _log_step(step, steps, loss_val, score_val, prob_val,
              gaussians, attr_map, attacked_props, original_values,
              subset_mask, attack_mode, target_value):
    """Print one diagnostic line for a PGD iteration."""
    with torch.no_grad():
        pert_norms, max_pert = [], 0.0
        for prop in attacked_props:
            attr_name, sub_idx = attr_map[prop]
            cur = get_attribute_slice(gaussians, attr_name, sub_idx)
            d = (cur - original_values[prop])[subset_mask]
            pert_norms.append(d.abs().mean().item())
            m = d.abs().max().item() if d.numel() > 0 else 0.0
            if m > max_pert:
                max_pert = m
        avg_pert = float(np.mean(pert_norms)) if pert_norms else 0.0

    direction = "MAX" if attack_mode == "untargeted" else f"->{target_value}"
    print(f"  [PGD {step + 1:3d}/{steps}]  "
          f"loss={loss_val:.6f}  "
          f"logit={score_val:.4f} (prob={prob_val:.4f}, {direction})  "
          f"|pert|_avg={avg_pert:.6f}  |pert|_max={max_pert:.6f}")


def run_classifier_pgd_attack(gaussians, scene, pipe, background,
                              subset_mask_np, attacked_props,
                              classifier, gs_render_fn,
                              target_year, target_value,
                              eps, alpha, steps, attack_mode, clip,
                              device, max_slices=0, n_ensemble_models=1,
                              clf_spatial_size=None, pad_depth=False,
                              log_every=1):
    """Run PGD on selected Gaussian attributes using the classifier's logits.

    Gradient flow::

        logit <- SybilNet (r3d_18 backbone + hazard head)
              <- preprocessed volume (1, 3, D, H, W)
              <- rendered slices (diff_gaussian_rasterization)
              <- GaussianModel parameter tensors

    Args:
        gaussians:         MedGS ``GaussianModel``.
        scene:             MedGS ``Scene`` (provides train cameras).
        pipe, background:  passed to the rasterizer.
        subset_mask_np:    ``(N,)`` bool numpy mask - which Gaussians to attack.
        attacked_props:    PLY-style attribute names (e.g. ``"f_dc_0"``).
        classifier:        Sybil ensemble from ``load_sybil_classifier``.
        gs_render_fn:      MedGS render function.
        target_year:       ``0..5`` = year 1..6, ``-1`` = mean.
        target_value:      target logit value (only used in ``targeted`` mode).
        eps, alpha, steps: PGD radius, step size, number of iterations.
        attack_mode:       ``"untargeted"`` or ``"targeted"``.
        clip:              ``(low, high)`` absolute bounds on attacked values.
        device:            torch device of the Gaussians.
        max_slices:        max slices used per forward (``0`` = all).
        n_ensemble_models: how many ensemble members to use per step.
        clf_spatial_size:  ``(H, W)`` volume spatial size for the classifier.
        pad_depth:         pad/crop depth to Sybil's training depth (200).
        log_every:         print loss/perturbation stats every ``N`` steps.
    """
    N = gaussians._xyz.shape[0]
    attr_map = build_attribute_map(gaussians)
    for prop in attacked_props:
        if prop not in attr_map:
            raise ValueError(
                f"'{prop}' is not a known attribute. "
                f"Available: {sorted(attr_map.keys())}"
            )

    subset_mask = torch.from_numpy(subset_mask_np).to(device)
    n_attack = int(subset_mask.sum().item())

    print("\n=== PGD attack (gradient from SybilNet logits) ===")
    print(f"Gaussians attacked:  {n_attack} / {N}")
    print(f"Attributes:          {attacked_props}")
    print(f"Mode:                {attack_mode}")
    print(f"target_year:         {target_year} (0..5 = year 1..6)")
    if attack_mode == "targeted":
        print(f"target_value:        {target_value}")
    print(f"eps={eps}, alpha={alpha}, steps={steps}, clip={clip}")

    if n_attack == 0:
        print("No Gaussians selected for attack.")
        return

    original_values = _snapshot_originals(gaussians, attr_map, attacked_props)
    _apply_random_start(gaussians, attr_map, attacked_props,
                        subset_mask, eps, clip)

    scene_cameras = scene.getTrainCameras()
    cam_indices = get_sorted_camera_indices(scene_cameras, deduplicate_mirrors=True)
    print(f"Scene cameras:       {len(scene_cameras)}")
    print(f"Slices (deduped):    {len(cam_indices)}")

    if max_slices > 0 and len(cam_indices) > max_slices:
        idx_np = np.linspace(0, len(cam_indices) - 1, max_slices).round().astype(int)
        cam_indices = [cam_indices[i] for i in idx_np]
        print(f"Subsampled to:       {len(cam_indices)} slices "
              f"(max_slices={max_slices})")

    if not cam_indices:
        print("ERROR: no usable cameras.")
        return

    ensemble_models, total_members = _select_ensemble_models(
        classifier, n_ensemble_models)
    print(f"Ensemble used/total: {len(ensemble_models)} / {total_members}")

    attacked_attr_names = list({attr_map[p][0] for p in attacked_props})

    for step in range(steps):
        # Enable autograd only on attacked tensors.
        for attr_name in ALL_GAUSSIAN_TENSORS:
            if not hasattr(gaussians, attr_name):
                continue
            t = getattr(gaussians, attr_name)
            t.requires_grad_(attr_name in attacked_attr_names)
            if t.grad is not None:
                t.grad.zero_()

        volume = render_volume_for_classifier(
            scene_cameras, cam_indices, gaussians, pipe, background,
            gs_render_fn=gs_render_fn,
            spatial_size=clf_spatial_size, pad_depth=pad_depth,
        )

        # If the classifier lives on a different device, the cross-device
        # copy is recorded in autograd, so gradients still propagate.
        clf_device = next(ensemble_models[0].parameters()).device
        if volume.device != clf_device:
            volume = volume.to(clf_device)

        logits = [m(volume)["logit"] for m in ensemble_models]
        logit_mean = torch.stack(logits).mean(dim=0)  # (B, 6)

        if target_year < 0 or target_year >= logit_mean.shape[-1]:
            score = logit_mean.mean(dim=-1)
        else:
            score = logit_mean[:, target_year]

        if attack_mode == "untargeted":
            # Maximize the chosen logit -> minimize -logit.
            loss = -score.sum()
        else:
            target = torch.full_like(score, float(target_value))
            loss = F.mse_loss(score, target)

        loss.backward()

        loss_val = float(loss.detach().item())
        score_val = float(score.detach().mean().item())
        prob_val = float(score.detach().mean().sigmoid().item())

        del volume, logits, logit_mean, score, loss

        # PGD step (sign of gradient) + projection back into the eps ball + clip.
        with torch.no_grad():
            for prop in attacked_props:
                attr_name, sub_idx = attr_map[prop]
                t = getattr(gaussians, attr_name)
                if t.grad is None:
                    continue

                grad_slice = get_gradient_slice(t.grad, sub_idx)
                grad_slice = torch.where(subset_mask, grad_slice,
                                         torch.zeros_like(grad_slice))

                cur = get_attribute_slice(gaussians, attr_name, sub_idx)
                # Loss minimization: x <- x - alpha * sign(grad(loss)).
                #   untargeted (loss = -score): this raises score.
                #   targeted   (loss = MSE):    this drives score -> target.
                new = cur - alpha * grad_slice.sign()

                orig = original_values[prop]
                delta = torch.clamp(new - orig, -eps, eps)
                new = torch.clamp(orig + delta, clip[0], clip[1])
                new = torch.where(subset_mask, new, orig)
                set_attribute_slice_(gaussians, attr_name, sub_idx, new)

            for attr_name in attacked_attr_names:
                getattr(gaussians, attr_name).requires_grad_(False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (step + 1) % max(1, log_every) == 0 or step == 0 or step == steps - 1:
            _log_step(step, steps, loss_val, score_val, prob_val,
                      gaussians, attr_map, attacked_props, original_values,
                      subset_mask, attack_mode, target_value)

    _highlight_top_perturbed(
        gaussians, attr_map, attacked_props,
        original_values, subset_mask, device,
        top_pct=10.0, marker_value=9.0,
    )
