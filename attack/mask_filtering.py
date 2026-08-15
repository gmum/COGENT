"""Select Gaussians that fall inside lesion masks across multiple camera views."""
import numpy as np
from PIL import Image

from .geometry import (
    build_3d_covariance,
    gaussians_intersect_mask,
    project_covariance_to_2d,
)


def find_gaussians_in_masks(xyz, scales, rot_angles, cameras,
                            white_threshold=128, sigma_factor=3.0):
    """Mark Gaussians whose 2D projection intersects any view's mask.

    Args:
        xyz:             ``(N, 3)`` world-space centers.
        scales:          ``(N, K)`` raw log-scales.
        rot_angles:      ``(N,)`` raw rotation logits.
        cameras:         list of camera dicts from ``load_cameras_with_masks``.
        white_threshold: pixel value cutoff (0-255) for treating a mask
                         pixel as "masked".
        sigma_factor:    radius (in standard deviations) for the
                         Mahalanobis intersection test.

    Returns:
        ``(N,)`` boolean array - True for Gaussians hit by at least one mask.
    """
    N = len(xyz)
    print("\nBuilding 3D covariances...")
    cov3d = build_3d_covariance(scales, rot_angles)

    hit = np.zeros(N, dtype=bool)
    n_empty = 0

    for i, cam in enumerate(cameras):
        mask_img = Image.open(cam["mask_path"]).convert("L")
        mW, mH = mask_img.size
        if (mW, mH) != (cam["width"], cam["height"]):
            mask_img = mask_img.resize((cam["width"], cam["height"]), Image.NEAREST)
        mask = np.array(mask_img) >= white_threshold

        if mask.any():
            cov2d, cx_px, cy_px, valid = project_covariance_to_2d(
                xyz, cov3d, cam["w2c"], cam["fx"], cam["fy"],
                cam["width"], cam["height"])
            hit |= gaussians_intersect_mask(
                cov2d, cx_px, cy_px, valid, mask, sigma_factor)
        else:
            n_empty += 1

        if (i + 1) % 10 == 0 or i == len(cameras) - 1:
            print(f"  [{i + 1:4d}/{len(cameras)}] total hits = {hit.sum()}/{N}")

    print(f"\nMask filtering: {hit.sum()}/{N} (empty masks: {n_empty})")
    return hit


def remove_largest_gaussians(hit, scales, remove_top_pct):
    """Drop the largest-area Gaussians from a hit set.

    Area is approximated as ``exp(scale_x) * exp(scale_z)``. The function
    removes the top ``remove_top_pct`` percent of the currently selected
    Gaussians and returns a fresh boolean mask. If ``remove_top_pct <= 0``
    the input is returned unchanged.
    """
    if remove_top_pct <= 0.0:
        return hit

    sx = np.exp(scales[:, 0].astype(np.float64))
    sz = np.exp(scales[:, -1].astype(np.float64))
    area = sx * sz

    hit_indices = np.where(hit)[0]
    if len(hit_indices) == 0:
        return hit

    threshold = np.percentile(area[hit_indices], 100.0 - remove_top_pct)
    too_large = hit & (area > threshold)
    print(f"  Removed {too_large.sum()} largest, kept {(hit & ~too_large).sum()}")
    return hit & ~too_large
