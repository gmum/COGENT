"""
Geometric utilities for projecting 3D Gaussians into 2D camera views.

Provides:
  * ``build_world_to_camera``  - extrinsic from MedGS camera pose
  * ``build_3d_covariance``    - cov3d from MedGS raw scale / rotation params
  * ``project_covariance_to_2d`` - EWA-style projection to screen space
  * ``gaussians_intersect_mask`` - Mahalanobis disk vs. boolean mask test
"""
import numpy as np

from .constants import EPS_SCALING_Y


def build_world_to_camera(position, rotation_c2w):
    """Compose a ``(4, 4)`` world-to-camera matrix from a camera pose.

    ``rotation_c2w`` is the camera-to-world rotation as stored in MedGS's
    ``cameras.json``. We build the c2w extrinsic and invert it.
    """
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.array(rotation_c2w)
    c2w[:3, 3] = np.array(position)
    return np.linalg.inv(c2w).astype(np.float32)


def build_3d_covariance(scales_raw, rot_angles_raw):
    """Compute 3D covariance matrices from MedGS raw scale / rotation params.

    MedGS Gaussians are 2D ellipses embedded in 3D (in the slice plane).
    The y-axis variance is collapsed to ``EPS_SCALING_Y``; only the x and
    z axes carry real scale. The rotation parameter is a single logit
    recovered as ``theta = sigmoid(logit) * pi``.

    Args:
        scales_raw:     ``(N, K)`` raw log-scales; uses columns 0 and -1
                        as ``sx`` and ``sz``.
        rot_angles_raw: ``(N,)`` raw rotation logit.

    Returns:
        ``(N, 3, 3)`` float32 covariance matrices.
    """
    N = scales_raw.shape[0]
    sx = np.exp(scales_raw[:, 0])
    sy = np.full(N, EPS_SCALING_Y, dtype=np.float32)
    sz = np.exp(scales_raw[:, -1])

    theta = (1.0 / (1.0 + np.exp(-rot_angles_raw.astype(np.float64)))) * np.pi
    cos_t = np.cos(theta).astype(np.float32)
    sin_t = np.sin(theta).astype(np.float32)
    zeros = np.zeros(N, dtype=np.float32)
    ones = np.ones(N, dtype=np.float32)

    R = np.stack([
        np.stack([cos_t, zeros,  sin_t], axis=1),
        np.stack([zeros,  ones,  zeros], axis=1),
        np.stack([-sin_t, zeros, cos_t], axis=1),
    ], axis=1)

    S2 = np.stack([sx ** 2, sy ** 2, sz ** 2], axis=1)
    RS = R * S2[:, np.newaxis, :]
    return (RS @ R.transpose(0, 2, 1)).astype(np.float32)


def project_covariance_to_2d(xyz_world, cov3d, w2c, fx, fy, width, height):
    """Project 3D Gaussian centers and covariances to 2D screen space.

    Uses the linearized perspective projection (Jacobian ``J``) from the
    EWA splatting paper:

        cov2d = J W cov3d W^T J^T

    plus a low-pass term (``+0.3`` on the diagonal) matching what
    ``diff_gaussian_rasterization`` adds for numerical stability.

    Returns:
        cov2d:  ``(N, 2, 2)`` float32 screen-space covariances.
        cx_px:  ``(N,)`` float32 x pixel coords (``-1`` if behind camera).
        cy_px:  ``(N,)`` float32 y pixel coords.
        inside: ``(N,)`` bool, True iff the center is in front of the camera
                and within the image bounds.
    """
    N = xyz_world.shape[0]
    W = w2c[:3, :3].astype(np.float64)

    ones = np.ones((N, 1), dtype=np.float64)
    xyz_h = np.concatenate([xyz_world.astype(np.float64), ones], axis=1)
    cam = (w2c.astype(np.float64) @ xyz_h.T).T
    xc, yc, zc = cam[:, 0], cam[:, 1], cam[:, 2]

    valid = zc > 1e-4
    z_safe = np.where(valid, zc, 1.0)

    cx_px = np.where(valid, fx * xc / z_safe + width / 2.0, -1.0)
    cy_px = np.where(valid, fy * yc / z_safe + height / 2.0, -1.0)

    J = np.zeros((N, 2, 3), dtype=np.float64)
    J[:, 0, 0] = fx / z_safe
    J[:, 0, 2] = -fx * xc / (z_safe ** 2)
    J[:, 1, 1] = fy / z_safe
    J[:, 1, 2] = -fy * yc / (z_safe ** 2)

    T = J @ W[np.newaxis, :, :]
    cov2d = (T @ cov3d.astype(np.float64)) @ T.transpose(0, 2, 1)
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    inside = (valid
              & (cx_px >= 0) & (cx_px < width)
              & (cy_px >= 0) & (cy_px < height))

    return (cov2d.astype(np.float32),
            cx_px.astype(np.float32),
            cy_px.astype(np.float32),
            inside)


def gaussians_intersect_mask(cov2d, cx_px, cy_px, valid, mask, sigma_factor=3.0):
    """Test whether each 2D Gaussian overlaps a boolean image mask.

    For every valid Gaussian we:
      1. compute the principal-axis radius
         ``sqrt(lambda_max) * sigma_factor``
      2. walk pixels in the axis-aligned bounding box of the resulting disk
      3. mark the Gaussian as intersecting if any masked pixel falls
         within the Mahalanobis disk of radius ``sigma_factor``.

    Returns:
        ``(N,)`` boolean array.
    """
    N = cov2d.shape[0]
    H, W = mask.shape
    hit = np.zeros(N, dtype=bool)

    a = cov2d[:, 0, 0].astype(np.float64)
    b = cov2d[:, 0, 1].astype(np.float64)
    c = cov2d[:, 1, 1].astype(np.float64)
    det = a * c - b * b
    det_ok = np.abs(det) > 1e-6

    trace = a + c
    disc = np.sqrt(np.maximum((a - c) ** 2 + 4 * b * b, 0.0))
    lam_max = (trace + disc) / 2.0
    radius = np.sqrt(np.maximum(lam_max, 0.0)) * sigma_factor
    sq = sigma_factor ** 2

    for n in np.where(valid & det_ok)[0]:
        cx = float(cx_px[n]); cy = float(cy_px[n]); r = float(radius[n])
        if r < 0.5:
            continue
        x0 = max(0, int(np.floor(cx - r))); x1 = min(W - 1, int(np.ceil(cx + r)))
        y0 = max(0, int(np.floor(cy - r))); y1 = min(H - 1, int(np.ceil(cy + r)))
        if x0 > x1 or y0 > y1:
            continue
        sub_mask = mask[y0:y1 + 1, x0:x1 + 1]
        if not sub_mask.any():
            continue
        ys, xs = np.where(sub_mask)
        dx = (xs + x0) - cx
        dy = (ys + y0) - cy
        d = det[n]
        inv_a =  c[n] / d
        inv_b = -b[n] / d
        inv_c =  a[n] / d
        mah2 = inv_a * dx * dx + 2 * inv_b * dx * dy + inv_c * dy * dy
        if np.any(mah2 <= sq):
            hit[n] = True
    return hit
