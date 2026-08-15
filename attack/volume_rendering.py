"""
Render a CT-like 3D volume from Gaussian-splatted slices, then apply the
Sybil preprocessing pipeline (resize, depth crop/pad, normalization).

Important deviations from ``sybil_utils.apply_sybil_transforms``:
  * uint8 quantization is skipped (zero-gradient operation),
  * resample-by-spacing is skipped (renders have no DICOM spacing); a
    trilinear spatial resize is used instead.

The output volume sits inside the autograd graph, so gradients from the
classifier propagate back through preprocessing and the rasterizer into
the Gaussian parameter tensors.
"""
import os

import torch
import torch.nn.functional as F

from .constants import (
    SYBIL_MEAN,
    SYBIL_STD,
    SYBIL_TARGET_DEPTH,
    SYBIL_TARGET_SPATIAL,
)


def get_sorted_camera_indices(scene_cameras, deduplicate_mirrors=True):
    """Return camera indices ordered by slice number.

    Names with a mirror suffix (e.g. ``"0042_1"``) are skipped when
    ``deduplicate_mirrors`` is True; only the canonical slice (``"0042"``
    or ``"0042_0"``) is kept.
    """
    seen_stems = set()
    indexed = []

    for i, cam in enumerate(scene_cameras):
        name = getattr(cam, "image_name", str(i))
        stem_full = os.path.splitext(os.path.basename(str(name)))[0]

        if "_" in stem_full:
            stem_base, suffix = stem_full.split("_", 1)
            is_mirror = suffix not in ("0", "")
        else:
            stem_base = stem_full
            is_mirror = False

        if deduplicate_mirrors and is_mirror:
            continue
        if deduplicate_mirrors and stem_base in seen_stems:
            continue
        seen_stems.add(stem_base)

        try:
            key = int(stem_base)
        except ValueError:
            key = i
        indexed.append((key, i))

    indexed.sort(key=lambda x: x[0])
    return [i for _, i in indexed]


def croppad_depth_axis(volume, target_depth):
    """Symmetrically crop or zero-pad the ``D`` axis of a ``(B, C, D, H, W)`` tensor."""
    D = volume.shape[2]
    if D == target_depth:
        return volume
    if D < target_depth:
        pad_total = target_depth - D
        pl = pad_total // 2
        pr = pad_total - pl
        return F.pad(volume, (0, 0, 0, 0, pl, pr), mode="constant", value=0)
    cl = (D - target_depth) // 2
    return volume[:, :, cl:cl + target_depth, :, :]


def render_volume_for_classifier(scene_cameras, cam_indices, gaussians,
                                 pipe, background, gs_render_fn,
                                 spatial_size=None, pad_depth=True):
    """Render slices, stack them into a volume and apply Sybil preprocessing.

    Args:
        scene_cameras: MedGS train camera list.
        cam_indices:   indices into ``scene_cameras`` in slice order.
        gaussians:     MedGS ``GaussianModel``.
        pipe:          MedGS pipeline params.
        background:    background color tensor.
        gs_render_fn:  the MedGS rasterizer (pass ``gaussian_renderer.render``).
        spatial_size:  ``(H, W)`` target spatial size. ``None`` uses Sybil's
                       256x256; smaller values (e.g. 128x128) cut VRAM
                       quadratically while still giving useful gradients.
        pad_depth:     if True, crop/pad the volume to depth 200 (Sybil's
                       training shape). If False, keep the natural slice
                       count - saves memory and avoids running the CNN on
                       lots of zero-padded planes.

    Returns:
        ``(1, 3, D, H, W)`` tensor on the device of ``gaussians``,
        normalized to Sybil's mean/std, attached to the autograd graph.
    """
    if spatial_size is None:
        spatial_size = SYBIL_TARGET_SPATIAL

    slices = []
    for cam_idx in cam_indices:
        cam = scene_cameras[cam_idx]
        out = gs_render_fn(cam, gaussians, pipe, background)
        img = torch.clamp(out["render"], 0.0, 1.0)
        if img.shape[0] > 1:
            img = img.mean(dim=0, keepdim=True)
        slices.append(img)

    volume = torch.stack(slices, dim=1).unsqueeze(0)  # (1, 1, D, H, W)

    _, _, D, H, W = volume.shape
    tH, tW = spatial_size
    if (H, W) != (tH, tW):
        volume = F.interpolate(volume, size=(D, tH, tW),
                               mode="trilinear", align_corners=False)

    if pad_depth:
        volume = croppad_depth_axis(volume, SYBIL_TARGET_DEPTH)

    # Match Sybil normalization (skip uint8 quantization - it would zero the gradient).
    volume = volume * 255.0
    volume = (volume - SYBIL_MEAN) / SYBIL_STD

    if volume.shape[1] == 1:
        volume = volume.repeat(1, 3, 1, 1, 1)

    return volume
