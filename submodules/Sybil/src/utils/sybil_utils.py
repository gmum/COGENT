from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

SYBIL_MEAN = 128.1722
SYBIL_STD = 87.1849

TRUE_SYBIL_SPACING = (1.40625, 1.40625, 2.5)
TRUE_SYBIL_INPUT_SHAPE = (256, 256, 200)

SYBIL_WINDOW_CENTER = -600
SYBIL_WINDOW_WIDTH = 1500
SYBIL_MIN_HU = SYBIL_WINDOW_CENTER - SYBIL_WINDOW_WIDTH / 2 # -1350
SYBIL_MAX_HU = SYBIL_WINDOW_CENTER + SYBIL_WINDOW_WIDTH / 2 # 150


def apply_sybil_transforms(x: torch.Tensor, input_spacing: tuple[float, float, float]):
    # Input: HU range of shape (N, C, W, H, D)
    # Outputs: Normalized and resampled tensor of shape (N, C, D, W, H) and the resampled shape

    x = (x - SYBIL_WINDOW_CENTER) / SYBIL_WINDOW_WIDTH
    x = torch.clamp(x, -1/2, 1/2) # [-1/2, 1/2]
    x = (2*x + 1) / 2 # [0, 1]

        # Normalize
    x = x * 255 # Convert to [0, 255] range
    prev_dtype = x.dtype
    x = x.to(torch.uint8).to(prev_dtype) # Sybil expects quantized inputs
    x = (x - SYBIL_MEAN) / SYBIL_STD

    # Resample and CropPad
    x = _resample(x, current_spacing=input_spacing, target_spacing=TRUE_SYBIL_SPACING)
    resampled_shape = x.shape
    x = _croppad(x, target_shape=TRUE_SYBIL_INPUT_SHAPE)

    # Change to shape [N, C, T=D, W, H]
    x = x.permute(0, 1, 4, 2, 3)

    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1, 1)

    return x, resampled_shape


def _resample(x: torch.Tensor, current_spacing, target_spacing):
    # x: (B, C, W, H, D)
    N, C, W, H, D = x.shape
    shape = np.array([W, H, D])
    current_spacing = np.array(current_spacing, dtype=np.float32)
    target_spacing = np.array(target_spacing, dtype=np.float32)
    new_shape = (current_spacing * shape / target_spacing).astype(np.int64)
    x = F.interpolate(x, size=tuple(new_shape), mode='trilinear', align_corners=True)
    return x


def _croppad(x: torch.Tensor, target_shape):
    # x: (B, C, W, H, D)
    N, C, W, H, D = x.shape
    target_shape = np.array(target_shape)
    shape = np.array([W, H, D])
    diff = target_shape - shape
    pad = np.maximum(diff, 0)
    crop = np.maximum(-diff, 0)

    pl = pad // 2
    pr = pad - pl
    cl = crop // 2
    cr = crop - cl

    padding = (pl[2], pr[2], pl[1], pr[1], pl[0], pr[0])
    x = F.pad(x, padding, mode='constant', value=0)
    slices = (slice(cl[0], x.shape[2] - cr[0]), slice(cl[1], x.shape[3] - cr[1]), slice(cl[2], x.shape[4] - cr[2]))
    x = x[:, :, slices[0], slices[1], slices[2]]
    return x


def get_spacing(affine: np.ndarray) -> np.ndarray:
    x_scale = np.linalg.norm(affine[:,0])
    y_scale = np.linalg.norm(affine[:,1])
    z_scale = np.linalg.norm(affine[:,2])
    return np.array([x_scale, y_scale, z_scale])