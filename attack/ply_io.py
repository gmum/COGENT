"""Read raw Gaussian attributes from a MedGS .ply file (no MedGS dependency)."""
import numpy as np
from plyfile import PlyData


def read_ply_attributes(ply_path):
    """Load xyz, scales and rotation angles from a MedGS Gaussian PLY.

    Args:
        ply_path: path to ``point_cloud.ply``.

    Returns:
        xyz:        ``(N, 3)`` float32, world-space positions.
        scales:     ``(N, K)`` float32, raw (log) scales for each axis.
        rot_angles: ``(N,)``   float32, raw rotation parameter (logit of theta).
    """
    plydata = PlyData.read(ply_path)
    v = plydata.elements[0]

    xyz = np.column_stack([
        np.asarray(v["x"]),
        np.asarray(v["y"]),
        np.asarray(v["z"]),
    ]).astype(np.float32)

    scale_names = sorted(
        [p.name for p in v.properties if p.name.startswith("scale_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    scales = np.column_stack(
        [np.asarray(v[s]) for s in scale_names]
    ).astype(np.float32)

    rot_angles = np.asarray(v["rot_0"]).astype(np.float32)

    print(f"  Gaussians in PLY: {len(xyz)}, scale fields: {scale_names}")
    return xyz, scales, rot_angles


def detect_sh_degree_from_ply(ply_path):
    """Infer the spherical-harmonics degree from the count of ``f_rest_*`` fields.

    A degree ``d`` SH expansion has ``3 * ((d + 1) ** 2 - 1)`` f_rest
    channels. Returns the detected degree, or ``None`` if it cannot be
    inferred (corrupt PLY, missing fields, etc.).
    """
    try:
        v = PlyData.read(ply_path).elements[0]
        n = sum(1 for p in v.properties if p.name.startswith("f_rest_"))
        sh = int(round((n / 3 + 1) ** 0.5)) - 1
        if 3 * ((sh + 1) ** 2 - 1) == n and sh >= 0:
            return sh
    except Exception:
        pass
    return None
