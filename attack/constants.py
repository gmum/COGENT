"""Constants used by the Sybil preprocessing pipeline and the MedGS geometry."""

# Sybil normalization statistics (see SybilInference/sybil_utils.py).
SYBIL_MEAN = 128.1722
SYBIL_STD = 87.1849

# Sybil expects a (1, 3, 200, 256, 256) tensor.
SYBIL_TARGET_DEPTH = 200
SYBIL_TARGET_SPATIAL = (256, 256)  # (H, W)

# MedGS models 2D ellipses embedded in 3D (in-slice). The y-axis scale is
# collapsed to this epsilon; only x and z carry real variance.
EPS_SCALING_Y = 1e-8
