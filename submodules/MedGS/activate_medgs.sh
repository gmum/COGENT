#!/usr/bin/env bash
# Source this file to activate the medgs environment with all required variables:
#
#   source activate_medgs.sh
#
# Works both locally and in SLURM batch scripts.

CONDA_BASE=$(conda info --base 2>/dev/null) || {
    echo "ERROR: conda not found. Load the appropriate module first." >&2
    return 1
}
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate medgs

export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_PATH="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0+PTX"
