#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-time environment setup for MedGS on a SLURM cluster.
#
# USAGE (run INTERACTIVELY, not as a batch job):
#
#   # 1. Request a GPU node so the build can see the GPU drivers:
#   srun --gres=gpu:1 --cpus-per-task=4 --mem=16G --pty bash
#
#   # 2. If conda is not on PATH by default, load the module first, e.g.:
#   #   module load miniconda3   (cluster-specific — check `module avail`)
#
#   # 3. Run this script:
#   bash setup.sh
#
# After the first run, subsequent jobs should source the activation block
# inside their SLURM batch scripts (see the comment at the bottom of this file).
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
ENV_NAME="medgs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse arguments ───────────────────────────────────────────────────────────
# Usage:  bash setup.sh [--cuda X.Y]
#   --cuda X.Y   CUDA version to use (default: 12.4 — safe for clusters)
#                Use 12.8 for local builds on Blackwell GPUs (RTX 5060 Ti etc.)
#
# CUDA version → PyTorch installation method:
#   11.8, 12.1, 12.4  →  conda (pytorch-cuda=X.Y, pytorch channel)
#   12.8              →  pip (https://download.pytorch.org/whl/cu128)
#                        PyTorch's conda channel caps at 12.4; pip has cu128
#                        wheels with native sm_120 support for Blackwell.
CUDA_VER="12.4"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) CUDA_VER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

VALID_CUDA_VERSIONS=("11.8" "12.1" "12.4" "12.8")
valid=0
for v in "${VALID_CUDA_VERSIONS[@]}"; do
    [[ "${CUDA_VER}" == "${v}" ]] && valid=1 && break
done
if [[ "${valid}" -eq 0 ]]; then
    echo "ERROR: --cuda ${CUDA_VER} is not supported." >&2
    echo "       Valid values: ${VALID_CUDA_VERSIONS[*]}" >&2
    exit 1
fi

echo ">>> CUDA version : ${CUDA_VER}"

# ── Derive TORCH_ARCH and PyTorch install method ──────────────────────────────
# CUDA 12.8+ supports sm_120 (Blackwell / RTX 5060 Ti).
if [[ "${CUDA_VER}" == "12.8" ]]; then
    TORCH_ARCH="7.0;7.5;8.0;8.6;9.0;12.0+PTX"
    PYTORCH_FROM_PIP=1
else
    TORCH_ARCH="7.0;7.5;8.0;8.6;9.0+PTX"
    PYTORCH_FROM_PIP=0
fi

# ── Substitute CUDA version into a temporary environment file ─────────────────
# For pip-pytorch builds, strip the conda pytorch lines (pytorch-cuda=X.Y doesn't
# exist for 12.8); pytorch is installed via pip after conda activate instead.
YML_TMP="$(mktemp /tmp/medgs_env_XXXXXX.yml)"
if [[ "${PYTORCH_FROM_PIP}" -eq 1 ]]; then
    sed "s/CUDA_VER/${CUDA_VER}/g" "${SCRIPT_DIR}/environment.yml" \
        | grep -v -E "^\s+- pytorch$|^\s+- torchvision$|^\s+- torchaudio$|^\s+- pytorch-cuda" \
        > "${YML_TMP}"
else
    sed "s/CUDA_VER/${CUDA_VER}/g" "${SCRIPT_DIR}/environment.yml" > "${YML_TMP}"
fi
trap 'rm -f "${YML_TMP}"' EXIT
YML="${YML_TMP}"

# ── 1. Bootstrap conda shell functions ────────────────────────────────────────
# Works whether conda was loaded via `module load` or is already on PATH.
CONDA_BASE=$(conda info --base 2>/dev/null) || {
    echo "ERROR: 'conda' not found. Load the appropriate module first." >&2
    echo "       e.g.  module load miniconda3" >&2
    exit 1
}
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ── 2. Create or update the conda environment ─────────────────────────────────
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo ">>> Environment '${ENV_NAME}' already exists — updating from ${YML}..."
    conda env update -n "${ENV_NAME}" -f "${YML}" --prune
else
    echo ">>> Creating environment '${ENV_NAME}' from ${YML}..."
    conda env create -f "${YML}"
fi

# ── 3. Activate the environment ───────────────────────────────────────────────
# Conda's activation scripts (e.g. libblas_mkl_activate.sh) reference variables
# that may not be set yet, which trips the -u (nounset) flag.  Suspend it only
# for the activate call, then restore strict mode immediately after.
set +u
conda activate "${ENV_NAME}"
set -u

# ── 3a. Install PyTorch via pip for CUDA versions not in the pytorch channel ──
# The pytorch conda channel caps at CUDA 12.4. For 12.8 (Blackwell/sm_120)
# we need pip wheels which include native sm_120 kernels.
if [[ "${PYTORCH_FROM_PIP}" -eq 1 ]]; then
    CUDA_SHORT="${CUDA_VER/./}"   # e.g. 12.8 → 128
    echo ">>> Installing PyTorch via pip (cu${CUDA_SHORT})..."
    pip install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT}"
fi

# ── 3b. Install hermetic C/C++ compilers into the already-created env ─────────
# These cannot be in environment.yml because their activation scripts mis-fire
# during conda's internal pip subprocess, causing a build failure.
# Installing them here (after activate) sets the correct CONDA_PREFIX.
echo ">>> Installing hermetic GCC/G++ compilers..."
# conda-forge's gcc activation/deactivation scripts reference variables that
# may be unset, tripping the -u flag.  Suspend nounset for this call only.
set +u
conda install -y -c conda-forge "gcc_linux-64=11.*" "gxx_linux-64=11.*"
set -u

# ── 4. Lock in the hermetic CUDA environment variables ────────────────────────
# Always anchor CUDA_HOME to CONDA_PREFIX — cuda-toolkit installs there.
# Do NOT use `which nvcc` here: the system nvcc (different version) may appear
# on PATH first, causing cpp_extension.py to detect the wrong CUDA version.
[[ -f "${CONDA_PREFIX}/bin/nvcc" ]] || {
    echo "ERROR: nvcc not found inside the conda environment." >&2
    echo "       Check that 'cuda-toolkit=${CUDA_VER}' was installed successfully." >&2
    exit 1
}

export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_PATH="${CONDA_PREFIX}"

# Prepend conda's bin/ so that every tool (nvcc, gcc, ld, …) is found here
# BEFORE anything on the system PATH (overrides system CUDA 13.x, etc.).
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# Hermetic C/C++ compilers — conda's gcc_linux-64 activation script already
# sets $CC/$CXX via etc/conda/activate.d, but we export them explicitly here
# so that sub-processes spawned by pip/setuptools inherit the correct values.
export CC
CC=$(which x86_64-conda-linux-gnu-gcc 2>/dev/null || which gcc)
export CXX
CXX=$(which x86_64-conda-linux-gnu-g++ 2>/dev/null || which g++)

# Prevents distutils / setuptools from probing the system /usr/include first.
export C_INCLUDE_PATH="${CUDA_HOME}/include:${CONDA_PREFIX}/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${C_INCLUDE_PATH}"

# Target GPU architectures for torch.utils.cpp_extension.CUDAExtension.
export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}"

echo ""
echo ">>> Compiler sanity check:"
echo "    nvcc  : $(nvcc --version | grep release)"
echo "    gcc   : $(${CC} --version | head -1)"
echo "    g++   : $(${CXX} --version | head -1)"
echo "    CUDA_HOME=${CUDA_HOME}"

# ── 5. Initialise git submodules ──────────────────────────────────────────────
echo ""
echo ">>> Initialising git submodules..."
cd "${SCRIPT_DIR}"

# gitlab.inria.fr is intermittently unreachable. Override simple-knn to use a
# GitHub mirror that carries identical code.  This only affects the local clone
# URL; it does not modify the committed .gitmodules file.
git config submodule.submodules/simple-knn.url https://github.com/camenduru/simple-knn.git

git submodule update --init --recursive

# ── 6. Install CUDA submodules in editable mode ───────────────────────────────
# --no-build-isolation is CRITICAL: it forces pip to compile against *this*
# conda environment's torch headers and libcudart, instead of downloading a
# fresh (potentially version-mismatched) torch into an isolated build venv.
echo ""
echo ">>> Installing diff-gaussian-rasterization..."
pip install --no-build-isolation -e "${SCRIPT_DIR}/submodules/diff-gaussian-rasterization"

echo ""
echo ">>> Installing fused-ssim..."
# fused-ssim's setup.py calls torch.cuda.get_device_capability() at build time and
# passes -arch=sm_XX directly to nvcc — bypassing TORCH_CUDA_ARCH_LIST entirely.
# Hiding the GPU forces it into its fallback multi-arch path (sm_75/80/89).
CUDA_VISIBLE_DEVICES="" pip install --no-build-isolation -e "${SCRIPT_DIR}/submodules/fused-ssim"

echo ""
echo ">>> Installing simple-knn..."
# simple_knn has no __init__.py so the editable-install import hook can't resolve
# it as a package. Install normally (not -e) so pip copies the .so to site-packages.
pip install --no-build-isolation "${SCRIPT_DIR}/submodules/simple-knn"

# ── 7. Final smoke test ───────────────────────────────────────────────────────
echo ""
echo ">>> Smoke test..."
python - <<'EOF'
import warnings
warnings.filterwarnings("ignore")   # suppress sm_120 PTX-JIT warning on Blackwell GPUs
import torch, diff_gaussian_rasterization  # noqa: F401
from simple_knn._C import distCUDA2  # noqa: F401
print(f"  torch        : {torch.__version__}")
print(f"  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU           : {torch.cuda.get_device_name(0)}")
print("  diff_gaussian_rasterization : OK")
print("  simple_knn                  : OK")
EOF

echo ""
echo "========================================================"
echo " Setup complete!"
echo " Conda env  : ${ENV_NAME}"
echo " CUDA_HOME  : ${CUDA_HOME}"
echo " TORCH_ARCH : ${TORCH_ARCH}"
echo "========================================================"
echo ""
echo "To use this environment in a SLURM batch script, add:"
echo ""
echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\""
echo "  conda activate ${ENV_NAME}"
echo "  export CUDA_HOME=\"\${CONDA_PREFIX}\""
echo "  export CUDA_PATH=\"\${CONDA_PREFIX}\""
echo "  export PATH=\"\${CONDA_PREFIX}/bin:\${PATH}\""
echo "  export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${CONDA_PREFIX}/lib64:\${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:\${LD_LIBRARY_PATH:-}\""
echo "  export TORCH_CUDA_ARCH_LIST=\"${TORCH_ARCH}\""
echo ""
