#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
ENV_NAME="sybil_interpretability"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Usage: bash setup.sh [--cuda 11.8|13.0]
# Default is auto: choose 11.8 for older GPUs, 13.0 for newer GPUs.
CUDA_VER="auto"
while [[ $# -gt 0 ]]; do
	case "$1" in
		--cuda) CUDA_VER="$2"; shift 2 ;;
		*) echo "Unknown argument: $1" >&2; exit 1 ;;
	esac
done

if [[ "${CUDA_VER}" == "13" ]]; then
	CUDA_VER="13.0"
fi

if [[ "${CUDA_VER}" != "auto" && "${CUDA_VER}" != "11.8" && "${CUDA_VER}" != "13.0" ]]; then
	echo "ERROR: --cuda must be one of: 11.8, 13.0" >&2
	exit 1
fi

GPU_CC_RAW=""
if command -v nvidia-smi >/dev/null 2>&1; then
	GPU_CC_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
fi

if [[ "${CUDA_VER}" == "auto" ]]; then
	if [[ -n "${GPU_CC_RAW}" ]]; then
		GPU_CC_MAJOR="${GPU_CC_RAW%%.*}"
		if [[ "${GPU_CC_MAJOR}" -lt 7 ]]; then
			CUDA_VER="11.8"
		else
			CUDA_VER="13.0"
		fi
	else
		CUDA_VER="11.8"
	fi
fi

if [[ -n "${GPU_CC_RAW}" ]]; then
	GPU_CC_MAJOR="${GPU_CC_RAW%%.*}"
	if [[ "${CUDA_VER}" == "13.0" && "${GPU_CC_MAJOR}" -lt 7 ]]; then
		echo "WARNING: GPU compute capability ${GPU_CC_RAW} is old for the CUDA 13.0 stack."
		echo "         Prefer: bash setup.sh --cuda 11.8"
	fi
fi

echo "=== 1. Initializing Git Submodules ==="
cd "${SCRIPT_DIR}"

# Initialize top-level submodules first
git submodule update --init

# Traverse into MedGS to properly patch the nested simple-knn URL
cd submodules/MedGS
git config submodule.submodules/simple-knn.url https://github.com/camenduru/simple-knn.git
git submodule update --init --recursive
cd "${SCRIPT_DIR}"

echo "=== 2. Setting up Conda Environment ==="
# Bootstrap conda shell functions reliably
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda env update -f environment.yml --prune

# Temporarily disable strict unbound variable checking for Conda's hooks
set +u
conda activate "${ENV_NAME}"
set -u

echo "=== 3. Installing CUDA Toolkit (${CUDA_VER}) + PyTorch ==="
# Explicitly including cuda-nvcc ensures the compiler is in the environment 
conda install -y -c nvidia "cuda-toolkit=${CUDA_VER}" "cuda-nvcc=${CUDA_VER}" "cuda-cudart-dev=${CUDA_VER}" "cuda-cccl=${CUDA_VER}"

if [[ "${CUDA_VER}" == "11.8" ]]; then
	PYTORCH_CUDA_TAG="cu118"
else
	PYTORCH_CUDA_TAG="cu130"
fi
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA_TAG}"

# Torch's shared libraries need to be visible to native MedGS extensions.
TORCH_LIB_DIR="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"

echo "=== 4. Configuring Compiler Environment ==="
export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_PATH="${CONDA_PREFIX}"
export NVCC="${CONDA_PREFIX}/bin/nvcc"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# Force distutils to use the Conda headers instead of probing the host's /usr/include
# We explicitly add the targets/x86_64-linux/include path where Conda hides CCCL headers
# Force distutils to use the Conda headers instead of probing the host's /usr/include
# We explicitly add the targets/x86_64-linux/include path where Conda hides CCCL headers
export C_INCLUDE_PATH="${CUDA_HOME}/include:${CUDA_HOME}/targets/x86_64-linux/include:${CONDA_PREFIX}/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${C_INCLUDE_PATH}"
export CPATH="${CUDA_HOME}/include:${CUDA_HOME}/targets/x86_64-linux/include:${CONDA_PREFIX}/include:${CPATH:-}"

# Explicitly tell the linker where to find libcudart.so
export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/lib/stubs:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
# Restore the hermetic compilers from the original script
echo "Installing hermetic GCC/G++ compilers..."
# Suspend strict unbound variable checking for Conda-forge's compiler hooks
set +u
conda install -y -c conda-forge "gcc_linux-64=11.*" "gxx_linux-64=11.*"
set -u

# Point explicitly to the isolated Conda compilers, NOT the host system
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"

# Force distutils to use the Conda headers instead of probing the host's /usr/include
export C_INCLUDE_PATH="${CUDA_HOME}/include:${CONDA_PREFIX}/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${C_INCLUDE_PATH}"

# Build for the detected GPU architecture when possible; otherwise use a safe default.
GPU_ARCH="$(python - <<'PY'
import subprocess
try:
	import torch
	cc = torch.cuda.get_device_capability(0)
	print(f"{cc[0]}.{cc[1]}+PTX")
except Exception:
	try:
		out = subprocess.check_output([
			"nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"
		], text=True).splitlines()[0].strip()
		print(f"{out}+PTX")
	except Exception:
		print("7.5+PTX")
PY
)"
export TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"
echo "Using CUDA ${CUDA_VER} with TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

echo "=== 5. Building C++/CUDA Extensions ==="

# Install ninja for significantly faster, parallelized C++ builds
pip install ninja

# Patch the rasterizer to use C++17 instead of C++20 to bypass glibc rsqrt conflicts
echo "Patching diff-gaussian-rasterization for glibc compatibility..."
sed -i 's/-std=c++20/-std=c++17/g' "${SCRIPT_DIR}/submodules/MedGS/submodules/diff-gaussian-rasterization/setup.py" || true

# --no-build-isolation ensures pip uses the active environment's PyTorch/CUDA headers
echo "Building diff-gaussian-rasterization..."
pip install --no-build-isolation -e "${SCRIPT_DIR}/submodules/MedGS/submodules/diff-gaussian-rasterization"

echo "Building fused-ssim (with GPU hidden)..."
CUDA_VISIBLE_DEVICES="" pip install --no-build-isolation -e "${SCRIPT_DIR}/submodules/MedGS/submodules/fused-ssim"

echo "Building simple-knn..."
pip install --no-build-isolation "${SCRIPT_DIR}/submodules/MedGS/submodules/simple-knn"

echo "=== Setup Complete! ==="