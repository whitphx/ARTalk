#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX is not set. Run this through the artalk-web conda/mamba environment." >&2
  exit 1
fi

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
RASTERIZER_REPO_URL="${RASTERIZER_REPO_URL:-https://github.com/xg-chu/diff-gaussian-rasterization.git}"
RASTERIZER_DIR="${RASTERIZER_DIR:-.local-build/xgchu-diff-gaussian-rasterization}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "nvcc not found at ${CUDA_HOME}/bin/nvcc. Set CUDA_HOME to your CUDA toolkit path." >&2
  exit 1
fi

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  echo "TORCH_CUDA_ARCH_LIST is not set." >&2
  echo "Example for Tesla P100: TORCH_CUDA_ARCH_LIST=6.0" >&2
  exit 1
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available from this Python environment")
print(f"torch={torch.__version__}, torch_cuda={torch.version.cuda}")
PY

mkdir -p "$(dirname "${RASTERIZER_DIR}")"
if [[ ! -d "${RASTERIZER_DIR}/.git" ]]; then
  git clone "${RASTERIZER_REPO_URL}" "${RASTERIZER_DIR}"
fi

python -m pip install --no-build-isolation --no-deps --force-reinstall "${RASTERIZER_DIR}"

python - <<'PY'
import diff_gaussian_rasterization_32d
print("diff_gaussian_rasterization_32d import OK")
PY
