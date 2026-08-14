#!/usr/bin/env bash
# Create conda env for Qwen-Image-Edit-2511.
# Heavy packages install on LOCAL DISK, exposed via symlink:
#   /data/agent/conda/envs/qwen-image-edit -> /root/conda-envs/qwen-image-edit
set -euo pipefail

LINK_PATH="${LINK_PATH:-/data/agent/conda/envs/qwen-image-edit}"
ENV_DIR="${ENV_DIR:-/root/conda-envs/qwen-image-edit}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CONDA_ROOT="${CONDA_ROOT:-/data/agent/conda}"
LOG="${LOG:-/data/agent/lixiao29/QualityInspection-sync/outputs/qwen_kf_zeroshot/setup_env.log}"

mkdir -p "$(dirname "$LOG")" /root/conda-envs
exec > >(tee -a "$LOG") 2>&1

export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"
export NO_PROXY="$no_proxy"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv-qwen-image-edit}"
export UV_LINK_MODE=copy
mkdir -p "$UV_CACHE_DIR"

echo "[$(date -Is)] setup start ENV_DIR=$ENV_DIR LINK_PATH=$LINK_PATH"

# If LINK_PATH is a real Ceph dir (not symlink), move it aside — do not install there.
if [[ -e "$LINK_PATH" && ! -L "$LINK_PATH" ]]; then
  bak="${LINK_PATH}.ceph.bak.$(date +%Y%m%d%H%M%S)"
  echo "moving Ceph env aside -> $bak"
  mv "$LINK_PATH" "$bak"
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  conda create -y -p "$ENV_DIR" "python=$PYTHON_VERSION"
fi

ln -sfn "$ENV_DIR" "$LINK_PATH"
ls -la "$LINK_PATH"

PY="$ENV_DIR/bin/python"
UV="${UV:-/root/.local/bin/uv}"
if [[ ! -x "$UV" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV="$HOME/.local/bin/uv"
fi

# PyTorch cu124: --no-deps then CUDA libs from PyPI.
if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "[$(date -Is)] installing torch+cu124"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install --no-deps \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
  "$UV" pip install --python "$PY" \
    filelock typing-extensions networkx jinja2 fsspec sympy \
    "nvidia-cuda-nvrtc-cu12==12.4.127" "nvidia-cuda-runtime-cu12==12.4.127" \
    "nvidia-cuda-cupti-cu12==12.4.127" "nvidia-cudnn-cu12==9.1.0.70" \
    "nvidia-cublas-cu12==12.4.5.8" "nvidia-cufft-cu12==11.2.1.3" \
    "nvidia-curand-cu12==10.3.5.147" "nvidia-cusolver-cu12==11.6.1.9" \
    "nvidia-cusparse-cu12==12.3.1.170" "nvidia-cusparselt-cu12==0.6.2" \
    "nvidia-nccl-cu12==2.21.5" "nvidia-nvtx-cu12==12.4.127" "nvidia-nvjitlink-cu12==12.4.127" \
    triton
fi

"$UV" pip install --python "$PY" \
  "git+https://github.com/huggingface/diffusers" \
  transformers accelerate safetensors pillow sentencepiece protobuf \
  opencv-python-headless numpy

"$PY" - <<'PY'
import torch, diffusers, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
from diffusers import QwenImageEditPlusPipeline
print("QwenImageEditPlusPipeline OK", QwenImageEditPlusPipeline)
PY

echo "[$(date -Is)] setup DONE"
