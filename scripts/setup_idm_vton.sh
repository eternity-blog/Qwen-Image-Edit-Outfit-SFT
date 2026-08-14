#!/usr/bin/env bash
# Setup IDM-VTON (teacher) for synthesizing unpaired try-on pairs.
# Env on LOCAL disk; weights under /data/agent/hf_models.
set -euo pipefail

LINK_ENV="${LINK_ENV:-/data/agent/conda/envs/idm-vton}"
ENV_DIR="${ENV_DIR:-/root/conda-envs/idm-vton}"
REPO_DIR="${REPO_DIR:-/data/agent/hf_models/modules/IDM-VTON}"
MODEL_DIR="${MODEL_DIR:-/data/agent/hf_models/yisol/IDM-VTON}"
LOG="${LOG:-/data/agent/lixiao29/QualityInspection-sync/outputs/qwen_vton_lora/logs/idm_setup.log}"
CONDA_ROOT="${CONDA_ROOT:-/data/agent/conda}"

mkdir -p "$(dirname "$LOG")" /root/conda-envs "$(dirname "$REPO_DIR")" "$(dirname "$MODEL_DIR")"
exec > >(tee -a "$LOG") 2>&1

export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"
export NO_PROXY="$no_proxy"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv-idm-vton}"
export UV_LINK_MODE=copy
mkdir -p "$UV_CACHE_DIR"

echo "[$(date -Is)] IDM-VTON setup start"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  conda create -y -p "$ENV_DIR" python=3.10
fi
ln -sfn "$ENV_DIR" "$LINK_ENV"

if [[ ! -f "$REPO_DIR/inference.py" ]]; then
  mkdir -p "$(dirname "$REPO_DIR")"
  rm -rf "$REPO_DIR"
  git clone --depth 1 https://github.com/yisol/IDM-VTON.git "$REPO_DIR"
fi

PY="$ENV_DIR/bin/python"
UV="${UV:-/root/.local/bin/uv}"
"$UV" pip install --python "$PY" pip

# PyTorch cu124: install wheels with --no-deps then CUDA libs from PyPI.
# (uv/pip cannot resolve nvidia-cudnn-cu12==9.1.0.70 from the pytorch index alone.)
if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "[$(date -Is)] installing torch+cu124"
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
  "diffusers==0.25.1" "transformers==4.36.2" "accelerate==0.26.1" \
  peft==0.7.1 einops opencv-python-headless pillow tqdm omegaconf \
  safetensors "huggingface_hub==0.20.3" sentencepiece protobuf scipy

if [[ ! -f "$MODEL_DIR/unet/config.json" ]]; then
  echo "[$(date -Is)] downloading yisol/IDM-VTON -> $MODEL_DIR"
  "$PY" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="yisol/IDM-VTON", local_dir="$MODEL_DIR")
print("done", "$MODEL_DIR")
PY
fi

mkdir -p /root/.cache/huggingface/hub
ln -sfn "$MODEL_DIR" "$REPO_DIR/ckpt_model" || true

echo "[$(date -Is)] IDM-VTON setup DONE"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
ls "$MODEL_DIR" | head
