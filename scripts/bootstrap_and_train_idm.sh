#!/usr/bin/env bash
# Recreate qwen-image-edit env (lost after reboot) then start 4-GPU IDM LoRA train.
set -euo pipefail

ROOT="${ROOT:-/data/agent/lixiao29/QualityInspection-sync}"
SCRIPTS="$ROOT/Qwen-Image-Edit/scripts"
LOG="$ROOT/outputs/qwen_vton_lora/logs/bootstrap_and_train_idm.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] bootstrap start"
bash "$SCRIPTS/setup_env.sh"

ENV_DIR="${ENV_DIR:-/root/conda-envs/qwen-image-edit}"
# Prefer symlink path if present.
if [[ -x /data/agent/conda/envs/qwen-image-edit/bin/python ]]; then
  ENV_DIR=/data/agent/conda/envs/qwen-image-edit
fi
PY="$ENV_DIR/bin/python"
UV="${UV:-/root/.local/bin/uv}"
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-/data/agent/hf_models/modules/DiffSynth-Studio}"

export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"
export NO_PROXY="$no_proxy"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv-qwen-image-edit}"
export UV_LINK_MODE=copy

echo "[$(date -Is)] install DiffSynth + train deps"
"$UV" pip install --python "$PY" -e "$DIFFSYNTH_DIR" accelerate modelscope sentencepiece imageio peft

"$PY" - <<'PY'
import torch, accelerate, peft
from diffusers import QwenImageEditPlusPipeline
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("accelerate", accelerate.__version__, "peft", peft.__version__)
print("QwenImageEditPlusPipeline OK")
PY

echo "[$(date -Is)] launching 4-GPU train"
GPU_LIST="${GPU_LIST:-5,4,3,1}" NUM_PROCESSES=4 LORA_RANK=16 NUM_EPOCHS=1 \
  bash "$SCRIPTS/train_idm_lora_multigpu.sh"

echo "[$(date -Is)] bootstrap+train finished"
