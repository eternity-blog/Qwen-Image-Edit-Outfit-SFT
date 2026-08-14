#!/usr/bin/env bash
# Train Qwen-Image-Edit-2511 LoRA on converted VTON data via DiffSynth-Studio,
# then fuse (拼接) LoRA weights into base DiT and save a fused checkpoint.
set -euo pipefail

ROOT="${ROOT:-/data/agent/lixiao29/QualityInspection-sync}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/qwen_vton_lora}"
DATA_ROOT="${DATA_ROOT:-/data/agent/hf_models/datasets/qwen_vton}"
MODEL_DIR="${MODEL_DIR:-/data/agent/hf_models/Qwen/Qwen-Image-Edit-2511}"
ENV_DIR="${ENV_DIR:-/data/agent/conda/envs/qwen-image-edit}"
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-/data/agent/hf_models/modules/DiffSynth-Studio}"
LORA_OUT="${LORA_OUT:-$OUT_ROOT/lora}"
FUSED_OUT="${FUSED_OUT:-/data/agent/hf_models/Qwen/Qwen-Image-Edit-2511-vton-lora-fused}"

NUM_EPOCHS="${NUM_EPOCHS:-1}"
LORA_RANK="${LORA_RANK:-16}"
LR="${LR:-1e-4}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
METADATA="${METADATA:-$DATA_ROOT/converted/metadata_train_subset4k.json}"
DATASET_BASE="${DATASET_BASE:-$DATA_ROOT/raw}"

mkdir -p "$OUT_ROOT" "$LORA_OUT" "$OUT_ROOT/logs"
LOG="$OUT_ROOT/logs/train_fuse.log"

export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"
export NO_PROXY="$no_proxy"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv-qwen-image-edit}"
# Prefer local files; DiffSynth still needs path= for skip-download mode.
export DIFFSYNTH_SKIP_DOWNLOAD=True

if [[ "${AUTO_GPU:-1}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] train+fuse start gpu=$CUDA_VISIBLE_DEVICES"

PY="$ENV_DIR/bin/python"
UV="${UV:-/root/.local/bin/uv}"
TOK="$MODEL_DIR/tokenizer"
PROC="$MODEL_DIR/processor"

# --- DiffSynth-Studio ---
if [[ ! -f "$DIFFSYNTH_DIR/pyproject.toml" ]]; then
  mkdir -p "$(dirname "$DIFFSYNTH_DIR")"
  rm -rf "$DIFFSYNTH_DIR"
  git clone --depth 1 https://github.com/modelscope/DiffSynth-Studio.git "$DIFFSYNTH_DIR"
fi
"$UV" pip install --python "$PY" -e "$DIFFSYNTH_DIR" accelerate modelscope sentencepiece imageio peft

if [[ ! -f "$METADATA" ]]; then
  echo "ERROR: missing metadata $METADATA"
  exit 1
fi

# DiffSynth --model_paths must be a JSON *string* (not a file path).
# Nested lists group sharded safetensors into one ModelConfig.
MODEL_PATHS_JSON="$("$PY" - <<PY
import json, glob
from pathlib import Path
md = Path("$MODEL_DIR")
tr = sorted(glob.glob(str(md / "transformer" / "diffusion_pytorch_model*.safetensors")))
te = sorted(glob.glob(str(md / "text_encoder" / "model*.safetensors")))
vae = sorted(glob.glob(str(md / "vae" / "diffusion_pytorch_model*.safetensors")))
assert tr and te and vae, (len(tr), len(te), len(vae))
print(json.dumps([tr, te, vae[0] if len(vae)==1 else vae]))
PY
)"
echo "[$(date -Is)] model_paths=$MODEL_PATHS_JSON"

echo "[$(date -Is)] launching LoRA training metadata=$METADATA samples=$("$PY" -c "import json;print(len(json.load(open('$METADATA'))))")"
cd "$DIFFSYNTH_DIR"
"$ENV_DIR/bin/accelerate" launch --num_processes 1 --mixed_precision bf16 \
  examples/qwen_image/model_training/train.py \
  --dataset_base_path "$DATASET_BASE" \
  --dataset_metadata_path "$METADATA" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels "$MAX_PIXELS" \
  --dataset_repeat "$DATASET_REPEAT" \
  --model_paths "$MODEL_PATHS_JSON" \
  --tokenizer_path "$TOK" \
  --processor_path "$PROC" \
  --learning_rate "$LR" \
  --num_epochs "$NUM_EPOCHS" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$LORA_OUT" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank "$LORA_RANK" \
  --use_gradient_checkpointing \
  --dataset_num_workers 4 \
  --find_unused_parameters \
  --zero_cond_t

echo "[$(date -Is)] fuse LoRA into base DiT (拼接权重)"
"$PY" "$ROOT/Qwen-Image-Edit/scripts/fuse_qwen_edit_lora.py" \
  --base-model "$MODEL_DIR" \
  --lora-path "$LORA_OUT" \
  --out-dir "$FUSED_OUT" \
  --lora-scale 1.0

echo "[$(date -Is)] train+fuse DONE"
ls -la "$LORA_OUT" | head
ls -la "$FUSED_OUT" | head
