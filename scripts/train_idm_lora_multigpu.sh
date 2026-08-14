#!/usr/bin/env bash
# Train Qwen-Image-Edit-2511 LoRA on IDM synthetic unpaired edit pairs (multi-GPU).
set -euo pipefail

ROOT="${ROOT:-/data/agent/lixiao29/QualityInspection-sync}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/qwen_vton_lora}"
DATA_ROOT="${DATA_ROOT:-/data/agent/hf_models/datasets/qwen_vton}"
MODEL_DIR="${MODEL_DIR:-/data/agent/hf_models/Qwen/Qwen-Image-Edit-2511}"
ENV_DIR="${ENV_DIR:-/root/conda-envs/qwen-image-edit}"
if [[ -x /data/agent/conda/envs/qwen-image-edit/bin/python ]]; then
  ENV_DIR=/data/agent/conda/envs/qwen-image-edit
fi
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-/data/agent/hf_models/modules/DiffSynth-Studio}"
LORA_OUT="${LORA_OUT:-$OUT_ROOT/lora_idm_train}"
FUSED_OUT="${FUSED_OUT:-/data/agent/hf_models/Qwen/Qwen-Image-Edit-2511-idm-lora-fused}"

NUM_EPOCHS="${NUM_EPOCHS:-1}"
LORA_RANK="${LORA_RANK:-16}"
LR="${LR:-1e-4}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
METADATA="${METADATA:-$DATA_ROOT/converted_idm_synth_train/metadata_train.json}"
DATASET_BASE="${DATASET_BASE:-$DATA_ROOT/converted_idm_synth_train/dataset_base}"
GPU_LIST="${GPU_LIST:-}"

mkdir -p "$OUT_ROOT" "$LORA_OUT" "$OUT_ROOT/logs"
LOG="$OUT_ROOT/logs/train_idm_lora_multigpu.log"

export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"
export NO_PROXY="$no_proxy"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TOKENIZERS_PARALLELISM=false

if [[ -z "$GPU_LIST" ]]; then
  GPU_LIST="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -n "$NUM_PROCESSES" | cut -d, -f1 | tr -d ' ' | paste -sd, -)"
fi
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
# Remap to local cuda:0..N-1 for accelerate
N_ACTUAL="$(echo "$GPU_LIST" | awk -F, '{print NF}')"
if [[ "$N_ACTUAL" -lt "$NUM_PROCESSES" ]]; then
  echo "ERROR: only $N_ACTUAL free GPUs, need $NUM_PROCESSES (list=$GPU_LIST)"
  exit 1
fi

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] train IDM LoRA start gpus=$CUDA_VISIBLE_DEVICES nproc=$NUM_PROCESSES rank=$LORA_RANK epochs=$NUM_EPOCHS"
echo "[$(date -Is)] metadata=$METADATA base=$DATASET_BASE"

PY="$ENV_DIR/bin/python"
TOK="$MODEL_DIR/tokenizer"
PROC="$MODEL_DIR/processor"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: missing python at $ENV_DIR"
  exit 1
fi
if [[ ! -f "$METADATA" ]]; then
  echo "ERROR: missing metadata $METADATA"
  exit 1
fi
if [[ ! -d "$DATASET_BASE" ]]; then
  echo "ERROR: missing dataset_base $DATASET_BASE"
  exit 1
fi

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

N_SAMPLES="$("$PY" -c "import json;print(len(json.load(open('$METADATA'))))")"
echo "[$(date -Is)] samples=$N_SAMPLES model_paths_ok"

cd "$DIFFSYNTH_DIR"
"$ENV_DIR/bin/accelerate" launch \
  --num_processes "$NUM_PROCESSES" \
  --multi_gpu \
  --mixed_precision bf16 \
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

echo "[$(date -Is)] fuse LoRA into base DiT -> $FUSED_OUT"
"$PY" "$ROOT/Qwen-Image-Edit/scripts/fuse_qwen_edit_lora.py" \
  --base-model "$MODEL_DIR" \
  --lora-path "$LORA_OUT" \
  --out-dir "$FUSED_OUT" \
  --lora-scale 1.0

echo "[$(date -Is)] TRAIN+FUSE DONE"
ls -lah "$LORA_OUT" | head
ls -lah "$FUSED_OUT" | head
