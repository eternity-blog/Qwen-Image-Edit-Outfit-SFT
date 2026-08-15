#!/usr/bin/env bash
# Full-parameter DiT SFT for Qwen-Image-Edit-2511 via DiffSynth + Accelerate DeepSpeed ZeRO.
#
# Default: ZeRO-3, no offload, 8 GPUs (≈8×80GB).
# Fewer GPUs / OOM: DS_PROFILE=zero2_offload NUM_PROCESSES=4
#
# Required env (via configs/env.local.sh or export):
#   MODEL_DIR, DIFFSYNTH_DIR, ENV_DIR, QWEN_VTON_DATA, OUTPUT_ROOT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_env.sh"

DATA_ROOT_VTON="${QWEN_VTON_DATA:-${DATA_ROOT}/datasets/qwen_vton}"
METADATA="${METADATA:-$DATA_ROOT_VTON/converted_idm_synth_train_v2/metadata_train.json}"
DATASET_BASE="${DATASET_BASE:-$DATA_ROOT_VTON/converted_idm_synth_train_v2/dataset_base}"
# Init weights: base, or a LoRA-fused full model directory
INIT_MODEL_DIR="${INIT_MODEL_DIR:-$MODEL_DIR}"

OUT_ROOT="${FULL_SFT_OUT:-$OUTPUT_ROOT/qwen_vton_full_sft}"
CKPT_OUT="${CKPT_OUT:-$OUT_ROOT/dit_full}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$CKPT_OUT" "$LOG_DIR"
LOG="$LOG_DIR/train_full_sft.log"

NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-1e-5}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
SAVE_STEPS="${SAVE_STEPS:-}"          # empty => save each epoch
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GPU_LIST="${GPU_LIST:-}"
# zero3 | zero2_offload
DS_PROFILE="${DS_PROFILE:-zero3}"

export http_proxy="${http_proxy:-${HTTP_PROXY:-}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY:-}}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-True}"
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

if [[ -z "$GPU_LIST" ]]; then
  GPU_LIST="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -n "$NUM_PROCESSES" | cut -d, -f1 | tr -d ' ' | paste -sd, -)"
fi
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
N_ACTUAL="$(echo "$GPU_LIST" | awk -F, '{print NF}')"
if [[ "$N_ACTUAL" -lt "$NUM_PROCESSES" ]]; then
  echo "ERROR: only $N_ACTUAL free GPUs in list=$GPU_LIST, need NUM_PROCESSES=$NUM_PROCESSES"
  exit 1
fi

PY="${ENV_DIR}/bin/python"
ACCEL="${ENV_DIR}/bin/accelerate"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: missing python at ENV_DIR=$ENV_DIR"
  exit 1
fi
if [[ ! -x "$ACCEL" ]]; then
  echo "ERROR: missing accelerate at $ACCEL (pip install accelerate deepspeed)"
  exit 1
fi
"$PY" -c "import deepspeed" 2>/dev/null || {
  echo "ERROR: deepspeed not installed in $ENV_DIR"
  echo "  $PY -m pip install deepspeed"
  exit 1
}
[[ -f "$METADATA" ]] || { echo "ERROR: missing metadata $METADATA"; exit 1; }
[[ -d "$DATASET_BASE" ]] || { echo "ERROR: missing dataset_base $DATASET_BASE"; exit 1; }
[[ -d "$DIFFSYNTH_DIR/examples/qwen_image/model_training" ]] || {
  echo "ERROR: DiffSynth train entry missing under $DIFFSYNTH_DIR"
  exit 1
}
[[ -d "$INIT_MODEL_DIR" ]] || { echo "ERROR: missing INIT_MODEL_DIR=$INIT_MODEL_DIR"; exit 1; }

# Generate accelerate config with correct process count
case "$DS_PROFILE" in
  zero3)
    TEMPLATE="$QWEN_OUTFIT_ROOT/configs/accelerate_zero3.yaml"
    ;;
  zero2_offload)
    TEMPLATE="$QWEN_OUTFIT_ROOT/configs/accelerate_zero2_offload.yaml"
    ;;
  *)
    echo "ERROR: DS_PROFILE must be zero3 or zero2_offload (got $DS_PROFILE)"
    exit 1
    ;;
esac
ACCEL_CFG="$OUT_ROOT/accelerate_${DS_PROFILE}_n${NUM_PROCESSES}.yaml"
"$PY" - "$TEMPLATE" "$ACCEL_CFG" "$NUM_PROCESSES" <<'PY'
import sys
from pathlib import Path
src, dst, n = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
text = src.read_text()
out = []
for line in text.splitlines(True):
    if line.startswith("num_processes:"):
        out.append(f"num_processes: {n}\n")
    else:
        out.append(line)
dst.write_text("".join(out))
print("wrote", dst)
PY

TOK="$INIT_MODEL_DIR/tokenizer"
PROC="$INIT_MODEL_DIR/processor"
# Edit-2511 may ship processor under model root or share with Qwen-Image-Edit
if [[ ! -d "$PROC" ]]; then
  PROC="$INIT_MODEL_DIR"
fi

MODEL_PATHS_JSON="$("$PY" - <<PY
import json, glob
from pathlib import Path
md = Path("$INIT_MODEL_DIR")
tr = sorted(glob.glob(str(md / "transformer" / "diffusion_pytorch_model*.safetensors")))
te = sorted(glob.glob(str(md / "text_encoder" / "model*.safetensors")))
vae = sorted(glob.glob(str(md / "vae" / "diffusion_pytorch_model*.safetensors")))
assert tr and te and vae, (md, len(tr), len(te), len(vae))
print(json.dumps([tr, te, vae[0] if len(vae) == 1 else vae]))
PY
)"

N_SAMPLES="$("$PY" -c "import json;print(len(json.load(open('$METADATA'))))")"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] FULL DiT SFT start"
echo "  gpus=$CUDA_VISIBLE_DEVICES nproc=$NUM_PROCESSES profile=$DS_PROFILE"
echo "  init=$INIT_MODEL_DIR"
echo "  metadata=$METADATA samples=$N_SAMPLES"
echo "  dataset_base=$DATASET_BASE"
echo "  out=$CKPT_OUT lr=$LR epochs=$NUM_EPOCHS"
echo "  accel_cfg=$ACCEL_CFG"

EXTRA_SAVE=()
if [[ -n "$SAVE_STEPS" ]]; then
  EXTRA_SAVE+=(--save_steps "$SAVE_STEPS")
fi

# Extra train.py args (logging). TensorBoard on by default so the loss curve is
# persisted locally at $CKPT_OUT/tensorboard_log (view: tensorboard --logdir=...).
# Backfill to wandb afterwards via scripts/logs_to_wandb.py (reads these events).
EXTRA_TRAIN_ARGS=()
if [[ "${ENABLE_TENSORBOARD_LOG:-1}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--enable_tensorboard_log)
fi
if [[ "${ENABLE_WANDB_LOG:-0}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--enable_wandb_log --wandb_project "${WANDB_PROJECT:-qwen-outfit-full-sft}")
fi

cd "$DIFFSYNTH_DIR"
"$ACCEL" launch \
  --config_file "$ACCEL_CFG" \
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
  --weight_decay "$WEIGHT_DECAY" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$CKPT_OUT" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers "${DATASET_NUM_WORKERS:-4}" \
  --find_unused_parameters \
  --zero_cond_t \
  "${EXTRA_SAVE[@]}" \
  "${EXTRA_TRAIN_ARGS[@]}"

echo "[$(date -Is)] TRAIN DONE -> $CKPT_OUT"
ls -lah "$CKPT_OUT" | head -20
echo "Next: apply DiT ckpt into a full model dir:"
echo "  python scripts/apply_full_dit_ckpt.py --base-model \"\$MODEL_DIR\" --ckpt \"$CKPT_OUT/epoch-0.safetensors\" --out-dir \"\$OUTPUT_ROOT/qwen_full_sft_fused\""
