#!/usr/bin/env bash
# Controlled counterpart to train_full_sft_zero3.sh: DiT-LoRA on the SAME v2 data.
#
# WHY THIS SCRIPT EXISTS
#   The earlier LoRA (scripts/train_idm_lora_multigpu.sh) was trained on v1 metadata
#   (2 English prompts, 76-223 chars) while the full SFT used v2 (one 1592-char
#   template). Evaluating both with live v2 prompts therefore put only the LoRA far
#   out of its training distribution, so that pair cannot answer "LoRA vs full-param".
#   This script holds the data fixed so the training method is the only変数.
#
# HELD CONSTANT vs train_full_sft_zero3.sh
#   metadata / dataset_base .... converted_idm_synth_train_v2 (identical files)
#   epochs ..................... 1
#   dataset_repeat ............. 1
#   max_pixels ................. 1048576
#   gradient checkpointing ..... on
#   zero_cond_t ................ on   (required by Qwen-Image-Edit-2511)
#   precision .................. bf16
#   NUM_PROCESSES .............. 8    -> effective batch 8, same 1427 optimizer steps
#
# NECESSARILY DIFFERENT (inherent to the method — state this in any writeup)
#   trainable params ........... DiT LoRA r16 (~0.118B) vs all of DiT (20.43B)
#   learning rate .............. 1e-4 (LoRA convention) vs 1e-5
#   distributed strategy ....... DDP (LoRA fits) vs DeepSpeed ZeRO-3 (full does not)
#
# Required env (configs/env.local.sh or exported):
#   MODEL_DIR, DIFFSYNTH_DIR, ENV_DIR, QWEN_VTON_DATA, OUTPUT_ROOT
#
# Usage:
#   source configs/env.local.sh
#   bash scripts/train_lora_v2_multigpu.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_env.sh"

DATA_ROOT_VTON="${QWEN_VTON_DATA:-${DATA_ROOT}/datasets/qwen_vton}"
METADATA="${METADATA:-$DATA_ROOT_VTON/converted_idm_synth_train_v2/metadata_train.json}"
DATASET_BASE="${DATASET_BASE:-$DATA_ROOT_VTON/converted_idm_synth_train_v2/dataset_base}"
INIT_MODEL_DIR="${INIT_MODEL_DIR:-$MODEL_DIR}"

OUT_ROOT="${LORA_V2_OUT:-$OUTPUT_ROOT/qwen_vton_lora_v2}"
LORA_OUT="${LORA_OUT:-$OUT_ROOT/lora_v2}"
FUSED_OUT="${FUSED_OUT:-$OUT_ROOT/fused}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LORA_OUT" "$LOG_DIR"
LOG="$LOG_DIR/train_lora_v2.log"

NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-16}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SAVE_STEPS="${SAVE_STEPS:-}"
GPU_LIST="${GPU_LIST:-}"
DO_FUSE="${DO_FUSE:-1}"
LORA_TARGETS="${LORA_TARGETS:-to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1}"

export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-True}"
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- GPU selection -----------------------------------------------------------
if [[ -z "$GPU_LIST" ]]; then
  GPU_LIST="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -n "$NUM_PROCESSES" | cut -d, -f1 | tr -d ' ' | paste -sd, -)"
fi
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
N_ACTUAL="$(echo "$GPU_LIST" | awk -F, '{print NF}')"
if [[ "$N_ACTUAL" -lt "$NUM_PROCESSES" ]]; then
  echo "ERROR: only $N_ACTUAL GPUs in list=$GPU_LIST, need NUM_PROCESSES=$NUM_PROCESSES"
  exit 1
fi
# DDP replicates the frozen base on every rank: ~55 GiB static + activations.
MIN_FREE_MIB="${MIN_FREE_MIB:-70000}"
for idx in ${GPU_LIST//,/ }; do
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx" | tr -d ' ')"
  if [[ "$free" -lt "$MIN_FREE_MIB" ]]; then
    echo "WARNING: GPU $idx has only ${free} MiB free (< ${MIN_FREE_MIB})."
    echo "         LoRA DDP still replicates the frozen 20B base per rank (~55 GiB)."
    echo "         Free the card, lower MAX_PIXELS, or raise MIN_FREE_MIB to silence."
  fi
done

# --- preflight ---------------------------------------------------------------
PY="${ENV_DIR}/bin/python"
ACCEL="${ENV_DIR}/bin/accelerate"
[[ -x "$PY" ]] || { echo "ERROR: missing python at ENV_DIR=$ENV_DIR"; exit 1; }
[[ -x "$ACCEL" ]] || { echo "ERROR: missing accelerate at $ACCEL"; exit 1; }
[[ -f "$METADATA" ]] || { echo "ERROR: missing metadata $METADATA"; exit 1; }
[[ -d "$DATASET_BASE" ]] || { echo "ERROR: missing dataset_base $DATASET_BASE"; exit 1; }
[[ -d "$INIT_MODEL_DIR" ]] || { echo "ERROR: missing INIT_MODEL_DIR=$INIT_MODEL_DIR"; exit 1; }
TRAIN_PY="$DIFFSYNTH_DIR/examples/qwen_image/model_training/train.py"
[[ -f "$TRAIN_PY" ]] || { echo "ERROR: DiffSynth train.py not found at $TRAIN_PY"; exit 1; }

TOK="$INIT_MODEL_DIR/tokenizer"
PROC="$INIT_MODEL_DIR/processor"
[[ -d "$PROC" ]] || PROC="$INIT_MODEL_DIR"

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
PROMPT_CHARS="$("$PY" -c "import json;d=json.load(open('$METADATA'));print(len(d[0]['prompt']))")"
EXPECTED_STEPS="$("$PY" -c "import math;print(math.ceil($N_SAMPLES/$NUM_PROCESSES))")"

EXTRA=()
[[ -n "$SAVE_STEPS" ]] && EXTRA+=(--save_steps "$SAVE_STEPS")
[[ "${ENABLE_TENSORBOARD_LOG:-1}" == "1" ]] && EXTRA+=(--enable_tensorboard_log)
if [[ "${ENABLE_WANDB_LOG:-0}" == "1" ]]; then
  EXTRA+=(--enable_wandb_log --wandb_project "${WANDB_PROJECT:-qwen-outfit-lora-v2}")
fi

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] LoRA-v2 controlled run start"
echo "  gpus=$CUDA_VISIBLE_DEVICES nproc=$NUM_PROCESSES rank=$LORA_RANK lr=$LR epochs=$NUM_EPOCHS"
echo "  init=$INIT_MODEL_DIR"
echo "  metadata=$METADATA"
echo "  samples=$N_SAMPLES prompt_chars=$PROMPT_CHARS expected_steps=$EXPECTED_STEPS"
echo "  out=$LORA_OUT"
if [[ "$PROMPT_CHARS" -lt 1000 ]]; then
  echo "  WARNING: prompt is only $PROMPT_CHARS chars — this looks like v1 metadata."
  echo "           The controlled comparison needs the v2 full-text prompts."
fi

cd "$DIFFSYNTH_DIR"
"$ACCEL" launch \
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
  --weight_decay "$WEIGHT_DECAY" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$LORA_OUT" \
  --lora_base_model "dit" \
  --lora_target_modules "$LORA_TARGETS" \
  --lora_rank "$LORA_RANK" \
  --use_gradient_checkpointing \
  --dataset_num_workers "${DATASET_NUM_WORKERS:-4}" \
  --find_unused_parameters \
  --zero_cond_t \
  "${EXTRA[@]}"

echo "[$(date -Is)] TRAIN DONE -> $LORA_OUT"
ls -lah "$LORA_OUT" | head -20

if [[ "$DO_FUSE" == "1" ]]; then
  echo "[$(date -Is)] fusing LoRA into a loadable model dir -> $FUSED_OUT"
  "$PY" "$QWEN_OUTFIT_ROOT/scripts/fuse_qwen_edit_lora.py" \
    --base-model "$INIT_MODEL_DIR" \
    --lora-path "$LORA_OUT" \
    --out-dir "$FUSED_OUT" \
    --lora-scale 1.0
  echo "[$(date -Is)] FUSE DONE -> $FUSED_OUT"
fi

FULL_SFT_DIR="${FULL_SFT_DIR:-$OUTPUT_ROOT/qwen_full_sft_fused}"
cat <<EOF

[$(date -Is)] NEXT — three-way eval. Run this on the machine that also holds the
full-SFT fused model, so base / lora_v2 / full_sft share one seed and one run.

  # 1) in-domain holdout (has GT). Drop --cpu-offload if a whole 80GB card is free.
  "\$ENV_DIR/bin/python" scripts/eval_viton_holdout.py \\
    --model base="\$MODEL_DIR" \\
    --model lora_v2=$FUSED_OUT \\
    --model full_sft=$FULL_SFT_DIR \\
    --out-dir "\$OUTPUT_ROOT/viton_holdout_3way" \\
    --n 6 --steps 40 --seed 0

  # 2) business domain (needs TestSet + an existing outfit_v2 run); one model per run
  CPU_OFFLOAD=1 MAX_SAMPLES=2 MODEL_B_LABEL="lora_v2 (r$LORA_RANK)" \\
    IDM_MODEL=$FUSED_OUT \\
    OUT_ROOT="\$OUTPUT_ROOT/case02_lora_v2" \\
    bash scripts/run_case02_v2_prompt_eval.sh

  # 3) metric visualisation
  "\$ENV_DIR/bin/python" scripts/visualize_metrics.py \\
    --eval-dir "\$OUTPUT_ROOT/viton_holdout_3way"

Keep seed/steps identical to the full SFT run or the comparison is void.
EOF
