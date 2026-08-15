#!/usr/bin/env bash
# Batch-aware multi-GPU IDM-VTON synthesis.
#
# Generalises run_idm_train_multigpu.sh: the output dir and the pairs file are
# parameters, so additional batches (b2, b3, …) land in their own directories and
# can be uploaded to Hugging Face as separate, clearly labelled batches.
#
# Each GPU runs one contiguous shard; all shards write into <out>/images and their
# own manifest shard, merged at the end. Re-running skips already-written images,
# so an interrupted batch resumes safely.
#
# Usage:
#   BATCH_ID=b2 GPU_LIST=0,5 bash scripts/run_idm_synth_batch.sh
#
# Env:
#   BATCH_ID   batch suffix, default b2
#   GPU_LIST   comma list; empty => pick the freest NUM_GPUS cards
#   NUM_GPUS   used only when GPU_LIST is empty (default 2)
#   PAIRS      pairs file; default <out>/pairs_<BATCH_ID>.txt (make_pair_batch.py)
#   LIMIT      cap pairs for a smoke test (default 0 = all)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_env.sh"

BATCH_ID="${BATCH_ID:-b2}"
DATA_ROOT_VTON="${QWEN_VTON_DATA:-${DATA_ROOT}/datasets/qwen_vton}"
VITON="${VITON_ROOT:-$DATA_ROOT_VTON/raw/viton_hd}"
OUT="${SYNTH_OUT:-$DATA_ROOT_VTON/synth/idm_unpaired_train_$BATCH_ID}"
PAIRS="${PAIRS:-$OUT/pairs_$BATCH_ID.txt}"
LIMIT="${LIMIT:-0}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_LIST="${GPU_LIST:-}"
TMUX_PREFIX="${TMUX_PREFIX:-idm-synth-$BATCH_ID}"

IDM_PY="${IDM_PY:-/root/conda-envs/idm-vton/bin/python}"
IDM_REPO="${IDM_REPO_DIR:-$DATA_ROOT/modules/IDM-VTON}"
# setup_idm_vton.sh pulls the HF repo to <DATA_ROOT>/yisol/IDM-VTON, while
# lib_env.sh defaults IDM_MODEL_DIR to <DATA_ROOT>/models/IDM-VTON. Take whichever
# actually exists so both layouts work.
IDM_MODEL="${IDM_MODEL:-}"
if [[ -z "$IDM_MODEL" ]]; then
  for cand in "${IDM_MODEL_DIR:-}" "$DATA_ROOT/yisol/IDM-VTON" "$DATA_ROOT/models/IDM-VTON"; do
    if [[ -n "$cand" && -f "$cand/model_index.json" ]]; then
      IDM_MODEL="$cand"
      break
    fi
  done
fi

LOG_DIR="$OUTPUT_ROOT/qwen_vton_synth/logs"
mkdir -p "$OUT/images" "$LOG_DIR"

[[ -x "$IDM_PY" ]] || { echo "ERROR: IDM python not found at $IDM_PY (run scripts/setup_idm_vton.sh)"; exit 1; }
[[ -d "$IDM_REPO" ]] || { echo "ERROR: IDM repo not found at $IDM_REPO"; exit 1; }
[[ -n "$IDM_MODEL" && -d "$IDM_MODEL" ]] || {
  echo "ERROR: IDM-VTON weights not found. Looked for model_index.json under:"
  echo "  ${IDM_MODEL_DIR:-<IDM_MODEL_DIR unset>}"
  echo "  $DATA_ROOT/yisol/IDM-VTON"
  echo "  $DATA_ROOT/models/IDM-VTON"
  echo "Set IDM_MODEL=/path/to/IDM-VTON or run scripts/setup_idm_vton.sh"
  exit 1
}
[[ -d "$VITON/train/image" ]] || { echo "ERROR: VITON train images missing under $VITON"; exit 1; }
[[ -f "$PAIRS" ]] || {
  echo "ERROR: pairs file $PAIRS not found. Generate it first:"
  echo "  python scripts/make_pair_batch.py --viton-root $VITON \\"
  echo "    --prev $DATA_ROOT_VTON/synth/idm_unpaired_train \\"
  echo "    --out $PAIRS --batch-id $BATCH_ID --seed 2"
  exit 1
}

if [[ -z "$GPU_LIST" ]]; then
  GPU_LIST="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -n "$NUM_GPUS" | cut -d, -f1 | tr -d ' ' | paste -sd, -)"
fi
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
N="${#GPUS[@]}"
[[ "$N" -ge 1 ]] || { echo "ERROR: no GPUs selected"; exit 1; }

N_PAIRS="$(wc -l < "$PAIRS")"
echo "[$(date -Is)] IDM synth batch=$BATCH_ID"
echo "  pairs=$PAIRS ($N_PAIRS)"
echo "  out=$OUT"
echo "  gpus=$GPU_LIST shards=$N"
echo "  teacher=$IDM_MODEL"
# batch1 measured ~920 imgs/hour/GPU
echo "  ETA ~$(python3 -c "print(f'{$N_PAIRS/(920*$N):.1f}')")h at ~920 img/h/GPU"

for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  sess="${TMUX_PREFIX}-s${i}"
  log="$LOG_DIR/idm_${BATCH_ID}_shard${i}_gpu${gpu}.log"
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" \
    "export CUDA_VISIBLE_DEVICES=$gpu; \
     exec > >(tee -a '$log') 2>&1; \
     echo \"[\$(date -Is)] batch=$BATCH_ID shard=$i/$N gpu=$gpu\"; \
     '$IDM_PY' '$SCRIPT_DIR/synthesize_unpaired_idm.py' \
       --data-root '$VITON' \
       --pairs '$PAIRS' \
       --phase train \
       --unpaired \
       --limit $LIMIT \
       --num-shards $N \
       --shard-id $i \
       --batch-size 1 \
       --model-dir '$IDM_MODEL' \
       --repo-dir '$IDM_REPO' \
       --out-dir '$OUT' \
       --device cuda:0; \
     echo \"[\$(date -Is)] shard $i DONE\""
  echo "  started $sess on GPU $gpu -> $log"
done

cat > "$LOG_DIR/idm_${BATCH_ID}_launch.txt" <<EOF
status=running
batch_id=$BATCH_ID
gpus=$GPU_LIST
n_shards=$N
pairs=$PAIRS
n_pairs=$N_PAIRS
out=$OUT
tmux_prefix=$TMUX_PREFIX
merge=bash $SCRIPT_DIR/merge_idm_shard_manifests.sh $OUT
EOF

cat <<EOF

Watch:   tmux ls | grep $TMUX_PREFIX
Progress: ls $OUT/images | wc -l    (target $N_PAIRS)

When every shard prints DONE:
  bash $SCRIPT_DIR/merge_idm_shard_manifests.sh $OUT
  # then rebuild metadata over ALL batches (batch1 + this one):
  #   see docs/DATA_SCALING_PLAN.md
EOF
