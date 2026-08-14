#!/usr/bin/env bash
# Multi-GPU sharded IDM synthesis on VITON-HD train unpaired pairs.
# Each GPU runs one shard; images share one out-dir; manifests are per-shard
# then merged. Skips already-written out_names (safe resume).
set -euo pipefail

ROOT="${ROOT:-/data/agent/lixiao29/QualityInspection-sync}"
SCRIPTS="$ROOT/Qwen-Image-Edit/scripts"
PY="${PY:-/root/conda-envs/idm-vton/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/agent/hf_models/datasets/qwen_vton/raw/viton_hd}"
SYNTH="${SYNTH:-/data/agent/hf_models/datasets/qwen_vton/synth/idm_unpaired_train}"
LOG_DIR="$ROOT/outputs/qwen_vton_lora/logs"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_LIST="${GPU_LIST:-}"   # e.g. "0,5,6,7"; empty => pick freest GPUs
TMUX_PREFIX="${TMUX_PREFIX:-lixiao-idm-train}"

mkdir -p "$SYNTH/images" "$LOG_DIR"

if [[ -z "$GPU_LIST" ]]; then
  GPU_LIST="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -n "$NUM_GPUS" | cut -d, -f1 | tr -d ' ' | paste -sd, -)"
fi
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
N="${#GPUS[@]}"
if [[ "$N" -lt 1 ]]; then
  echo "ERROR: no GPUs selected"
  exit 1
fi

echo "[$(date -Is)] launch train unpaired synth gpus=$GPU_LIST n_shards=$N out=$SYNTH"
echo "pairs=$(wc -l < "$DATA_ROOT/train_pairs.txt") (almost all cross-id)"

for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  sess="${TMUX_PREFIX}-s${i}"
  log="$LOG_DIR/idm_train_shard${i}_gpu${gpu}.log"
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" \
    "export CUDA_VISIBLE_DEVICES=$gpu; \
     exec > >(tee -a '$log') 2>&1; \
     echo \"[\$(date -Is)] shard=$i/$N gpu=$gpu\"; \
     '$PY' '$SCRIPTS/synthesize_unpaired_idm.py' \
       --data-root '$DATA_ROOT' \
       --pairs train_pairs.txt \
       --phase train \
       --unpaired \
       --limit 0 \
       --num-shards $N \
       --shard-id $i \
       --batch-size 1 \
       --model-dir /data/agent/hf_models/yisol/IDM-VTON \
       --repo-dir /data/agent/hf_models/modules/IDM-VTON \
       --out-dir '$SYNTH' \
       --device cuda:0; \
     echo \"[\$(date -Is)] shard $i DONE\""
  echo "  started $sess on GPU $gpu -> $log"
done

cat > "$LOG_DIR/idm_train_multigpu_launch.txt" <<EOF
status=running
gpus=$GPU_LIST
n_shards=$N
synth=$SYNTH
tmux_prefix=$TMUX_PREFIX
merge=bash $SCRIPTS/merge_idm_shard_manifests.sh $SYNTH
note=when all shards DONE, merge manifests then convert
EOF

echo "Watch: tmux ls | grep $TMUX_PREFIX"
echo "After all shards finish:"
echo "  bash $SCRIPTS/merge_idm_shard_manifests.sh $SYNTH"
echo "  then convert with convert_idm_synth_to_qwen_edit.py"
