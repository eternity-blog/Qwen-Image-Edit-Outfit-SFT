#!/usr/bin/env bash
# Wait for multi-GPU VITON-train IDM shards to finish, then merge + convert.
# All outputs stay under /data/agent/hf_models/datasets/qwen_vton/.
set -euo pipefail

ROOT="${ROOT:-/data/agent/lixiao29/QualityInspection-sync}"
SCRIPTS="$ROOT/Qwen-Image-Edit/scripts"
PY="${PY:-/root/conda-envs/idm-vton/bin/python}"

SYNTH="${SYNTH:-/data/agent/hf_models/datasets/qwen_vton/synth/idm_unpaired_train}"
CONV="${CONV:-/data/agent/hf_models/datasets/qwen_vton/converted_idm_synth_train}"
RAW_ROOT="${RAW_ROOT:-/data/agent/hf_models/datasets/qwen_vton/raw}"
LOG="$ROOT/outputs/qwen_vton_lora/logs/idm_train_merge_convert.log"
EXPECTED="${EXPECTED:-11647}"
TMUX_PREFIX="${TMUX_PREFIX:-lixiao-idm-train}"
POLL_SEC="${POLL_SEC:-60}"

mkdir -p "$(dirname "$LOG")" "$SYNTH" "$CONV"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] wait shards under $SYNTH (expect ~$EXPECTED)"

shard_sum() {
  local s=0 f n
  shopt -s nullglob
  for f in "$SYNTH"/manifest.shard*.jsonl; do
    n=$(wc -l < "$f")
    s=$((s + n))
  done
  echo "$s"
}

shards_running() {
  tmux ls 2>/dev/null | grep -q "^${TMUX_PREFIX}-s" || return 1
  pgrep -f "synthesize_unpaired_idm.py.*idm_unpaired_train" >/dev/null 2>&1
}

while true; do
  sum=$(shard_sum)
  running=0
  if shards_running; then running=1; fi
  echo "[$(date -Is)] shard_sum=$sum running=$running"

  if [[ "$running" -eq 0 ]]; then
    echo "[$(date -Is)] no train synth process; proceed with sum=$sum"
    break
  fi
  # Soft early exit if we already hit target and processes linger on last flush
  if [[ "$sum" -ge "$EXPECTED" ]]; then
    sleep 30
    sum2=$(shard_sum)
    if [[ "$sum2" -ge "$EXPECTED" ]] && ! shards_running; then
      echo "[$(date -Is)] target reached sum=$sum2"
      break
    fi
    if [[ "$sum2" -ge "$EXPECTED" ]]; then
      # still running but count complete — wait until processes exit
      :
    fi
  fi
  sleep "$POLL_SEC"
done

echo "[$(date -Is)] merge manifests"
bash "$SCRIPTS/merge_idm_shard_manifests.sh" "$SYNTH"
merged=$(wc -l < "$SYNTH/manifest.jsonl")
echo "[$(date -Is)] merged_lines=$merged"

echo "[$(date -Is)] convert -> $CONV"
"$PY" "$SCRIPTS/convert_idm_synth_to_qwen_edit.py" \
  --synth-dir "$SYNTH" \
  --raw-root "$RAW_ROOT" \
  --out-dir "$CONV"

echo "[$(date -Is)] DONE"
cat "$CONV/stats.json"
ls -la "$CONV" | head
# pointer for humans
cat > /data/agent/hf_models/datasets/qwen_vton/DATA_LAYOUT.txt <<EOF
qwen_vton data layout (under /data/agent/hf_models/datasets/qwen_vton)

raw/
  viton_hd/          original VITON-HD
  dresscode/         DressCode

synth/
  idm_unpaired/           test unpaired IDM synth (2032) + manifest.jsonl
  idm_unpaired_train/     train unpaired IDM synth (~11647) + manifest.jsonl

converted_idm_synth/       DiffSynth metadata from TEST synth
converted_idm_synth_train/ DiffSynth metadata from TRAIN synth  (this job)

Each converted_* has:
  metadata_train.json / .jsonl
  metadata_val.json / .jsonl
  stats.json
  dataset_base/  (symlinks to viton_hd + idm_synth images)
EOF
echo "[$(date -Is)] wrote DATA_LAYOUT.txt"
