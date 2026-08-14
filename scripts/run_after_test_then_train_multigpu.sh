#!/usr/bin/env bash
# After VITON test unpaired (2032) finishes: convert test set, then launch
# multi-GPU VITON train unpaired synthesis.
set -euo pipefail

ROOT="${ROOT:-/data/agent/lixiao29/QualityInspection-sync}"
SCRIPTS="$ROOT/Qwen-Image-Edit/scripts"
PY="${PY:-/root/conda-envs/idm-vton/bin/python}"
TEST_SYNTH="${TEST_SYNTH:-/data/agent/hf_models/datasets/qwen_vton/synth/idm_unpaired}"
TEST_CONV="${TEST_CONV:-/data/agent/hf_models/datasets/qwen_vton/converted_idm_synth}"
TRAIN_SYNTH="${TRAIN_SYNTH:-/data/agent/hf_models/datasets/qwen_vton/synth/idm_unpaired_train}"
LOG="$ROOT/outputs/qwen_vton_lora/logs/idm_after_test_chain.log"
NUM_GPUS="${NUM_GPUS:-4}"
EXPECTED_TEST="${EXPECTED_TEST:-2032}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] wait for test synth >= $EXPECTED_TEST"
while true; do
  n=0
  if [[ -f "$TEST_SYNTH/manifest.jsonl" ]]; then
    n=$(wc -l < "$TEST_SYNTH/manifest.jsonl")
  fi
  # Also count if batch script still running
  if [[ "$n" -ge "$EXPECTED_TEST" ]]; then
    echo "[$(date -Is)] test manifest ready n=$n"
    break
  fi
  if ! pgrep -f "synthesize_unpaired_idm.py.*idm_unpaired[^-]" >/dev/null 2>&1 \
     && ! pgrep -f "run_idm_batch_synth.sh" >/dev/null 2>&1; then
    # process gone but incomplete — still convert what we have, then continue train
    echo "[$(date -Is)] WARN test synth process stopped early n=$n; continue with available"
    break
  fi
  echo "[$(date -Is)] waiting test n=$n / $EXPECTED_TEST"
  sleep 120
done

echo "[$(date -Is)] convert test synth -> $TEST_CONV"
"$PY" "$SCRIPTS/convert_idm_synth_to_qwen_edit.py" \
  --synth-dir "$TEST_SYNTH" \
  --raw-root /data/agent/hf_models/datasets/qwen_vton/raw \
  --out-dir "$TEST_CONV"

echo "[$(date -Is)] launch multi-GPU train unpaired NUM_GPUS=$NUM_GPUS"
NUM_GPUS="$NUM_GPUS" SYNTH="$TRAIN_SYNTH" bash "$SCRIPTS/run_idm_train_multigpu.sh"

echo "[$(date -Is)] chain launched (train shards running in tmux)"
