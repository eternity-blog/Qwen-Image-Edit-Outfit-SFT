#!/usr/bin/env bash
# Rewrite IDM synth metadata with FULL Outfit v2 prompts (no IDM re-run, no training).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_env.sh"

SCRIPTS="$QWEN_OUTFIT_ROOT/scripts"
DATA_ROOT_VTON="${QWEN_VTON_DATA:-${DATA_ROOT}/datasets/qwen_vton}"
SYNTH_DIR="${SYNTH_DIR:-$DATA_ROOT_VTON/synth/idm_unpaired_train}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT_VTON/converted_idm_synth_train_v2}"
LOG_DIR="$OUTPUT_ROOT/qwen_vton_lora/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/convert_idm_v2.log"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] convert IDM synth -> v2 full prompts"
echo "repo=$QWEN_OUTFIT_ROOT synth=$SYNTH_DIR out=$OUT_DIR"

if [[ -n "${ENV_DIR:-}" && -x "${ENV_DIR}/bin/python" ]]; then
  PY="$ENV_DIR/bin/python"
else
  PY="${PYTHON:-python3}"
fi

"$PY" "$SCRIPTS/convert_idm_synth_to_qwen_edit_v2.py" \
  --synth-dir "$SYNTH_DIR" \
  --raw-root "$DATA_ROOT_VTON/raw" \
  --out-dir "$OUT_DIR" \
  --facing front \
  --category upper \
  --val-ratio 0.02 \
  --seed 0 \
  --audit-n 50

AUDIT_OUT="$OUTPUT_ROOT/qwen_vton_lora/v2_prompt_convert_audit"
mkdir -p "$AUDIT_OUT"
cp -f "$OUT_DIR/stats.json" "$AUDIT_OUT/"
cp -f "$OUT_DIR/prompt_example_full.txt" "$AUDIT_OUT/"
cp -f "$OUT_DIR/prompt_audit_sample.json" "$AUDIT_OUT/"
echo "[$(date -Is)] DONE audit -> $AUDIT_OUT"
cat "$OUT_DIR/stats.json"
