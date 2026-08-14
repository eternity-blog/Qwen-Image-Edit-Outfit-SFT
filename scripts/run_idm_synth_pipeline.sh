#!/usr/bin/env bash
# Orchestrate: setup IDM-VTON -> synthesize unpaired try-on -> convert to Qwen-Edit metadata.
set -euo pipefail

ROOT="${ROOT:-/data/agent/lixiao29/QualityInspection-sync}"
SCRIPTS="$ROOT/Qwen-Image-Edit/scripts"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/qwen_vton_lora}"
DATA_ROOT="${DATA_ROOT:-/data/agent/hf_models/datasets/qwen_vton}"
VITON_ROOT="${VITON_ROOT:-$DATA_ROOT/raw/viton_hd}"
SYNTH_DIR="${SYNTH_DIR:-$DATA_ROOT/synth/idm_unpaired}"
CONV_DIR="${CONV_DIR:-$DATA_ROOT/converted_idm_synth}"
ENV_DIR="${ENV_DIR:-/data/agent/conda/envs/idm-vton}"
REPO_DIR="${REPO_DIR:-/data/agent/hf_models/modules/IDM-VTON}"
MODEL_DIR="${MODEL_DIR:-/data/agent/hf_models/yisol/IDM-VTON}"
LIMIT="${LIMIT:-64}"          # smoke default; set 2032 for full test unpaired
PHASE="${PHASE:-test}"
PAIRS="${PAIRS:-test_pairs.txt}"
LOG="$OUT_ROOT/logs/idm_synth_pipeline.log"

mkdir -p "$OUT_ROOT/logs" "$SYNTH_DIR"
exec > >(tee -a "$LOG") 2>&1

export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"
export NO_PROXY="$no_proxy"

if [[ "${AUTO_GPU:-1}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "[$(date -Is)] idm synth pipeline start gpu=$CUDA_VISIBLE_DEVICES limit=$LIMIT"

bash "$SCRIPTS/setup_idm_vton.sh"

PY="$ENV_DIR/bin/python"
echo "[$(date -Is)] synthesize unpaired try-on"
"$PY" "$SCRIPTS/synthesize_unpaired_idm.py" \
  --data-root "$VITON_ROOT" \
  --pairs "$PAIRS" \
  --phase "$PHASE" \
  --unpaired \
  --limit "$LIMIT" \
  --batch-size 1 \
  --model-dir "$MODEL_DIR" \
  --repo-dir "$REPO_DIR" \
  --out-dir "$SYNTH_DIR" \
  --device cuda:0

echo "[$(date -Is)] convert to Qwen-Edit metadata"
"$PY" "$SCRIPTS/convert_idm_synth_to_qwen_edit.py" \
  --synth-dir "$SYNTH_DIR" \
  --raw-root "$DATA_ROOT/raw" \
  --out-dir "$CONV_DIR"

echo "[$(date -Is)] DONE"
echo "synth=$SYNTH_DIR"
echo "converted=$CONV_DIR"
echo "dataset_base=$(cat "$CONV_DIR/stats.json")"
wc -l "$SYNTH_DIR/manifest.jsonl" || true
