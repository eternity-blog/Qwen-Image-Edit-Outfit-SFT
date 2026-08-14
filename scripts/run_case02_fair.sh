#!/usr/bin/env bash
# Fair zero-shot compare: same source / products / production prompt / canvas as GPT.
set -euo pipefail

ROOT="${ROOT:-/data/agent/lixiao29/QualityInspection-sync}"
ENV_DIR="${ENV_DIR:-/data/agent/conda/envs/qwen-image-edit}"
MODEL_DIR="${MODEL_DIR:-/data/agent/hf_models/Qwen/Qwen-Image-Edit-2511}"
CASE_ID="${CASE_ID:-02}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/outfit_v2_case02_full/case02-full-v2b}"
TESTSET_DIR="${TESTSET_DIR:-$ROOT/kling-aigc-engine/TestSet}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/qwen_kf_zeroshot/case${CASE_ID}_fair}"
DEVICE="${DEVICE:-cuda:0}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"
SHOTS="${SHOTS:-0,1}"
ROLES="${ROLES:-start}"
STEPS="${STEPS:-40}"

if [[ "${AUTO_GPU:-1}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"
export NO_PROXY="$no_proxy"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "[$(date -Is)] FAIR compare start case=$CASE_ID device=$DEVICE gpu=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"

"$ENV_DIR/bin/python" "$ROOT/Qwen-Image-Edit/scripts/zero_shot_compare.py" \
  --fair \
  --run-dir "$RUN_DIR" \
  --testset-dir "$TESTSET_DIR" \
  --case-id "$CASE_ID" \
  --out-dir "$OUT_DIR" \
  --model-dir "$MODEL_DIR" \
  --device "$DEVICE" \
  --shots "$SHOTS" \
  --roles "$ROLES" \
  --max-samples "$MAX_SAMPLES" \
  --steps "$STEPS" \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] FAIR compare DONE" | tee -a "$LOG"
