#!/usr/bin/env bash
# Smoke: case02 base vs fused LoRA (same inputs), then compose 4-panel grids.
set -euo pipefail

ROOT="${ROOT:-/data/agent/lixiao29/QualityInspection-sync}"
SCRIPTS="$ROOT/Qwen-Image-Edit/scripts"
ENV_DIR="${ENV_DIR:-/data/agent/conda/envs/qwen-image-edit}"
BASE_MODEL="${BASE_MODEL:-/data/agent/hf_models/Qwen/Qwen-Image-Edit-2511}"
FUSED_MODEL="${FUSED_MODEL:-/data/agent/hf_models/Qwen/Qwen-Image-Edit-2511-vton-lora-fused}"
CASE_ID="${CASE_ID:-02}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/outfit_v2_case02_full/case02-full-v2b}"
TESTSET_DIR="${TESTSET_DIR:-$ROOT/kling-aigc-engine/TestSet}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/qwen_kf_zeroshot/case${CASE_ID}_fused_vs_base}"
BASE_OUT="$OUT_ROOT/base"
FUSED_OUT="$OUT_ROOT/fused"
DEVICE="${DEVICE:-cuda:0}"
PROMPT_MODE="${PROMPT_MODE:-short}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"
SHOTS="${SHOTS:-0,1}"
ROLES="${ROLES:-start}"
STEPS="${STEPS:-40}"
PRODUCT_LIMIT="${PRODUCT_LIMIT:-2}"
SIZE="${SIZE:-720x1280}"

if [[ "${AUTO_GPU:-1}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUT_ROOT" "$BASE_OUT" "$FUSED_OUT"
LOG="$OUT_ROOT/run.log"

export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"
export NO_PROXY="$no_proxy"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] fused vs base smoke start gpu=$CUDA_VISIBLE_DEVICES"

COMMON=(
  --run-dir "$RUN_DIR"
  --testset-dir "$TESTSET_DIR"
  --case-id "$CASE_ID"
  --device "$DEVICE"
  --prompt-mode "$PROMPT_MODE"
  --shots "$SHOTS"
  --roles "$ROLES"
  --max-samples "$MAX_SAMPLES"
  --product-limit "$PRODUCT_LIMIT"
  --steps "$STEPS"
  --size "$SIZE"
  --seed 0
)

echo "[$(date -Is)] === BASE ==="
"$ENV_DIR/bin/python" "$SCRIPTS/zero_shot_compare.py" \
  "${COMMON[@]}" \
  --out-dir "$BASE_OUT" \
  --model-dir "$BASE_MODEL"

echo "[$(date -Is)] === FUSED ==="
"$ENV_DIR/bin/python" "$SCRIPTS/zero_shot_compare.py" \
  "${COMMON[@]}" \
  --out-dir "$FUSED_OUT" \
  --model-dir "$FUSED_MODEL"

echo "[$(date -Is)] === COMPOSE GRIDS ==="
"$ENV_DIR/bin/python" "$SCRIPTS/compose_fused_vs_base.py" \
  --base-dir "$BASE_OUT" \
  --fused-dir "$FUSED_OUT" \
  --out-dir "$OUT_ROOT"

echo "[$(date -Is)] DONE -> $OUT_ROOT"
ls -la "$OUT_ROOT/grids" || true
cat "$OUT_ROOT/summary.md" || true
