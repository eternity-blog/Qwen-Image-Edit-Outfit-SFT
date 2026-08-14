#!/usr/bin/env bash
# Case02 compare: base vs IDM-LoRA fused (and optional old VTON fused).
set -euo pipefail

ROOT="${ROOT:-/data/agent/lixiao29/QualityInspection-sync}"
SCRIPTS="$ROOT/Qwen-Image-Edit/scripts"
ENV_DIR="${ENV_DIR:-/data/agent/conda/envs/qwen-image-edit}"
BASE_MODEL="${BASE_MODEL:-/data/agent/hf_models/Qwen/Qwen-Image-Edit-2511}"
IDM_MODEL="${IDM_MODEL:-/data/agent/hf_models/Qwen/Qwen-Image-Edit-2511-idm-lora-fused}"
VTON_MODEL="${VTON_MODEL:-/data/agent/hf_models/Qwen/Qwen-Image-Edit-2511-vton-lora-fused}"
CASE_ID="${CASE_ID:-02}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/outfit_v2_case02_full/case02-full-v2b}"
TESTSET_DIR="${TESTSET_DIR:-$ROOT/kling-aigc-engine/TestSet}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/qwen_kf_zeroshot/case${CASE_ID}_idm_vs_base}"
DEVICE="${DEVICE:-cuda:0}"
PROMPT_MODE="${PROMPT_MODE:-short}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"
SHOTS="${SHOTS:-0,1}"
ROLES="${ROLES:-start}"
STEPS="${STEPS:-40}"
PRODUCT_LIMIT="${PRODUCT_LIMIT:-2}"
SIZE="${SIZE:-720x1280}"
RUN_VTON="${RUN_VTON:-1}"

if [[ "${AUTO_GPU:-1}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

mkdir -p "$OUT_ROOT"/{base,idm,vton,grids}
LOG="$OUT_ROOT/run.log"

export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"
export NO_PROXY="$no_proxy"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] idm vs base compare start gpu=$CUDA_VISIBLE_DEVICES prompt=$PROMPT_MODE"

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

run_one() {
  local name="$1" model="$2" out="$3"
  if [[ -f "$out/qwen/00_start.png" && -f "$out/run_meta.json" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Is)] skip $name (exists)"
    return 0
  fi
  echo "[$(date -Is)] === $name ==="
  "$ENV_DIR/bin/python" "$SCRIPTS/zero_shot_compare.py" \
    "${COMMON[@]}" \
    --out-dir "$out" \
    --model-dir "$model"
}

run_one base "$BASE_MODEL" "$OUT_ROOT/base"
run_one idm "$IDM_MODEL" "$OUT_ROOT/idm"
if [[ "$RUN_VTON" == "1" && -f "$VTON_MODEL/model_index.json" ]]; then
  run_one vton "$VTON_MODEL" "$OUT_ROOT/vton"
fi

echo "[$(date -Is)] === COMPOSE ==="
"$ENV_DIR/bin/python" "$SCRIPTS/compose_idm_compare.py" \
  --base-dir "$OUT_ROOT/base" \
  --idm-dir "$OUT_ROOT/idm" \
  --vton-dir "$OUT_ROOT/vton" \
  --out-dir "$OUT_ROOT"

echo "[$(date -Is)] DONE -> $OUT_ROOT"
ls -la "$OUT_ROOT/grids" || true
cat "$OUT_ROOT/summary.md" || true
