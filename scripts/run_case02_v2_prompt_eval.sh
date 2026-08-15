#!/usr/bin/env bash
# P0: Case02 eval with LIVE Outfit v2 garment-only prompt (not stored old keyframe_prompt).
# Compares Qwen base vs IDM-LoRA fused under the same production-aligned prompt.
set -euo pipefail

ROOT="${ROOT:-/data/agent/lixiao29/QualityInspection-sync}"
SCRIPTS="$ROOT/Qwen-Image-Edit/scripts"
ENV_DIR="${ENV_DIR:-/data/agent/conda/envs/qwen-image-edit}"
BASE_MODEL="${BASE_MODEL:-/data/agent/hf_models/Qwen/Qwen-Image-Edit-2511}"
IDM_MODEL="${IDM_MODEL:-/data/agent/hf_models/Qwen/Qwen-Image-Edit-2511-idm-lora-fused}"
CASE_ID="${CASE_ID:-02}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/outfit_v2_case02_full/case02-full-v2b}"
TESTSET_DIR="${TESTSET_DIR:-$ROOT/kling-aigc-engine/TestSet}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/qwen_kf_zeroshot/case${CASE_ID}_v2prompt_idm_vs_base}"
DEVICE="${DEVICE:-cuda:0}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"
SHOTS="${SHOTS:-0,1}"
ROLES="${ROLES:-start}"
STEPS="${STEPS:-40}"
# v2 planner keeps primary + up to 2 supplements
PRODUCT_LIMIT="${PRODUCT_LIMIT:-3}"
SIZE="${SIZE:-720x1280}"
# Case02 product-card visual facts (SKU-level; facing-specific backside handled by template).
PRODUCT_VISUAL_FACTS="${PRODUCT_VISUAL_FACTS:-运动短袖上衣 + 短裤套装；服装本体可见回力品牌标识；背面未知}"

if [[ "${AUTO_GPU:-1}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

mkdir -p "$OUT_ROOT"/{base,idm,grids,prompts}
LOG="$OUT_ROOT/run.log"

export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"
export NO_PROXY="$no_proxy"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] P0 live-v2-prompt eval start gpu=$CUDA_VISIBLE_DEVICES"

COMMON=(
  --run-dir "$RUN_DIR"
  --testset-dir "$TESTSET_DIR"
  --case-id "$CASE_ID"
  --device "$DEVICE"
  --prompt-mode v2
  --product-visual-facts "$PRODUCT_VISUAL_FACTS"
  --shots "$SHOTS"
  --roles "$ROLES"
  --max-samples "$MAX_SAMPLES"
  --product-limit "$PRODUCT_LIMIT"
  --steps "$STEPS"
  --size "$SIZE"
  --max-sequence-length 1024
  --seed 0
)

# On a shared box a single 80GB card may not hold DiT+TE+VAE (~57.7GB bf16)
# alongside another tenant's allocation. CPU_OFFLOAD=1 streams modules per-step.
if [[ "${CPU_OFFLOAD:-0}" == "1" ]]; then
  COMMON+=(--cpu-offload)
fi

# Dump prompts once (no model load) for audit.
"$ENV_DIR/bin/python" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "$SCRIPTS")
import zero_shot_compare as z
samples = z.samples_from_outfit_run(
    run_dir=Path("$RUN_DIR"),
    testset_dir=Path("$TESTSET_DIR"),
    case_id="$CASE_ID",
    roles=[r for r in "$ROLES".split(",") if r],
    shot_indices=[int(x) for x in "$SHOTS".split(",") if x.strip()!=""],
    max_samples=int("$MAX_SAMPLES"),
    prompt_mode="v2",
    product_limit=int("$PRODUCT_LIMIT"),
    out_width=int("${SIZE%x*}"),
    out_height=int("${SIZE#*x}"),
    match_gpt_size=False,
    product_visual_facts="""$PRODUCT_VISUAL_FACTS""",
)
out = Path("$OUT_ROOT/prompts")
out.mkdir(parents=True, exist_ok=True)
meta = []
for s in samples:
    tag = f"{s.shot_index:02d}_{s.role}"
    (out / f"{tag}.txt").write_text(s.prompt, encoding="utf-8")
    meta.append({
        "tag": tag,
        "prompt_chars": len(s.prompt),
        "n_products": len(s.product_paths),
        "products": s.product_paths,
        "source": s.source_path,
    })
    print(tag, "chars", len(s.prompt), "products", len(s.product_paths))
(out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote prompts to", out)
PY

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

echo "[$(date -Is)] === COMPOSE ==="
"$ENV_DIR/bin/python" "$SCRIPTS/compose_idm_compare.py" \
  --base-dir "$OUT_ROOT/base" \
  --idm-dir "$OUT_ROOT/idm" \
  --vton-dir "" \
  --idm-label "${MODEL_B_LABEL:-IDM-LoRA}" \
  --out-dir "$OUT_ROOT"

# rewrite summary header for P0
cat > "$OUT_ROOT/summary.md" <<EOF
# Case02 P0: live Outfit v2 prompt — base vs IDM-LoRA

- prompt_mode: **live v2** (\`outfit_garment_only_keyframe_prompt\`)
- product_visual_facts: \`$PRODUCT_VISUAL_FACTS\`
- max_sequence_length: 1024 (EditPlus API check only; encode does not truncate — see truncation_probe.json)
- size: \`$SIZE\` steps=\`$STEPS\` shots=\`$SHOTS\`
- products: up to $PRODUCT_LIMIT

Panels: source | GPT Image 2 | Qwen base | IDM-LoRA

Prompts dumped under \`prompts/\`.
EOF
ls -la "$OUT_ROOT/grids" || true
cat "$OUT_ROOT/summary.md"

echo "[$(date -Is)] DONE -> $OUT_ROOT"
