#!/usr/bin/env bash
# Resume full VITON-HD test unpaired IDM synthesis (skips already-done samples).
set -euo pipefail

ROOT="${ROOT:-/data/agent/lixiao29/QualityInspection-sync}"
SCRIPTS="$ROOT/Qwen-Image-Edit/scripts"
PY="${PY:-/root/conda-envs/idm-vton/bin/python}"
SYNTH="${SYNTH:-/data/agent/hf_models/datasets/qwen_vton/synth/idm_unpaired}"
CONV="${CONV:-/data/agent/hf_models/datasets/qwen_vton/converted_idm_synth}"
LOG="$ROOT/outputs/qwen_vton_lora/logs/idm_batch_synth.log"

export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"
export NO_PROXY="$no_proxy"

if [[ "${AUTO_GPU:-1}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

mkdir -p "$(dirname "$LOG")" "$SYNTH"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] batch start gpu=$CUDA_VISIBLE_DEVICES existing=$(wc -l < "$SYNTH/manifest.jsonl" 2>/dev/null || echo 0)"
"$PY" "$SCRIPTS/synthesize_unpaired_idm.py" \
  --data-root /data/agent/hf_models/datasets/qwen_vton/raw/viton_hd \
  --pairs test_pairs.txt \
  --phase test \
  --unpaired \
  --limit 0 \
  --batch-size 1 \
  --model-dir /data/agent/hf_models/yisol/IDM-VTON \
  --repo-dir /data/agent/hf_models/modules/IDM-VTON \
  --out-dir "$SYNTH" \
  --device cuda:0

echo "[$(date -Is)] convert metadata"
"$PY" "$SCRIPTS/convert_idm_synth_to_qwen_edit.py" \
  --synth-dir "$SYNTH" \
  --raw-root /data/agent/hf_models/datasets/qwen_vton/raw \
  --out-dir "$CONV"

echo "[$(date -Is)] BATCH DONE"
wc -l "$SYNTH/manifest.jsonl"
cat "$CONV/stats.json"
