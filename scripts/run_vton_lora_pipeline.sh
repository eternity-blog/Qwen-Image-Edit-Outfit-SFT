#!/usr/bin/env bash
# Orchestrate: download (VITON first) -> convert -> train LoRA -> fuse.
# DressCode continues downloading in background after VITON is ready.
set -euo pipefail

ROOT="${ROOT:-/data/agent/lixiao29/QualityInspection-sync}"
SCRIPTS="$ROOT/Qwen-Image-Edit/scripts"
DATA_ROOT="${DATA_ROOT:-/data/agent/hf_models/datasets/qwen_vton}"
ENV_DIR="${ENV_DIR:-/data/agent/conda/envs/qwen-image-edit}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/qwen_vton_lora}"
mkdir -p "$OUT_ROOT/logs" "$DATA_ROOT/logs"
LOG="$OUT_ROOT/logs/pipeline.log"

export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"
export NO_PROXY="$no_proxy"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] pipeline start"

# 1) VITON only first (fast)
SKIP_DRESSCODE=1 bash "$SCRIPTS/download_vton_datasets.sh"

# 2) Kick off DressCode download in background (large)
if [[ ! -f "$DATA_ROOT/logs/dresscode_ready" ]]; then
  echo "[$(date -Is)] starting DressCode download in background tmux lixiao-dresscode-dl"
  tmux has-session -t lixiao-dresscode-dl 2>/dev/null && tmux kill-session -t lixiao-dresscode-dl || true
  tmux new -d -s lixiao-dresscode-dl "SKIP_DRESSCODE=0 bash $SCRIPTS/download_vton_datasets.sh"
fi

# 3) Convert whatever is available (VITON now; DressCode later if ready)
echo "[$(date -Is)] convert"
"$ENV_DIR/bin/python" "$SCRIPTS/convert_vton_to_qwen_edit.py" \
  --raw-root "$DATA_ROOT/raw" \
  --out-dir "$DATA_ROOT/converted"

"$ENV_DIR/bin/python" - <<'PY'
import json
from pathlib import Path
src = Path("/data/agent/hf_models/datasets/qwen_vton/converted/metadata_train.json")
rows = json.loads(src.read_text())
viton = [r for r in rows if r.get("source")=="viton_hd"]
dress = [r for r in rows if r.get("source")=="dresscode"]
subset = (dress[:2000] + viton[:2000]) if dress else viton[:4000]
if not subset:
    subset = rows[:4000]
out = Path("/data/agent/hf_models/datasets/qwen_vton/converted/metadata_train_subset4k.json")
out.write_text(json.dumps(subset, ensure_ascii=False, indent=2))
print("subset", len(subset), "viton", len(viton), "dress", len(dress), "->", out)
PY

# 4) Train + fuse
echo "[$(date -Is)] train+fuse"
METADATA="$DATA_ROOT/converted/metadata_train_subset4k.json" \
NUM_EPOCHS="${NUM_EPOCHS:-1}" \
LORA_RANK="${LORA_RANK:-16}" \
DATASET_REPEAT="${DATASET_REPEAT:-1}" \
bash "$SCRIPTS/train_vton_lora.sh"

echo "[$(date -Is)] pipeline DONE"
echo "DressCode dl session: tmux capture-pane -pt lixiao-dresscode-dl | tail"
