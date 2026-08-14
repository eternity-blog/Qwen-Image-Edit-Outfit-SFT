#!/usr/bin/env bash
# Download VITON-HD + DressCode into /data/agent/hf_models/datasets/qwen_vton/raw
set -euo pipefail

ROOT="${ROOT:-/data/agent/hf_models/datasets/qwen_vton}"
RAW="$ROOT/raw"
LOG="$ROOT/logs/download.log"
mkdir -p "$RAW/viton_hd" "$RAW/dresscode" "$ROOT/logs"

export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"
export NO_PROXY="$no_proxy"
export HF_HUB_ENABLE_HF_TRANSFER=0

PY="${PY:-/data/agent/conda/envs/qwen-image-edit/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python3; fi

HF=huggingface-cli
if ! command -v "$HF" >/dev/null 2>&1; then
  HF=/usr/local/bin/huggingface-cli
fi

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] download start"

# --- VITON-HD classic package (~4.9GB): zalando-hd-resized.zip ---
# Contains train/{image,cloth,...} and test/{...}
echo "[$(date -Is)] VITON-HD (skush1/viton-hd zalando-hd-resized.zip)"
"$HF" download skush1/viton-hd \
  --repo-type dataset \
  --local-dir "$RAW/viton_hd" \
  --include "zalando-hd-resized.zip"

if [[ ! -d "$RAW/viton_hd/train/image" && ! -d "$RAW/viton_hd/train" ]]; then
  echo "extracting zalando-hd-resized.zip ..."
  "$PY" - <<'PY'
import zipfile
from pathlib import Path
root = Path("/data/agent/hf_models/datasets/qwen_vton/raw/viton_hd")
zpath = root / "zalando-hd-resized.zip"
assert zpath.is_file(), zpath
with zipfile.ZipFile(zpath) as zf:
    zf.extractall(root)
# normalize nested folder if any
for p in root.iterdir():
    if p.is_dir() and (p / "train").is_dir() and p.name != "train":
        for child in p.iterdir():
            dest = root / child.name
            if not dest.exists():
                child.rename(dest)
        break
print("children", sorted(x.name for x in root.iterdir())[:30])
train = root / "train"
if train.is_dir():
    print("train", sorted(x.name for x in train.iterdir())[:20])
PY
fi

date -Is > "$ROOT/logs/viton_ready"
echo "[$(date -Is)] VITON ready"

if [[ "${SKIP_DRESSCODE:-0}" == "1" ]]; then
  echo "[$(date -Is)] SKIP_DRESSCODE=1"
  du -sh "$RAW/viton_hd" || true
  exit 0
fi

# --- DressCode (~72GB split archive) ---
echo "[$(date -Is)] DressCode parts (JianhaoZeng/Dresscode)"
"$HF" download JianhaoZeng/Dresscode \
  --repo-type dataset \
  --local-dir "$RAW/dresscode" \
  --include "DressCode_part_aa" "DressCode_part_ab" "README.md"

if [[ ! -d "$RAW/dresscode/DressCode" ]]; then
  echo "concat + extract DressCode (ZIP split parts)..."
  ZIP="$RAW/dresscode/DressCode.zip"
  if [[ ! -f "$ZIP" ]]; then
    if [[ -f "$RAW/dresscode/DressCode.tar" ]]; then
      mv "$RAW/dresscode/DressCode.tar" "$ZIP"
    else
      cat "$RAW/dresscode/DressCode_part_aa" "$RAW/dresscode/DressCode_part_ab" > "$ZIP"
    fi
  fi
  # Prefer dedicated extractor (handles missing system unzip)
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  "$PY" "$SCRIPT_DIR/extract_dresscode_zip.py" --zip "$ZIP" \
    --out-extract "$RAW/dresscode/extract" \
    --final-dir "$RAW/dresscode/DressCode" \
    --ready-flag "$ROOT/logs/dresscode_ready"
fi

echo "[$(date -Is)] download DONE"
du -sh "$RAW/viton_hd" "$RAW/dresscode" "$RAW/dresscode/DressCode" 2>/dev/null || true
ls "$RAW/dresscode/DressCode" 2>/dev/null | head -40
