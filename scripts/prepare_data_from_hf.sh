#!/usr/bin/env bash
# Download the published HF synth dataset and wire dataset_base for DiffSynth training.
#
# HF layout uses images/part-XXXX/ (≤10k files/dir). Training metadata expects flat
# idm_synth/<name>.jpg under dataset_base — this script builds a flat symlink view,
# then (re)runs v2 convert against local VITON.
#
# Usage:
#   export HF_TOKEN=...   # if private
#   export QWEN_VTON_DATA=/path/to/qwen_vton
#   # put VITON-HD under $QWEN_VTON_DATA/raw/viton_hd  (CC BY-NC; download yourself)
#   bash scripts/prepare_data_from_hf.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_env.sh"

HF_REPO="${HF_REPO:-lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling}"
DATA_ROOT_VTON="${QWEN_VTON_DATA:-${DATA_ROOT}/datasets/qwen_vton}"
HF_DIR="${HF_DIR:-$DATA_ROOT_VTON/from_hf}"
VITON_DIR="${VITON_ROOT:-$DATA_ROOT_VTON/raw/viton_hd}"

PY="${ENV_DIR:+$ENV_DIR/bin/python}"
PY="${PY:-python3}"

mkdir -p "$DATA_ROOT_VTON" "$HF_DIR" "$DATA_ROOT_VTON/raw" "$DATA_ROOT_VTON/synth"

echo "[$(date -Is)] download $HF_REPO -> $HF_DIR"
"$PY" - <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="$HF_REPO",
    repo_type="dataset",
    local_dir="$HF_DIR",
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
)
print("download ok")
PY

# Link / copy synth trees into canonical paths
for split in idm_unpaired idm_unpaired_train; do
  src="$HF_DIR/synth/$split"
  dst="$DATA_ROOT_VTON/synth/$split"
  if [[ -d "$src" ]]; then
    mkdir -p "$dst"
    # manifest
    if [[ -f "$src/manifest.jsonl" ]]; then
      cp -f "$src/manifest.jsonl" "$dst/manifest.jsonl"
    fi
    # flatten part-* into images/ via symlinks (no duplicate bytes)
    img_dst="$dst/images"
    rm -rf "$img_dst"
    mkdir -p "$img_dst"
    if [[ -d "$src/images" ]]; then
      echo "flatten $src/images -> $img_dst"
      "$PY" - <<PY
from pathlib import Path
src = Path("$src/images")
dst = Path("$img_dst")
n = 0
for p in sorted(src.rglob("*")):
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        link = dst / p.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(p.resolve())
        n += 1
print(f"symlinked {n} images")
PY
    fi
  fi
done

if [[ ! -d "$VITON_DIR/train" && ! -d "$VITON_DIR/test" ]]; then
  echo "WARNING: VITON-HD not found at $VITON_DIR"
  echo "  Download VITON-HD yourself (CC BY-NC) and place under raw/viton_hd/{train,test}/..."
  echo "  Then re-run: bash scripts/run_convert_idm_v2.sh"
  exit 0
fi

echo "[$(date -Is)] rebuild full-v2 metadata + dataset_base"
SYNTH_DIR="$DATA_ROOT_VTON/synth/idm_unpaired_train" \
  OUT_DIR="$DATA_ROOT_VTON/converted_idm_synth_train_v2" \
  bash "$SCRIPT_DIR/run_convert_idm_v2.sh"

echo "[$(date -Is)] prepare DONE"
echo "  metadata: $DATA_ROOT_VTON/converted_idm_synth_train_v2/metadata_train.json"
echo "  dataset_base: $DATA_ROOT_VTON/converted_idm_synth_train_v2/dataset_base"
