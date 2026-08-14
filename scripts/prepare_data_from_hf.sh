#!/usr/bin/env bash
# Download ALL public training inputs from Hugging Face and wire dataset_base.
#
# Pulls:
#   1) lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling  (IDM synth + v2 metadata files)
#   2) skush1/viton-hd  (zalando-hd-resized.zip → raw/viton_hd)  CC BY-NC
# Then flattens part-* images and rebuilds full-v2 DiffSynth metadata.
#
# No user-supplied dataset paths required. Optional: HF_TOKEN if rate-limited.
#
# Usage:
#   source configs/env.local.sh   # or rely on lib_env defaults
#   bash scripts/prepare_data_from_hf.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_env.sh"

HF_REPO="${HF_REPO:-lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling}"
VITON_HF_REPO="${VITON_HF_REPO:-skush1/viton-hd}"
DATA_ROOT_VTON="${QWEN_VTON_DATA:-${DATA_ROOT}/datasets/qwen_vton}"
HF_DIR="${HF_DIR:-$DATA_ROOT_VTON/from_hf}"
VITON_DIR="${VITON_ROOT:-$DATA_ROOT_VTON/raw/viton_hd}"

if [[ -n "${ENV_DIR:-}" && -x "${ENV_DIR}/bin/python" ]]; then
  PY="$ENV_DIR/bin/python"
else
  PY="${PYTHON:-python3}"
fi

mkdir -p "$DATA_ROOT_VTON" "$HF_DIR" "$VITON_DIR" "$DATA_ROOT_VTON/synth"

echo "[$(date -Is)] [1/3] download synth dataset $HF_REPO -> $HF_DIR"
"$PY" - <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="$HF_REPO",
    repo_type="dataset",
    local_dir="$HF_DIR",
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
)
print("synth download ok")
PY

echo "[$(date -Is)] [2/3] ensure VITON-HD at $VITON_DIR (public HF: $VITON_HF_REPO)"
if [[ -d "$VITON_DIR/train/image" || -d "$VITON_DIR/train" ]]; then
  echo "VITON already present, skip download"
else
  "$PY" - <<PY
from huggingface_hub import hf_hub_download
import os, zipfile
from pathlib import Path

root = Path("$VITON_DIR")
root.mkdir(parents=True, exist_ok=True)
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
zpath = Path(
    hf_hub_download(
        repo_id="$VITON_HF_REPO",
        repo_type="dataset",
        filename="zalando-hd-resized.zip",
        local_dir=str(root),
        token=token,
    )
)
print("downloaded", zpath, "size", zpath.stat().st_size)
print("extracting...")
with zipfile.ZipFile(zpath) as zf:
    zf.extractall(root)
# normalize nested folder if zip has a single top-level dir containing train/
for p in root.iterdir():
    if p.is_dir() and (p / "train").is_dir() and p.name not in {"train", "test"}:
        for child in p.iterdir():
            dest = root / child.name
            if not dest.exists():
                child.rename(dest)
        break
train = root / "train"
assert train.is_dir(), f"expected {train} after extract"
print("VITON ready", sorted(x.name for x in train.iterdir())[:12])
PY
fi

echo "[$(date -Is)] flatten HF synth part-* -> synth/*/images"
for split in idm_unpaired idm_unpaired_train; do
  src="$HF_DIR/synth/$split"
  dst="$DATA_ROOT_VTON/synth/$split"
  if [[ ! -d "$src" ]]; then
    echo "WARN: missing $src"
    continue
  fi
  mkdir -p "$dst"
  if [[ -f "$src/manifest.jsonl" ]]; then
    cp -f "$src/manifest.jsonl" "$dst/manifest.jsonl"
  fi
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
done

if [[ ! -d "$VITON_DIR/train/image" && ! -d "$VITON_DIR/train" ]]; then
  echo "ERROR: VITON-HD still missing at $VITON_DIR after download"
  exit 1
fi

echo "[$(date -Is)] [3/3] rebuild full-v2 metadata + dataset_base"
SYNTH_DIR="$DATA_ROOT_VTON/synth/idm_unpaired_train" \
  OUT_DIR="$DATA_ROOT_VTON/converted_idm_synth_train_v2" \
  bash "$SCRIPT_DIR/run_convert_idm_v2.sh"

echo "[$(date -Is)] prepare DONE — all inputs from public HF"
echo "  metadata: $DATA_ROOT_VTON/converted_idm_synth_train_v2/metadata_train.json"
echo "  dataset_base: $DATA_ROOT_VTON/converted_idm_synth_train_v2/dataset_base"
echo "  note: VITON-HD is CC BY-NC — research / non-commercial only"
