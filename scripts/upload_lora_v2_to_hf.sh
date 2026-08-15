#!/usr/bin/env bash
# Upload the LoRA-v2 fused model to the HF repo under a subfolder, keeping the
# repo root (full-SFT model) intact. Non-destructive: re-running re-updates only
# the lora_v2/ subfolder, never touches root or deletes anything.
#
# Usage:
#   source configs/env.local.sh
#   bash scripts/upload_lora_v2_to_hf.sh
# Override: REPO_ID / FUSED / PATH_IN_REPO / commit message via env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_env.sh"

# This box reaches HF through an overseas proxy; xet protocol hangs through it,
# so force plain HTTPS (see memory: machine-cuda-torch-constraint).
export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-$http_proxy}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

REPO_ID="${REPO_ID:-lee31221/Qwen-Image-Edit-Outfit-2511-SFT}"
FUSED="${FUSED:-$OUTPUT_ROOT/qwen_vton_lora_v2/fused}"
PATH_IN_REPO="${PATH_IN_REPO:-lora_v2}"
COMMIT_MSG="${COMMIT_MSG:-Add LoRA-v2 fused model (same v2 data as full SFT; controlled counterpart)}"

LOG_DIR="$OUTPUT_ROOT/qwen_vton_lora_v2/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/upload_lora_v2_to_hf.log"

PY="${ENV_DIR}/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: missing python at ENV_DIR=$ENV_DIR"; exit 1; }
[[ -d "$FUSED" ]] || { echo "ERROR: fused dir not found: $FUSED"; exit 1; }
[[ -f "$FUSED/model_index.json" ]] || { echo "ERROR: $FUSED/model_index.json missing — not a loadable pipeline"; exit 1; }

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] upload lora_v2 fused -> $REPO_ID/$PATH_IN_REPO"
echo "  src=$FUSED ($(du -sh "$FUSED" 2>/dev/null | cut -f1))"
echo "  proxy=$http_proxy  HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"

# Sanity: must be logged in (write access to lee31221/*).
"$PY" - <<'PY' || { echo "ERROR: not logged in to HF as a user with write access to $REPO_ID"; exit 1; }
from huggingface_hub import whoami
info = whoami()
print("  hf user:", info.get("name"))
PY

"$PY" - "$REPO_ID" "$FUSED" "$PATH_IN_REPO" "$COMMIT_MSG" <<'PY'
import os, sys, traceback
from huggingface_hub import upload_folder
repo_id, fused, path_in_repo, msg = sys.argv[1:5]
try:
    url = upload_folder(
        folder_path=fused,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type="model",
        commit_message=msg,
    )
    print(f"[UPLOAD DONE] {url}")
except Exception:
    print("[UPLOAD FAILED]")
    traceback.print_exc()
    sys.exit(1)
PY

echo "[$(date -Is)] upload script finished"
