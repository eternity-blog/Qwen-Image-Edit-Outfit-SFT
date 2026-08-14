#!/usr/bin/env bash
# Shared path defaults for all shell entrypoints.
# Override by exporting vars before calling scripts, or by copying
# configs/env.example.sh -> configs/env.local.sh and sourcing it.

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QWEN_OUTFIT_ROOT="${QWEN_OUTFIT_ROOT:-$_REPO_ROOT}"

# Optional local overrides (gitignored)
if [[ -f "$QWEN_OUTFIT_ROOT/configs/env.local.sh" ]]; then
  # shellcheck disable=SC1091
  source "$QWEN_OUTFIT_ROOT/configs/env.local.sh"
fi

# Data / models live outside git (large + often NC-licensed).
export DATA_ROOT="${DATA_ROOT:-${QWEN_OUTFIT_ROOT}/data}"
export MODEL_DIR="${MODEL_DIR:-${DATA_ROOT}/models/Qwen-Image-Edit-2511}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${QWEN_OUTFIT_ROOT}/outputs}"
export ENV_DIR="${ENV_DIR:-${CONDA_PREFIX:-}}"

# DiffSynth + IDM teacher (optional installs)
export DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-${DATA_ROOT}/modules/DiffSynth-Studio}"
export IDM_REPO_DIR="${IDM_REPO_DIR:-${DATA_ROOT}/modules/IDM-VTON}"
export IDM_MODEL_DIR="${IDM_MODEL_DIR:-${DATA_ROOT}/models/IDM-VTON}"

export VITON_ROOT="${VITON_ROOT:-${DATA_ROOT}/datasets/qwen_vton/raw/viton_hd}"
export QWEN_VTON_DATA="${QWEN_VTON_DATA:-${DATA_ROOT}/datasets/qwen_vton}"

mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
