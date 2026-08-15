#!/usr/bin/env python3
"""record_training_details.py — capture a detailed, machine-readable snapshot of
the training run (config + env + dataset stats + latest loss/step + gpu mem).

Writes a JSON + a markdown summary so the user can review training details after
the run, and so logs_to_wandb.py has a config to attach.

Usage:
    python scripts/record_training_details.py \
        --log $OUTPUT_ROOT/qwen_vton_full_sft/logs/train_full_sft.log \
        --metadata $METADATA \
        --out-dir $OUTPUT_ROOT/qwen_vton_full_sft
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path


STEP_RE = re.compile(
    r"(?:step|Step)\s+(\d+)(?:\s*/\s*(\d+))?[^\d]*(?:loss[=:]?\s*)?([0-9]+\.?[0-9]*(?:e-?\d+)?)"
    r"(?:.*?lr[=:]?\s*([0-9]+\.?[0-9]*(?:e-?\d+)?))?",
    re.IGNORECASE,
)


def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=30)
    except Exception as e:
        return f"<{e}>"


def parse_loss_tail(log: Path):
    rows = []
    if not log.exists():
        return rows
    with log.open("r", errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                try:
                    rows.append({
                        "step": int(m.group(1)),
                        "total": int(m.group(2)) if m.group(2) else None,
                        "loss": float(m.group(3)),
                        "lr": float(m.group(4)) if m.group(4) else None,
                    })
                except (TypeError, ValueError):
                    pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--launch-cmd", default="", help="the exact command used to launch")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = Path(args.log)
    meta = Path(args.metadata)

    n_samples = -1
    sample = None
    if meta.exists():
        data = json.loads(meta.read_text())
        n_samples = len(data)
        sample = data[0] if data else None

    rows = parse_loss_tail(log)
    last = rows[-1] if rows else None

    nvidia = sh("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader")

    rec = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "repo_root": "/opt/lx/Qwen-Image-Edit-Outfit-SFT",
        "env_dir": os.environ.get("ENV_DIR"),
        "model_dir": os.environ.get("MODEL_DIR"),
        "diffsynth_dir": os.environ.get("DIFFSYNTH_DIR"),
        "qwen_vton_data": os.environ.get("QWEN_VTON_DATA"),
        "output_root": os.environ.get("OUTPUT_ROOT"),
        "gpu": {
            "count": sh("nvidia-smi -L | wc -l").strip(),
            "list": sh("nvidia-smi -L").strip().splitlines(),
            "current": nvidia.strip().splitlines(),
        },
        "dataset": {
            "metadata": str(meta),
            "n_samples": n_samples,
            "sample_prompt_chars": len(sample["prompt"]) if sample and "prompt" in sample else None,
        },
        "training": {
            "log": str(log),
            "launch_cmd": args.launch_cmd,
            "loss_steps_parsed": len(rows),
            "latest_step": last,
        },
        "env_versions": {
            "torch": sh("$ENV_DIR/bin/python -c 'import torch;print(torch.__version__)'").strip(),
            "deepspeed": sh("$ENV_DIR/bin/python -c 'import deepspeed;print(deepspeed.__version__)'").strip(),
            "accelerate": sh("$ENV_DIR/bin/python -c 'import accelerate;print(accelerate.__version__)'").strip(),
            "diffusers": sh("$ENV_DIR/bin/python -c 'import diffusers;print(diffusers.__version__)'").strip(),
        },
    }

    (out / "training_details.json").write_text(json.dumps(rec, indent=2, default=str))
    print(json.dumps(rec, indent=2, default=str))


if __name__ == "__main__":
    main()
