#!/usr/bin/env python3
"""logs_to_wandb.py — Backfill training metrics into a wandb run.

DiffSynth's training loop (diffsynth/diffusion/runner.py) drives progress with
tqdm and routes loss ONLY through ModelLogger.on_step_end(loss=loss). With
--enable_tensorboard_log (default in train_full_sft_zero3.sh), DiffSynth writes
a local SummaryWriter to <output_path>/tensorboard_log/. This script reads those
TensorBoard event files as the canonical source and pushes the full loss curve
to a wandb run — so you get proper metrics after the fact ("跑完后补充wandb").

Also attaches a launch_config.json as the run config, and optionally parses a
`nvidia-smi ... -l 60` CSV log for GPU mem/util curves.

Usage (run AFTER training, in the training env):
    $ENV_DIR/bin/python -m pip install wandb          # once
    $ENV_DIR/bin/python -m wandb login                # or export WANDB_API_KEY=...

    $ENV_DIR/bin/python scripts/logs_to_wandb.py \
        --tb-dir   $OUTPUT_ROOT/qwen_vton_full_sft/dit_full/tensorboard_log \
        --config-json $OUTPUT_ROOT/qwen_vton_full_sft/launch_config.json \
        --nvidia-smi-log $OUTPUT_ROOT/qwen_vton_full_sft/logs/nvidia_smi.csv \
        --project  qwen-outfit-full-sft \
        --run-name full_sft_8gpu_zero3

Falls back to regex-parsing the text log (--log) if TensorBoard events are absent.
Idempotent: pass --resume-id <wandb_run_id> to append to an existing run.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

try:
    import wandb
except ImportError:
    sys.exit("wandb not installed. Run: $ENV_DIR/bin/python -m pip install wandb")


# ---- fallback: regex for loss/step lines (DiffSynth doesn't print these by
#      default, but some forks print "step N: loss X" or "Step N/M: loss=X") ----
STEP_RE = re.compile(
    r"(?:step|Step)\s+(\d+)(?:\s*/\s*(\d+))?"
    r"[^\d]*"
    r"(?:loss[=:]?\s*)?([0-9]+\.?[0-9]*(?:e-?\d+)?)"
    r"(?:.*?lr[=:]?\s*([0-9]+\.?[0-9]*(?:e-?\d+)?))?",
    re.IGNORECASE,
)


def read_tensorboard_scalars(logdir: Path):
    """Read all scalar tags from a TensorBoard log dir. Returns {tag: [(step, value, wall_time)]}."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print(f"[warn] tensorboard not installed; cannot read events from {logdir}")
        return {}
    ea = event_accumulator.EventAccumulator(
        str(logdir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(e.step, e.value, e.wall_time) for e in ea.Scalars(tag)]
    return out


def parse_loss_lines(log_path: Path):
    """Fallback: parse loss/step/lr lines from a text log."""
    rows = []
    if not log_path or not log_path.exists():
        return rows
    with log_path.open("r", errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if not m:
                continue
            try:
                rows.append({
                    "step": int(m.group(1)),
                    "total_steps": int(m.group(2)) if m.group(2) else None,
                    "loss": float(m.group(3)),
                    "lr": float(m.group(4)) if m.group(4) else None,
                })
            except (TypeError, ValueError):
                continue
    return rows


def parse_nvidia_smi(path: Path):
    """Parse a `nvidia-smi --query-gpu=... --format=csv -l N` log into rows."""
    if not path or not path.exists():
        return []
    rows = []
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return []
        h = [c.strip().lower() for c in header]
        idx = next((i for i, c in enumerate(h) if "index" in c), None)
        mu = next((i for i, c in enumerate(h) if "memory.used" in c), None)
        mt = next((i for i, c in enumerate(h) if "memory.total" in c), None)
        ut = next((i for i, c in enumerate(h) if "utilization.gpu" in c), None)
        for r in reader:
            try:
                rec = {}
                if idx is not None:
                    rec["gpu"] = int(re.sub(r"\D", "", r[idx]) or 0)
                if mu is not None:
                    rec["mem_used_mib"] = int(re.sub(r"\D", "", r[mu]) or 0)
                if mt is not None:
                    rec["mem_total_mib"] = int(re.sub(r"\D", "", r[mt]) or 0)
                if ut is not None:
                    rec["gpu_util_pct"] = int(re.sub(r"\D", "", r[ut]) or 0)
                if rec:
                    rows.append(rec)
            except (ValueError, IndexError):
                continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb-dir", help="TensorBoard event dir (PRIMARY source)")
    ap.add_argument("--log", help="text train log (FALLBACK if no --tb-dir)")
    ap.add_argument("--project", default="qwen-outfit-full-sft")
    ap.add_argument("--run-name", default="full_sft_8gpu_zero3")
    ap.add_argument("--config-json", help="launch_config.json to log as run config")
    ap.add_argument("--nvidia-smi-log", help="optional nvidia-smi csv log")
    ap.add_argument("--resume-id", help="wandb run id to resume/append")
    ap.add_argument("--dry-run", action="store_true", help="parse only, don't push")
    args = ap.parse_args()

    scalars = {}
    if args.tb_dir:
        tb = Path(args.tb_dir)
        if tb.exists():
            scalars = read_tensorboard_scalars(tb)
            print(f"[tb] read {len(scalars)} scalar tag(s) from {tb}")
            for tag, pts in scalars.items():
                print(f"     {tag}: {len(pts)} points", pts[:1], "...", pts[-1:] if pts else [])
        else:
            print(f"[warn] --tb-dir {tb} does not exist")

    # fallback to text log if tensorboard yielded nothing
    if not scalars and args.log:
        rows = parse_loss_lines(Path(args.log))
        print(f"[log] parsed {len(rows)} loss lines from {args.log}")
        if rows:
            scalars["loss"] = [(r["step"], r["loss"], 0.0) for r in rows]

    n_loss = len(scalars.get("loss", []))
    print(f"total loss points: {n_loss}")
    if n_loss == 0:
        print("[warn] no loss data found — enable --enable_tensorboard_log during training")

    config = {}
    if args.config_json:
        cp = Path(args.config_json)
        if cp.exists():
            config = json.loads(cp.read_text())
            print(f"config keys: {list(config.keys())}")

    nvmi = parse_nvidia_smi(Path(args.nvidia_smi_log) if args.nvidia_smi_log else None)
    print(f"nvidia-smi rows: {len(nvmi)}")

    if args.dry_run:
        print("[dry-run] not pushing to wandb")
        return

    run = wandb.init(
        project=args.project,
        name=args.run_name,
        id=args.resume_id,
        resume="allow" if args.resume_id else None,
        config=config,
        reinit=True,
    )
    # log all scalar tags (loss, and any others present)
    for tag, pts in scalars.items():
        wandb_tag = f"train/{tag}" if not tag.startswith("train/") else tag
        for step, value, _wt in pts:
            wandb.log({wandb_tag: value}, step=step)
    # gpu metrics on a separate counter axis
    for i, m in enumerate(nvmi):
        wandb.log({**m, "_gpu_step": i})
    run.finish()
    print(f"wandb run '{args.run_name}' updated: {run.url}")


if __name__ == "__main__":
    main()
