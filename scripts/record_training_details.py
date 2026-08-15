#!/usr/bin/env python3
"""record_training_details.py — machine-readable snapshot of a training run.

Captures config + environment + dataset stats + the loss curve, so a run can be
reviewed after the fact and `logs_to_wandb.py` has a config to attach.

Loss comes from the TensorBoard events DiffSynth writes when the training script
runs with `--enable_tensorboard_log` (on by default in train_full_sft_zero3.sh).
DiffSynth routes loss only through ModelLogger, so scraping the text log finds
nothing — point --tb-dir at `<output_path>/tensorboard_log`.

Writes `training_details.json` and `training_details.md` into --out-dir.

Usage:
    python scripts/record_training_details.py \\
        --log $OUTPUT_ROOT/qwen_vton_full_sft/logs/train_full_sft.log \\
        --metadata $METADATA \\
        --tb-dir $OUTPUT_ROOT/qwen_vton_full_sft/dit_full/tensorboard_log \\
        --out-dir $OUTPUT_ROOT/qwen_vton_full_sft
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=30
        ).strip()
    except Exception as e:  # noqa: BLE001 - best-effort probe, never fail the snapshot
        return f"<{type(e).__name__}: {e}>"


def pkg_version(mod: str) -> str:
    """Version from the *current* interpreter, not a guessed env path."""
    try:
        return __import__(mod).__version__
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}>"


def read_tb_loss(tb_dir: Path) -> list[tuple[int, float]]:
    if not tb_dir or not tb_dir.is_dir():
        return []
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("[warn] tensorboard not installed; loss curve will be empty")
        return []
    ea = event_accumulator.EventAccumulator(
        str(tb_dir), size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    tag = "loss" if "loss" in tags else (tags[0] if tags else None)
    if tag is None:
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def summarise_loss(points: list[tuple[int, float]], window: int = 100) -> dict:
    if not points:
        return {"n_points": 0}
    vals = [v for _, v in points]
    head = vals[:window]
    tail = vals[-window:]
    drop = (
        (statistics.fmean(head) - statistics.fmean(tail)) / statistics.fmean(head) * 100.0
        if head and statistics.fmean(head)
        else None
    )
    return {
        "n_points": len(vals),
        "first": vals[0],
        "last": vals[-1],
        "mean": statistics.fmean(vals),
        "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        f"mean_first_{window}": statistics.fmean(head),
        f"mean_last_{window}": statistics.fmean(tail),
        "window_mean_drop_pct": drop,
        "note": (
            "Single-step loss is noise-dominated in diffusion training (random timestep "
            "and noise per step). Judge progress by the window means, not first/last."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="train_full_sft.log (for provenance)")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tb-dir", default="", help="<output_path>/tensorboard_log")
    ap.add_argument("--launch-cmd", default="", help="the exact command used to launch")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = Path(args.log)
    meta = Path(args.metadata)

    n_samples, sample = -1, None
    if meta.exists():
        data = json.loads(meta.read_text())
        n_samples = len(data)
        sample = data[0] if data else None

    loss_points = read_tb_loss(Path(args.tb_dir)) if args.tb_dir else []
    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "env": {
            k: os.environ.get(k)
            for k in ("ENV_DIR", "MODEL_DIR", "DIFFSYNTH_DIR", "QWEN_VTON_DATA", "OUTPUT_ROOT")
        },
        "gpu": {
            "count": sh("nvidia-smi -L | wc -l"),
            "list": sh("nvidia-smi -L").splitlines(),
            "current": sh(
                "nvidia-smi --query-gpu=index,name,memory.used,memory.total,"
                "utilization.gpu --format=csv,noheader"
            ).splitlines(),
        },
        "dataset": {
            "metadata": str(meta),
            "n_samples": n_samples,
            "sample_prompt_chars": len(sample["prompt"]) if sample and "prompt" in sample else None,
        },
        "training": {
            "log": str(log),
            "log_exists": log.exists(),
            "launch_cmd": args.launch_cmd,
            "tb_dir": args.tb_dir,
            "loss": summarise_loss(loss_points),
        },
        "versions": {m: pkg_version(m) for m in ("torch", "deepspeed", "accelerate", "diffusers")},
    }

    (out / "training_details.json").write_text(json.dumps(rec, indent=2, default=str))

    loss = rec["training"]["loss"]
    md = [
        "# Training details",
        "",
        f"- captured: {rec['timestamp']}",
        f"- samples: {n_samples}, prompt chars: {rec['dataset']['sample_prompt_chars']}",
        f"- GPUs: {rec['gpu']['count']}",
        f"- versions: {rec['versions']}",
        "",
    ]
    if loss.get("n_points"):
        md += [
            "## Loss",
            "",
            f"- points: {loss['n_points']}",
            f"- window mean: {loss['mean_first_100']:.5f} -> {loss['mean_last_100']:.5f}"
            f" ({loss['window_mean_drop_pct']:.1f}% drop)",
            f"- overall mean {loss['mean']:.5f} (stdev {loss['stdev']:.5f})",
            "",
            f"> {loss['note']}",
        ]
    else:
        md += ["## Loss", "", "No TensorBoard scalars found — pass --tb-dir."]
    (out / "training_details.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(rec, indent=2, default=str))
    print(f"\nwrote {out / 'training_details.json'} and {out / 'training_details.md'}")


if __name__ == "__main__":
    main()
