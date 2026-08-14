#!/usr/bin/env python3
"""Extract DressCode.zip (possibly misnamed .tar) with progress logging."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import zipfile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--zip",
        default="/data/agent/hf_models/datasets/qwen_vton/raw/dresscode/DressCode.zip",
    )
    ap.add_argument(
        "--out-extract",
        default="/data/agent/hf_models/datasets/qwen_vton/raw/dresscode/extract",
    )
    ap.add_argument(
        "--final-dir",
        default="/data/agent/hf_models/datasets/qwen_vton/raw/dresscode/DressCode",
    )
    ap.add_argument(
        "--ready-flag",
        default="/data/agent/hf_models/datasets/qwen_vton/logs/dresscode_ready",
    )
    args = ap.parse_args()

    zpath = Path(args.zip)
    if not zpath.is_file():
        alt = zpath.with_suffix(".tar")
        if alt.is_file():
            zpath = alt.rename(zpath)
            print("renamed", alt, "->", zpath)
        else:
            raise SystemExit(f"missing zip {args.zip}")

    out = Path(args.out_extract)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("zip", zpath, "size_gb", round(zpath.stat().st_size / 1e9, 2), flush=True)
    with zipfile.ZipFile(zpath) as zf:
        infos = zf.infolist()
        n = len(infos)
        print("n_members", n, flush=True)
        print("sample", [i.filename for i in infos[:8]], flush=True)
        for i, info in enumerate(infos):
            zf.extract(info, out)
            if i % 5000 == 0 or i + 1 == n:
                print(f"extracted {i+1}/{n} {info.filename[:100]}", flush=True)

    final = Path(args.final_dir)
    if final.exists():
        shutil.rmtree(final)
    nested = out / "DressCode"
    if nested.is_dir():
        nested.rename(final)
    else:
        final.mkdir(parents=True, exist_ok=True)
        for child in out.iterdir():
            dest = final / child.name
            if not dest.exists():
                child.rename(dest)

    ready = Path(args.ready_flag)
    ready.parent.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    ready.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    print("DONE final", final, "children", sorted(p.name for p in final.iterdir())[:40], flush=True)


if __name__ == "__main__":
    main()
