#!/usr/bin/env python3
"""Convert IDM-VTON synthetic unpaired try-on outputs into Qwen-Image-Edit metadata.

Each synth sample becomes a REAL edit pair:
  edit_image = [person_wearing_A, garment_B]
  image      = synth_person_wearing_B   # teacher GT
  prompt     = locality / garment-swap instruction
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


PROMPTS = [
    (
        "Edit image 1 only. Replace the clothing on the person in image 1 with the "
        "garment shown in image 2. Keep the person's identity, face, hair, pose, "
        "hands, body shape, camera, and background unchanged. Do not redraw the scene."
    ),
    (
        "图像编辑，不是重新创作。第1张是人物底图，第2张是目标服装参考图。"
        "将第1张人物身上的对应服饰替换为第2张服装，保持人物身份、姿态、手势、背景和构图不变。"
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--synth-dir",
        required=True,
        help="IDM synth out dir containing manifest.jsonl and images/",
    )
    ap.add_argument(
        "--raw-root",
        default="/data/agent/hf_models/datasets/qwen_vton/raw",
        help="Root that person/cloth relative paths join against (parent of viton_hd)",
    )
    ap.add_argument(
        "--out-dir",
        default="/data/agent/hf_models/datasets/qwen_vton/converted_idm_synth",
    )
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--symlink-images",
        action="store_true",
        help="Also symlink synth images under out_dir/synth_images for a single base path",
    )
    args = ap.parse_args()

    synth = Path(args.synth_dir)
    raw = Path(args.raw_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Unified base for DiffSynth: use a staging root with viton_hd + synth
    # metadata paths relative to staging.
    staging = out / "dataset_base"
    staging.mkdir(parents=True, exist_ok=True)
    viton_link = staging / "viton_hd"
    if not viton_link.exists():
        viton_link.symlink_to(raw / "viton_hd")
    synth_img_link = staging / "idm_synth"
    if not synth_img_link.exists():
        synth_img_link.symlink_to(synth / "images")

    rows: List[dict] = []
    for line in (synth / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        person = rec["person"]  # e.g. test/image/xxx.jpg relative to viton_hd
        cloth = rec["cloth"]
        out_name = rec["out_name"]
        # verify files
        if not (raw / "viton_hd" / person).is_file():
            print("missing person", person)
            continue
        if not (raw / "viton_hd" / cloth).is_file():
            print("missing cloth", cloth)
            continue
        if not (synth / "images" / out_name).is_file():
            print("missing synth", out_name)
            continue
        prompt = PROMPTS[len(rows) % len(PROMPTS)]
        rows.append(
            {
                "image": f"idm_synth/{out_name}",
                "edit_image": [f"viton_hd/{person}", f"viton_hd/{cloth}"],
                "prompt": prompt,
                "source": "idm_vton_synth",
                "category": "upper",
                "person": f"viton_hd/{person}",
                "cloth": f"viton_hd/{cloth}",
                "id": Path(out_name).stem,
            }
        )

    if not rows:
        raise SystemExit("no valid synth rows")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio)) if args.val_ratio > 0 else 0
    val = rows[:n_val]
    train = rows[n_val:]

    def dump(name: str, data: List[dict]) -> None:
        (out / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {name}: {len(data)}")

    dump("metadata_train", train)
    dump("metadata_val", val)
    (out / "metadata.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stats = {
        "n_total": len(rows),
        "n_train": len(train),
        "n_val": len(val),
        "dataset_base_path": str(staging),
        "note": "True cross-garment edit pairs; target is IDM-VTON synth GT",
        "license_note": "IDM-VTON is CC BY-NC-SA 4.0 — research/non-commercial use",
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("stats", stats)


if __name__ == "__main__":
    main()
