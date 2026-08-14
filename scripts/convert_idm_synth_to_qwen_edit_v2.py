#!/usr/bin/env python3
"""Convert IDM-VTON synth pairs to Qwen-Edit metadata with FULL Outfit v2 prompts.

Does not re-run IDM. Reuses existing synth images; only rewrites prompts/metadata.
Multi-reference edit_image lists are deferred (see ../TODO.md).

Output layout:
  <out-dir>/dataset_base/{viton_hd -> ..., idm_synth -> ...}
  <out-dir>/metadata_train.jsonl
  <out-dir>/metadata_val.jsonl
  <out-dir>/stats.json
  <out-dir>/prompt_audit_sample.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import prompts_train_v2 as pv2  # noqa: E402


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
        help="Parent of viton_hd/",
    )
    ap.add_argument(
        "--out-dir",
        default="/data/agent/hf_models/datasets/qwen_vton/converted_idm_synth_train_v2",
    )
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--facing",
        default="front",
        choices=["front", "back", "side", ""],
        help="Facing block for v2 prompt; VITON default front",
    )
    ap.add_argument("--category", default="upper")
    ap.add_argument("--audit-n", type=int, default=50)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only convert first N valid rows (debug)",
    )
    args = ap.parse_args()

    synth = Path(args.synth_dir)
    raw = Path(args.raw_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    staging = out / "dataset_base"
    staging.mkdir(parents=True, exist_ok=True)
    viton_link = staging / "viton_hd"
    if not viton_link.exists():
        viton_link.symlink_to(raw / "viton_hd")
    synth_img_link = staging / "idm_synth"
    if synth_img_link.exists() or synth_img_link.is_symlink():
        synth_img_link.unlink()
    synth_img_link.symlink_to(synth / "images")

    facing = args.facing
    product_text = pv2.default_product_text(args.category)
    visual_facts = pv2.default_visual_facts(args.category, facing=facing)

    manifest = synth / "manifest.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"missing {manifest}")

    rows: List[dict] = []
    skipped = {"person": 0, "cloth": 0, "synth": 0}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        person = rec["person"]
        cloth = rec["cloth"]
        out_name = rec["out_name"]
        if not (raw / "viton_hd" / person).is_file():
            skipped["person"] += 1
            continue
        if not (raw / "viton_hd" / cloth).is_file():
            skipped["cloth"] += 1
            continue
        if not (synth / "images" / out_name).is_file():
            skipped["synth"] += 1
            continue

        prompt = pv2.build_train_v2_prompt(
            product_text=product_text,
            product_visual_facts=visual_facts,
            facing=facing,
            overlay_placement="none",
            is_tail=False,
            n_product_refs=1,
        )
        rows.append(
            {
                "image": f"idm_synth/{out_name}",
                "edit_image": [f"viton_hd/{person}", f"viton_hd/{cloth}"],
                "prompt": prompt,
                "source": "idm_vton_synth_v2prompt",
                "category": args.category,
                "person": f"viton_hd/{person}",
                "cloth": f"viton_hd/{cloth}",
                "id": Path(out_name).stem,
                "prompt_style": "outfit_garment_only_v2_full",
                "facing": facing,
                "overlay_placement": "none",
                "n_product_refs": 1,
                "prompt_chars": len(prompt),
            }
        )
        if args.limit > 0 and len(rows) >= args.limit:
            break

    if not rows:
        raise SystemExit(f"no valid rows; skipped={skipped}")

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

    chars = [r["prompt_chars"] for r in rows]
    chars_sorted = sorted(chars)
    audit = []
    for r in rows[: max(0, args.audit_n)]:
        audit.append(
            {
                "id": r["id"],
                "prompt_chars": r["prompt_chars"],
                "edit_image": r["edit_image"],
                "image": r["image"],
                "prompt_head": r["prompt"][:400],
                "prompt_tail": r["prompt"][-300:],
            }
        )
    (out / "prompt_audit_sample.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Also dump one full prompt for human read
    (out / "prompt_example_full.txt").write_text(rows[0]["prompt"], encoding="utf-8")

    stats = {
        "n_total": len(rows),
        "n_train": len(train),
        "n_val": len(val),
        "skipped": skipped,
        "dataset_base_path": str(staging),
        "synth_dir": str(synth),
        "prompt_style": "outfit_garment_only_v2_full",
        "prompt_chars_min": min(chars),
        "prompt_chars_max": max(chars),
        "prompt_chars_p50": chars_sorted[len(chars_sorted) // 2],
        "facing": facing,
        "overlay_placement": "none",
        "n_product_refs": 1,
        "multi_ref": False,
        "note": (
            "Full live v2 garment-only prompts; 2-image inputs only. "
            "Multi-ref deferred — see Qwen-Image-Edit/TODO.md. "
            "Images reused from IDM synth (no teacher re-run)."
        ),
        "license_note": (
            "VITON-HD CC BY-NC 4.0; IDM-VTON CC BY-NC-SA 4.0 — research/non-commercial"
        ),
    }
    (out / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("stats", json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
