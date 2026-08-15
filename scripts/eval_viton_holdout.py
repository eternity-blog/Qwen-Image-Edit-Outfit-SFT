#!/usr/bin/env python3
"""eval_viton_holdout.py — compare models on the held-out VITON-HD test split.

Training used `synth/idm_unpaired_train` (VITON-HD *train*). This script evaluates on
`synth/idm_unpaired` (VITON-HD *test*), which no training step ever saw, and puts the
IDM-VTON teacher output next to each model so you can see whether the student learned
the garment-swap behaviour rather than just memorising.

Panels per sample: person | garment | IDM teacher | <model A> | <model B> ...

The prompt is rebuilt with the *same* builder and the *same* arguments the training
metadata used (`convert_idm_synth_to_qwen_edit_v2.py`: category=upper, facing=front,
overlay=none, 1 product ref), so eval and training stay on one prompt distribution.

Example:
    python scripts/eval_viton_holdout.py \\
        --model base=/data/agent/hf_models/Qwen/Qwen-Image-Edit-2511 \\
        --model full_sft=/data/agent/hf_models/Qwen/Qwen-Image-Edit-Outfit-2511-SFT \\
        --out-dir $OUTPUT_ROOT/viton_holdout_eval --n 6 --cpu-offload
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import prompts_train_v2 as pv2  # noqa: E402
import zero_shot_compare as z  # noqa: E402


def pick_rows(manifest: Path, viton_root: Path, synth_images: Path, n: int, seed: int):
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        person = viton_root / rec["person"]
        cloth = viton_root / rec["cloth"]
        teacher = synth_images / rec["out_name"]
        if person.is_file() and cloth.is_file() and teacher.is_file():
            rows.append(
                {
                    "id": Path(rec["out_name"]).stem,
                    "person": str(person),
                    "cloth": str(cloth),
                    "teacher": str(teacher),
                }
            )
    if not rows:
        raise SystemExit(f"no usable rows from {manifest}")
    random.Random(seed).shuffle(rows)
    return rows[:n]


def grid(panels: list[tuple[str, Image.Image]], title: str) -> Image.Image:
    labelled = [z.label_image(img, name) for name, img in panels]
    gap = 8
    total_w = sum(p.size[0] for p in labelled) + gap * (len(labelled) - 1)
    total_h = max(p.size[1] for p in labelled) + 36
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0), font=ImageFont.load_default())
    x = 0
    for p in labelled:
        canvas.paste(p, (x, 36))
        x += p.size[0] + gap
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="/data/agent/hf_models/datasets/qwen_vton/synth/idm_unpaired/manifest.jsonl",
    )
    ap.add_argument(
        "--viton-root",
        default="/data/agent/hf_models/datasets/qwen_vton/raw/viton_hd",
    )
    ap.add_argument(
        "--synth-images",
        default="/data/agent/hf_models/datasets/qwen_vton/synth/idm_unpaired/images",
    )
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeatable; evaluated in the given order",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--size", default="768x1024", help="WxH; VITON native, under the 1MP train cap")
    ap.add_argument("--true-cfg-scale", type=float, default=4.0)
    ap.add_argument("--max-sequence-length", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cpu-offload", action="store_true", default=False)
    ap.add_argument("--category", default="upper")
    ap.add_argument("--facing", default="front")
    args = ap.parse_args()

    models = []
    for spec in args.model:
        if "=" not in spec:
            raise SystemExit(f"--model expects NAME=PATH, got {spec}")
        name, path = spec.split("=", 1)
        if not (Path(path) / "model_index.json").is_file():
            raise SystemExit(f"{name}: no model_index.json under {path}")
        models.append((name, Path(path)))

    width, height = (int(v) for v in args.size.lower().split("x"))
    out = Path(args.out_dir)
    (out / "grids").mkdir(parents=True, exist_ok=True)

    rows = pick_rows(
        Path(args.manifest), Path(args.viton_root), Path(args.synth_images), args.n, args.seed
    )
    prompt = pv2.build_train_v2_prompt(
        product_text=pv2.default_product_text(args.category),
        product_visual_facts=pv2.default_visual_facts(args.category, facing=args.facing),
        facing=args.facing,
        overlay_placement="none",
        is_tail=False,
        n_product_refs=1,
    )
    (out / "prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"samples={len(rows)} prompt_chars={len(prompt)} size={width}x{height}", flush=True)
    for r in rows:
        print("  ", r["id"], flush=True)

    results: dict[str, dict] = {}
    for name, path in models:
        model_out = out / name
        model_out.mkdir(parents=True, exist_ok=True)
        pipe = z.load_pipeline(path, args.device, torch.bfloat16, cpu_offload=args.cpu_offload)
        per_model = {}
        for i, r in enumerate(rows, 1):
            dest = model_out / f"{r['id']}.jpg"
            if dest.is_file():
                print(f"[{name} {i}/{len(rows)}] skip {r['id']} (exists)", flush=True)
                continue
            person = z.load_rgb(r["person"])
            cloth = z.load_rgb(r["cloth"])
            t0 = time.time()
            img = z.run_edit(
                pipe,
                source=person,
                products=[cloth],
                prompt=prompt,
                steps=args.steps,
                seed=args.seed,
                true_cfg_scale=args.true_cfg_scale,
                width=width,
                height=height,
                negative_prompt="",
                max_sequence_length=args.max_sequence_length,
            )
            img.save(dest, quality=95)
            teacher = z.load_rgb(r["teacher"])
            mad_person, hist_person = z.mad_and_hist(person, img)
            mad_teacher, hist_teacher = z.mad_and_hist(teacher, img)
            per_model[r["id"]] = {
                "secs": round(time.time() - t0, 1),
                "mad_vs_person": round(mad_person, 3),
                "hist_vs_person": round(hist_person, 4),
                "mad_vs_teacher": round(mad_teacher, 3),
                "hist_vs_teacher": round(hist_teacher, 4),
            }
            print(
                f"[{name} {i}/{len(rows)}] {r['id']} {per_model[r['id']]['secs']}s "
                f"MAD person/teacher={mad_person:.1f}/{mad_teacher:.1f}",
                flush=True,
            )
        results[name] = per_model
        del pipe
        torch.cuda.empty_cache()

    for r in rows:
        panels = [
            ("person (source)", z.load_rgb(r["person"])),
            ("garment (ref)", z.load_rgb(r["cloth"])),
            ("IDM-VTON teacher", z.load_rgb(r["teacher"])),
        ]
        for name, _ in models:
            p = out / name / f"{r['id']}.jpg"
            if p.is_file():
                panels.append((name, z.load_rgb(p)))
        g = grid(panels, f"VITON-HD holdout {r['id']}  steps={args.steps} seed={args.seed}")
        gp = out / "grids" / f"{r['id']}_compare.jpg"
        g.save(gp, quality=92)
        print("wrote", gp, flush=True)

    (out / "metrics.json").write_text(
        json.dumps({"rows": rows, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# VITON-HD holdout eval (test split — unseen in training)",
        "",
        f"- samples: {len(rows)}, steps {args.steps}, seed {args.seed}, size {width}x{height}",
        f"- prompt: full Outfit v2, {len(prompt)} chars (same builder/args as training metadata)",
        "- `mad_vs_person`: lower = kept more of the source frame",
        "- `mad_vs_teacher`: lower = closer to the IDM-VTON teacher it was trained to imitate",
        "",
        "| model | mean MAD vs person | mean MAD vs teacher | mean hist vs teacher | mean secs |",
        "|---|---|---|---|---|",
    ]
    for name, per in results.items():
        if not per:
            continue
        k = len(per)
        lines.append(
            f"| {name} "
            f"| {sum(v['mad_vs_person'] for v in per.values()) / k:.2f} "
            f"| {sum(v['mad_vs_teacher'] for v in per.values()) / k:.2f} "
            f"| {sum(v['hist_vs_teacher'] for v in per.values()) / k:.4f} "
            f"| {sum(v['secs'] for v in per.values()) / k:.0f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("DONE ->", out)


if __name__ == "__main__":
    main()
