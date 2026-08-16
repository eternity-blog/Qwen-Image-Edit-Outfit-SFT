#!/usr/bin/env python3
"""compose_case02_matrix.py — one grid per shot holding every case02 configuration.

The business-domain runs accumulated across several output dirs (one per model,
plus a reference-count ablation), so comparing them meant opening four grids side
by side. This collapses them into a single strip per shot, in a fixed order, with
the MAD-vs-source printed under each panel.

    python scripts/compose_case02_matrix.py \\
        --run-root $OUTPUT_ROOT/qwen_kf_zeroshot \\
        --out-dir  $OUTPUT_ROOT/qwen_kf_zeroshot/case02_matrix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import zero_shot_compare as z  # noqa: E402

# label -> (run dir, which sub-run: base|idm)
# order here is the panel order
LAYOUT = [
    ("base  3 refs", "case02_v2_fullsft_vs_base_0815", "base"),
    ("idm_lora_v1  3 refs", "case02_v2_lora_vs_base_0815", "idm"),
    ("lora_v2_lr1e-4  3 refs", "case02_lorav2_0816", "idm"),
    ("full_sft  3 refs", "case02_v2_fullsft_vs_base_0815", "idm"),
    ("base  1 ref", "case02_fullsft_1ref_0816", "base"),
    ("full_sft  1 ref", "case02_fullsft_1ref_0816", "idm"),
]
PANEL_W = 300
LABEL_H = 30


def panel(img: Image.Image, label: str, sub: str = "") -> Image.Image:
    im = img.convert("RGB")
    w, h = im.size
    nh = max(1, int(round(h * (PANEL_W / float(w)))))
    im = im.resize((PANEL_W, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (PANEL_W, nh + LABEL_H), (245, 245, 245))
    canvas.paste(im, (0, LABEL_H))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, PANEL_W, LABEL_H], fill=(30, 30, 30))
    f = ImageFont.load_default()
    d.text((6, 3), label, fill=(255, 255, 255), font=f)
    if sub:
        d.text((6, 16), sub, fill=(170, 220, 170), font=f)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shots", default="00_start,01_start")
    args = ap.parse_args()

    root = Path(args.run_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for shot in [s.strip() for s in args.shots.split(",") if s.strip()]:
        # source and GPT come from any run's meta; they are identical across runs
        meta_path = root / LAYOUT[0][1] / LAYOUT[0][2] / "run_meta.json"
        meta = json.loads(meta_path.read_text())
        sample = next(
            (
                s
                for s in (meta.get("samples") or [])
                if f"{int(s['shot_index']):02d}_{s['role']}" == shot
            ),
            None,
        )
        if sample is None:
            print(f"skip {shot}: not in {meta_path}")
            continue

        source = z.load_rgb(sample["source_path"])
        panels = [panel(source, "source frame")]
        gpt_path = Path(sample.get("gpt_path") or "")
        if gpt_path.is_file():
            gpt = z.load_rgb(gpt_path)
            mad, _ = z.mad_and_hist(source, gpt)
            panels.append(panel(gpt, "GPT Image 2  3 refs", f"MAD vs source {mad:.1f}"))

        for label, run, sub in LAYOUT:
            p = root / run / sub / "qwen" / f"{shot}.png"
            if not p.is_file():
                print(f"  missing {p}")
                continue
            img = z.load_rgb(p)
            mad, _ = z.mad_and_hist(source, img)
            panels.append(panel(img, label, f"MAD vs source {mad:.1f}"))

        gap, top = 8, 34
        w = len(panels) * PANEL_W + (len(panels) - 1) * gap
        h = max(p.size[1] for p in panels) + top
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        d.text(
            (8, 10),
            f"case02 {shot} — live Outfit v2 prompt, seed 0, 40 steps, 720x1280."
            "  '3 refs' = source + 3 product images (what production sends);"
            "  '1 ref' = source + 1 (what training used).",
            fill=(0, 0, 0),
            font=ImageFont.load_default(),
        )
        x = 0
        for p in panels:
            canvas.paste(p, (x, top))
            x += PANEL_W + gap
        dest = out / f"{shot}_matrix.jpg"
        canvas.save(dest, quality=92)
        print("wrote", dest)


if __name__ == "__main__":
    main()
