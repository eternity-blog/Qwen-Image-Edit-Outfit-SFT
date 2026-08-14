#!/usr/bin/env python3
"""Compose case02 grids: source | GPT | base | fused.

Expects two zero_shot_compare output dirs with matching qwen/<tag>.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def label_image(img: Image.Image, label: str, width: int = 320) -> Image.Image:
    im = img.convert("RGB").copy()
    w, h = im.size
    nh = max(1, int(round(h * (width / float(w)))))
    im = im.resize((width, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, nh + 28), (245, 245, 245))
    canvas.paste(im, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, 28], fill=(30, 30, 30))
    font = ImageFont.load_default()
    draw.text((8, 8), label, fill=(255, 255, 255), font=font)
    return canvas


def make_grid(
    source: Image.Image,
    gpt: Image.Image,
    base: Image.Image,
    fused: Image.Image,
    title: str,
) -> Image.Image:
    panels = [
        label_image(source, "source"),
        label_image(gpt, "GPT Image 2"),
        label_image(base, "Qwen base"),
        label_image(fused, "Qwen fused LoRA"),
    ]
    gap = 8
    total_w = sum(p.size[0] for p in panels) + gap * (len(panels) - 1)
    total_h = max(p.size[1] for p in panels) + 36
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    x = 0
    for p in panels:
        canvas.paste(p, (x, 36))
        x += p.size[0] + gap
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="zero_shot out dir for base model")
    ap.add_argument("--fused-dir", required=True, help="zero_shot out dir for fused model")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    fused_dir = Path(args.fused_dir)
    out_dir = Path(args.out_dir)
    grid_dir = out_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    base_meta = json.loads((base_dir / "run_meta.json").read_text(encoding="utf-8"))
    samples = base_meta.get("samples") or []
    rows = []
    for s in samples:
        tag = f"{int(s['shot_index']):02d}_{s['role']}"
        base_img = base_dir / "qwen" / f"{tag}.png"
        fused_img = fused_dir / "qwen" / f"{tag}.png"
        if not base_img.is_file() or not fused_img.is_file():
            print(f"skip missing {tag}")
            continue
        source = Image.open(s["source_path"]).convert("RGB")
        gpt = Image.open(s["gpt_path"]).convert("RGB")
        base = Image.open(base_img).convert("RGB")
        fused = Image.open(fused_img).convert("RGB")
        grid = make_grid(source, gpt, base, fused, title=f"case{s['case_id']} {tag} base vs fused")
        gpath = grid_dir / f"{tag}_base_vs_fused.jpg"
        grid.save(gpath, quality=92)
        rows.append({"tag": tag, "grid": str(gpath), "base": str(base_img), "fused": str(fused_img)})
        print("wrote", gpath)

    summary = out_dir / "summary.md"
    lines = [
        "# case02 fused vs base smoke",
        "",
        f"- base model dir: `{base_meta.get('model_dir')}`",
        f"- fused out: `{fused_dir}`",
        f"- prompt_mode: `{base_meta.get('prompt_mode')}`",
        f"- fair: `{base_meta.get('fair')}`",
        f"- steps/seed: {base_meta.get('steps')}/{base_meta.get('seed')}",
        "",
        "## Grids",
        "",
    ]
    for r in rows:
        lines.append(f"- `{r['tag']}`: `{r['grid']}`")
    lines.extend(
        [
            "",
            "## How to read",
            "",
            "- Compare **Qwen base** vs **Qwen fused LoRA** for locality / garment adherence.",
            "- GPT is reference only; this LoRA was VITON reconstruction prior, not case02 SFT.",
            "",
        ]
    )
    summary.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "compose_meta.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("DONE", summary)


if __name__ == "__main__":
    main()
