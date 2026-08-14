#!/usr/bin/env python3
"""Preview IDM synthetic training pairs: person | cloth | synth GT.

Writes grids under outputs/ so mutagen syncs them back locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def panel(img: Image.Image, label: str, width: int = 360) -> Image.Image:
    im = img.convert("RGB")
    w, h = im.size
    nh = max(1, int(round(h * (width / float(w)))))
    im = im.resize((width, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, nh + 28), (245, 245, 245))
    canvas.paste(im, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, 28], fill=(30, 30, 30))
    draw.text((8, 8), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--converted-dir",
        default="/data/agent/hf_models/datasets/qwen_vton/converted_idm_synth",
    )
    ap.add_argument(
        "--out-dir",
        default="/data/agent/lixiao29/QualityInspection-sync/outputs/qwen_vton_lora/idm_synth_preview",
    )
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    conv = Path(args.converted_dir)
    stats = json.loads((conv / "stats.json").read_text(encoding="utf-8"))
    base = Path(stats["dataset_base_path"])
    rows = json.loads((conv / "metadata_train.json").read_text(encoding="utf-8"))
    out = Path(args.out_dir)
    grid_dir = out / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for i, r in enumerate(rows[: args.limit]):
        person = Image.open(base / r["edit_image"][0]).convert("RGB")
        cloth = Image.open(base / r["edit_image"][1]).convert("RGB")
        target = Image.open(base / r["image"]).convert("RGB")
        panels = [
            panel(person, "edit[0] person (A)"),
            panel(cloth, "edit[1] cloth (B)"),
            panel(target, "image = IDM GT (wear B)"),
        ]
        gap = 8
        tw = sum(p.size[0] for p in panels) + gap * 2
        th = max(p.size[1] for p in panels) + 36
        canvas = Image.new("RGB", (tw, th), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        title = f"{r.get('id', i)}  person!=target training pair"
        draw.text((8, 8), title, fill=(0, 0, 0), font=ImageFont.load_default())
        x = 0
        for p in panels:
            canvas.paste(p, (x, 36))
            x += p.size[0] + gap
        gpath = grid_dir / f"{i:02d}_{r.get('id', i)}.jpg"
        canvas.save(gpath, quality=92)
        written.append(str(gpath))
        print("wrote", gpath)

    summary = out / "README.md"
    summary.write_text(
        "\n".join(
            [
                "# IDM synth training pairs preview",
                "",
                "Each grid is one **supervised edit pair**:",
                "",
                "1. `edit_image[0]` person wearing A",
                "2. `edit_image[1]` garment B",
                "3. `image` IDM synthetic try-on (person wearing B) = training target",
                "",
                f"dataset_base: `{base}`",
                f"n_preview: {len(written)} / metadata_train={len(rows)}",
                "",
                "## Grids",
                "",
                *[f"- `{p}`" for p in written],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("DONE", summary)


if __name__ == "__main__":
    main()
