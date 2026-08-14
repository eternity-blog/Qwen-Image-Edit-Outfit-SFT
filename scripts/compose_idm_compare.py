#!/usr/bin/env python3
"""Compose case02 grids: source | GPT | base | VTON-LoRA | IDM-LoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def label_image(img: Image.Image, label: str, width: int = 280) -> Image.Image:
    im = img.convert("RGB").copy()
    w, h = im.size
    nh = max(1, int(round(h * (width / float(w)))))
    im = im.resize((width, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, nh + 28), (245, 245, 245))
    canvas.paste(im, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, 28], fill=(30, 30, 30))
    font = ImageFont.load_default()
    draw.text((6, 8), label, fill=(255, 255, 255), font=font)
    return canvas


def make_grid(panels: list[tuple[str, Image.Image]], title: str) -> Image.Image:
    labeled = [label_image(im, name) for name, im in panels]
    gap = 8
    total_w = sum(p.size[0] for p in labeled) + gap * (len(labeled) - 1)
    total_h = max(p.size[1] for p in labeled) + 36
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    x = 0
    for p in labeled:
        canvas.paste(p, (x, 36))
        x += p.size[0] + gap
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--idm-dir", required=True)
    ap.add_argument("--vton-dir", default="")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    idm_dir = Path(args.idm_dir)
    vton_dir = Path(args.vton_dir) if args.vton_dir else None
    out_dir = Path(args.out_dir)
    grid_dir = out_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    base_meta = json.loads((base_dir / "run_meta.json").read_text(encoding="utf-8"))
    samples = base_meta.get("samples") or []
    rows = []
    for s in samples:
        tag = f"{int(s['shot_index']):02d}_{s['role']}"
        base_img = base_dir / "qwen" / f"{tag}.png"
        idm_img = idm_dir / "qwen" / f"{tag}.png"
        if not base_img.is_file() or not idm_img.is_file():
            print(f"skip missing {tag}")
            continue
        source = Image.open(s["source_path"]).convert("RGB")
        gpt_path = Path(s.get("gpt_path") or "")
        gpt = Image.open(gpt_path).convert("RGB") if gpt_path.is_file() else source
        panels: list[tuple[str, Image.Image]] = [
            ("source", source),
            ("GPT Image 2", gpt),
            ("Qwen base", Image.open(base_img).convert("RGB")),
        ]
        if vton_dir is not None:
            vton_img = vton_dir / "qwen" / f"{tag}.png"
            if vton_img.is_file():
                panels.append(("VTON-LoRA", Image.open(vton_img).convert("RGB")))
        panels.append(("IDM-LoRA", Image.open(idm_img).convert("RGB")))
        title = f"case02 {tag}  prompt={base_meta.get('prompt_mode', '?')}"
        grid = make_grid(panels, title)
        out_path = grid_dir / f"{tag}_compare.jpg"
        grid.save(out_path, quality=92)
        rows.append({"tag": tag, "grid": str(out_path), "n_panels": len(panels)})
        print("wrote", out_path)

    summary = [
        "# Case02: base vs IDM-LoRA (vs VTON-LoRA)",
        "",
        f"- prompt_mode: `{base_meta.get('prompt_mode')}`",
        f"- steps: `{base_meta.get('steps')}`",
        f"- n_grids: {len(rows)}",
        "",
        "Panels: source | GPT Image 2 | Qwen base | [VTON-LoRA] | IDM-LoRA",
        "",
        "Note: short prompt matches LoRA training style; EditPlus does not truncate live v2 prompts (domain gap > length).",
        "",
    ]
    for r in rows:
        summary.append(f"- `{r['tag']}` → `{r['grid']}`")
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (out_dir / "compose_meta.json").write_text(
        json.dumps({"rows": rows, "base_meta": base_meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("DONE", out_dir)


if __name__ == "__main__":
    main()
