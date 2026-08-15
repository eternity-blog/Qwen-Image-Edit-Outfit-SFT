#!/usr/bin/env python3
"""visualize_metrics.py — render the MAD / HS-histogram eval metrics as pictures.

The eval scripts report two scalars per image pair:

* **MAD** — mean absolute difference of the two greyscale images (0-255 scale)
* **hist corr** — correlation of two 32x32 Hue-Saturation histograms (-1..1)

Scalars hide *where* and *how* two images differ: "redrew the whole frame" and
"edited the garment well" can land on a similar MAD. This script draws them.

    column = object                 row = view
    -------------------------------------------------------------
    person | teacher | model_a | model_b     row 1: the images
           |         |         |             row 2: |x - teacher| heatmap
           |         |         |             row 3: Hue-Saturation histogram
                                             row 4: difference CDF (mpl only)

Row 2 column 1 is the key panel: ``|person - teacher|`` shows which pixels a
*correct* garment swap is supposed to touch — the visual form of the
``MAD(person, teacher)`` baseline. A model that redraws the whole frame lights up
everywhere; a model doing local editing lights up the same region as the baseline.

Backends:

* ``--backend mpl`` (default) — matplotlib. Adds a numeric colorbar, real
  Hue/Saturation axes and a per-pixel difference CDF. Use for reports.
* ``--backend cv2`` — opencv + pillow only, no extra dependency. Larger, denser
  image panels; good for quick batch inspection.

Usage:
    python scripts/visualize_metrics.py --eval-dir $OUTPUT_ROOT/viton_holdout_eval
    python scripts/visualize_metrics.py --eval-dir ... --backend cv2 --limit 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Fixed clip so heatmaps stay comparable across panels, samples and runs.
DIFF_CLIP = 128.0
PANEL_W = 300
LABEL_H = 26


# --------------------------------------------------------------------------- #
# metrics (identical maths to zero_shot_compare.mad_and_hist)
# --------------------------------------------------------------------------- #
def to_rgb(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def grey(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def diff_map(a: Image.Image, b: Image.Image) -> tuple[np.ndarray, float]:
    """Per-pixel |grey(a) - grey(b)| and its mean (= MAD, on the unclipped values)."""
    b = b.resize(a.size, Image.Resampling.LANCZOS)
    d = np.abs(grey(a) - grey(b))
    return d, float(d.mean())


def hs_hist(img: Image.Image) -> np.ndarray:
    hsv = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


def hist_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


# --------------------------------------------------------------------------- #
# cv2 backend
# --------------------------------------------------------------------------- #
def _heat_image(d: np.ndarray) -> Image.Image:
    scaled = np.clip(d / DIFF_CLIP, 0.0, 1.0) * 255.0
    heat = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_TURBO)
    return Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))


def _hist_image(h: np.ndarray, size: int = 256) -> Image.Image:
    v = np.sqrt(np.clip(h, 0, None))
    v = v / (v.max() + 1e-8)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    out = Image.fromarray(np.flipud(img).copy())
    d = ImageDraw.Draw(out)
    f = ImageFont.load_default()
    d.text((4, size - 14), "H 0", fill=(255, 255, 255), font=f)
    d.text((size - 34, size - 14), "H 180", fill=(255, 255, 255), font=f)
    d.text((4, 4), "S 255", fill=(255, 255, 255), font=f)
    return out


def _panel(img: Image.Image, label: str, width: int = PANEL_W) -> Image.Image:
    im = img.convert("RGB")
    w, h = im.size
    nh = max(1, int(round(h * (width / float(w)))))
    im = im.resize((width, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, nh + LABEL_H), (245, 245, 245))
    canvas.paste(im, (0, LABEL_H))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, width, LABEL_H], fill=(30, 30, 30))
    d.text((6, 7), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return canvas


def render_cv2(rid, person, teacher, outs, dest) -> None:
    h_teacher = hs_hist(teacher)
    base_d, base_mad = diff_map(person, teacher)

    row_img = [_panel(person, "person (source)"), _panel(teacher, "IDM teacher (GT)")]
    row_heat = [
        _panel(_heat_image(base_d), f"|person-teacher| MAD={base_mad:.1f}  <-baseline"),
        _panel(Image.new("RGB", person.size, (255, 255, 255)), "(GT itself)"),
    ]
    row_hist = [
        _panel(_hist_image(hs_hist(person)), "H-S hist: person"),
        _panel(_hist_image(h_teacher), "H-S hist: teacher"),
    ]
    for name, im in outs.items():
        d, mad = diff_map(im, teacher)
        hh = hs_hist(im)
        row_img.append(_panel(im, name))
        row_heat.append(_panel(_heat_image(d), f"|{name}-teacher| MAD={mad:.1f}"))
        row_hist.append(
            _panel(_hist_image(hh), f"H-S hist: {name} corr={hist_corr(h_teacher, hh):.3f}")
        )

    rows = [row_img, row_heat, row_hist]
    gap, top = 8, 34
    n_cols = max(len(r) for r in rows)
    row_h = [max(p.size[1] for p in r) for r in rows]
    canvas = Image.new(
        "RGB",
        (n_cols * PANEL_W + (n_cols - 1) * gap, top + sum(row_h) + gap * (len(rows) - 1)),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 10),
        f"{rid}   heatmap: |grey(a)-grey(b)| TURBO clipped at {int(DIFF_CLIP)}"
        f"   (blue=identical, red=very different)   hist: 32x32 Hue-Saturation, sqrt scale",
        fill=(0, 0, 0),
        font=ImageFont.load_default(),
    )
    y = top
    for r, hgt in zip(rows, row_h):
        x = 0
        for p in r:
            canvas.paste(p, (x, y))
            x += PANEL_W + gap
        y += hgt + gap
    canvas.save(dest, quality=92)


# --------------------------------------------------------------------------- #
# matplotlib backend
# --------------------------------------------------------------------------- #
def render_mpl(rid, person, teacher, outs, dest) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    cols = ["person (source)", "IDM teacher (GT)"] + list(outs.keys())
    imgs = [person, teacher] + list(outs.values())
    n = len(cols)

    fig = plt.figure(figsize=(3.6 * n + 1.4, 13.2))
    gs = GridSpec(
        4, n + 1, figure=fig,
        width_ratios=[1] * n + [0.05],
        height_ratios=[1.35, 1.35, 0.8, 0.75],
        hspace=0.42, wspace=0.22,
    )

    for i, (name, im) in enumerate(zip(cols, imgs)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.axis("off")

    h_teacher = hs_hist(teacher)
    base_d, base_mad = diff_map(person, teacher)
    heat_specs = [("|person - teacher|  (baseline)", base_d, base_mad), (None, None, None)]
    for name, im in outs.items():
        d, mad = diff_map(im, teacher)
        heat_specs.append((f"|{name} - teacher|", d, mad))

    handle = None
    for i, (title, d, mad) in enumerate(heat_specs):
        ax = fig.add_subplot(gs[1, i])
        if d is None:
            ax.text(0.5, 0.5, "GT itself\n(zero by definition)",
                    ha="center", va="center", fontsize=11, color="#888")
            ax.axis("off")
            continue
        handle = ax.imshow(d, cmap="turbo", vmin=0, vmax=DIFF_CLIP)
        ax.set_title(f"{title}\nMAD = {mad:.1f}", fontsize=10)
        ax.axis("off")
    if handle is not None:
        cb = fig.colorbar(handle, cax=fig.add_subplot(gs[1, n]))
        cb.set_label("|grey difference| (0-255, clipped)", fontsize=9)

    hist_specs = [("person", hs_hist(person)), ("teacher", h_teacher)]
    hist_specs += [(name, hs_hist(im)) for name, im in outs.items()]
    for i, (name, h) in enumerate(hist_specs):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(
            np.sqrt(np.clip(h, 0, None)).T,
            origin="lower", aspect="auto", cmap="viridis", extent=(0, 180, 0, 255),
        )
        c = "" if name in ("person", "teacher") else f"  corr={hist_corr(h_teacher, h):.3f}"
        ax.set_title(f"H-S hist: {name}{c}", fontsize=10)
        ax.set_xlabel("Hue (0-180)", fontsize=9)
        # only the leftmost panel keeps y ticks, otherwise neighbouring labels collide
        if i == 0:
            ax.set_ylabel("Saturation (0-255)", fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=8)

    ax = fig.add_subplot(gs[3, :n])
    for label, d in [("person (baseline)", base_d)] + [
        (name, diff_map(im, teacher)[0]) for name, im in outs.items()
    ]:
        v = np.sort(d.ravel())
        ax.plot(v, np.linspace(0, 100, v.size), label=label, linewidth=1.8)
    ax.set_xlim(0, DIFF_CLIP)
    ax.set_ylim(0, 100)
    ax.set_xlabel("per-pixel |grey difference| vs teacher", fontsize=9)
    ax.set_ylabel("% of pixels below", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(
        "Difference CDF — a curve hugging the left edge means most pixels already match the teacher",
        fontsize=10, pad=12,
    )

    fig.suptitle(
        f"{rid}   heatmap clipped at {int(DIFF_CLIP)} (blue = identical, red = very different)",
        fontsize=12, y=0.995,
    )
    fig.savefig(dest, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="output dir of eval_viton_holdout.py")
    ap.add_argument("--out-dir", default="", help="defaults to <eval-dir>/viz")
    ap.add_argument("--backend", default="mpl", choices=["mpl", "cv2"])
    ap.add_argument("--models", default="", help="comma list; default = all in metrics.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    rows = meta["rows"][: args.limit] if args.limit else meta["rows"]
    names = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else list(meta["results"].keys())
    )
    render = render_mpl if args.backend == "mpl" else render_cv2

    for r in rows:
        rid = r["id"]
        outs = {}
        for nm in names:
            p = eval_dir / nm / f"{rid}.jpg"
            if p.is_file():
                outs[nm] = to_rgb(p)
        if not outs:
            print(f"skip {rid}: no model images")
            continue
        dest = out_dir / f"{rid}_metrics.jpg"
        render(rid, to_rgb(r["person"]), to_rgb(r["teacher"]), outs, dest)
        print("wrote", dest)

    print("DONE ->", out_dir)


if __name__ == "__main__":
    main()
