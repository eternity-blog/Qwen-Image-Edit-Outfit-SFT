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
                                             row 4: per-pixel difference CDF

Row 2 column 1 is the key panel: ``|person - teacher|`` shows which pixels a
*correct* garment swap is supposed to touch — the visual form of the
``MAD(person, teacher)`` baseline. A model that redraws the whole frame lights up
everywhere; a model doing local editing lights up the same region as the baseline.

Usage:
    python scripts/visualize_metrics.py --eval-dir $OUTPUT_ROOT/viton_holdout_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

# Fixed clip so heatmaps stay comparable across panels, samples and runs.
DIFF_CLIP = 128.0


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
def render(rid, person, teacher, outs, dest) -> None:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="output dir of eval_viton_holdout.py")
    ap.add_argument("--out-dir", default="", help="defaults to <eval-dir>/viz")
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
