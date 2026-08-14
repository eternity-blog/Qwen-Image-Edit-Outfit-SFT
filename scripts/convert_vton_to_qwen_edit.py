#!/usr/bin/env python3
"""Convert VITON-HD / DressCode into DiffSynth Qwen-Image-Edit metadata.

Output layout:
  <out_dir>/
    images/                 # optional symlinks or copied refs
    metadata.jsonl          # one sample per line (json)
    metadata.json           # list form for DiffSynth UnifiedDataset
    stats.json

Each sample:
  {
    "image": "<tryon relative path>",          # training target
    "edit_image": ["<person>", "<garment>"],   # multi-image condition
    "prompt": "...",
    "source": "viton_hd|dresscode",
    "category": "upper|lower|dresses|..."
  }

DiffSynth expects paths relative to --dataset_base_path.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROMPT_TEMPLATES = [
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


def rel(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def exists_pair(person: Path, cloth: Path, tryon: Path) -> bool:
    return person.is_file() and cloth.is_file() and tryon.is_file()


def collect_viton_hd(raw_dir: Path, base: Path, split: str, limit: int) -> List[dict]:
    """VITON-HD typical layout after unzip:
    train/image|cloth|image-parse|... OR train/<id>_*.jpg variants.

    TryOnVirtual/VITON-HD-IMAGE often unpacks to:
      train/image/*.jpg
      train/cloth/*.jpg
      train/cloth-mask/*.jpg
    Target try-on for paired training is usually the same person image for
    reconstruction-style pairs is NOT available as separate tryon — official
    VITON-HD uses unpaired cloth with person; the supervised try-on target in
    many pipelines is synthesized.

    For real paired try-on, we use the common convention when files share stem:
      image/<id>.jpg + cloth/<id>.jpg  and treat image as BOTH person and target
      only when a dedicated tryon/ folder exists.

    Better: if `image` and `cloth` share ids, create pairs where:
      edit_image=[image, cloth], image=image
    This trains garment-conditioned reconstruction / identity preservation with
    cloth reference — a practical Stage-1 prior used when GT try-on isn't stored.

    If folder `tryon` or `gt` exists, prefer that as target.
    """
    root = raw_dir / split
    if not root.is_dir():
        # sometimes zip extracts flat
        candidates = [p for p in raw_dir.iterdir() if p.is_dir() and split in p.name.lower()]
        root = candidates[0] if candidates else root
    if not root.is_dir():
        print(f"[warn] missing VITON split dir: {root}")
        return []

    image_dir = None
    cloth_dir = None
    tryon_dir = None
    for name in ("image", "images", "person"):
        if (root / name).is_dir():
            image_dir = root / name
            break
    for name in ("cloth", "clothes", "garment", "clothing"):
        if (root / name).is_dir():
            cloth_dir = root / name
            break
    for name in ("tryon", "gt", "target", "result"):
        if (root / name).is_dir():
            tryon_dir = root / name
            break

    if image_dir is None or cloth_dir is None:
        # fallback: scan one level
        print(f"[warn] VITON {split} unexpected layout under {root}: {[p.name for p in root.iterdir()][:20]}")
        return []

    samples: List[dict] = []
    cloth_map = {p.stem: p for p in cloth_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}
    tryon_map = {}
    if tryon_dir is not None:
        tryon_map = {p.stem: p for p in tryon_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}

    for person in sorted(image_dir.glob("*")):
        if person.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        cloth = cloth_map.get(person.stem)
        if cloth is None:
            continue
        target = tryon_map.get(person.stem, person)
        prompt = PROMPT_TEMPLATES[len(samples) % len(PROMPT_TEMPLATES)]
        samples.append(
            {
                "image": rel(target, base),
                "edit_image": [rel(person, base), rel(cloth, base)],
                "prompt": prompt,
                "source": "viton_hd",
                "category": "upper",
                "split": split,
                "id": person.stem,
            }
        )
        if limit > 0 and len(samples) >= limit:
            break
    return samples


def collect_dresscode(raw_dir: Path, base: Path, limit_per_cat: int) -> List[dict]:
    """DressCode layout:
      DressCode/upper_body|lower_body|dresses/
        images/, clothes/, ...
        and paired lists in *.txt
    """
    root = raw_dir
    if (raw_dir / "DressCode").is_dir():
        root = raw_dir / "DressCode"

    samples: List[dict] = []
    cats = {
        "upper_body": "upper",
        "lower_body": "lower",
        "dresses": "dresses",
    }
    for folder, cat in cats.items():
        cat_dir = root / folder
        if not cat_dir.is_dir():
            continue
        image_dir = None
        cloth_dir = None
        for name in ("images", "image"):
            if (cat_dir / name).is_dir():
                image_dir = cat_dir / name
                break
        for name in ("clothes", "cloth", "garments"):
            if (cat_dir / name).is_dir():
                cloth_dir = cat_dir / name
                break
        if image_dir is None or cloth_dir is None:
            print(f"[warn] DressCode {folder} layout: {[p.name for p in cat_dir.iterdir()][:30]}")
            continue

        # Prefer official pair list if present
        pair_files = list(cat_dir.glob("*pairs*.txt")) + list(cat_dir.glob("*pair*.txt"))
        n = 0
        if pair_files:
            for pf in pair_files:
                for line in pf.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = re.split(r"[\s,]+", line.strip())
                    if len(parts) < 2:
                        continue
                    # common: <person_id> <cloth_id>  or filenames
                    a, b = parts[0], parts[1]
                    person = _resolve_image(image_dir, a)
                    cloth = _resolve_image(cloth_dir, b)
                    if person is None or cloth is None:
                        continue
                    prompt = PROMPT_TEMPLATES[n % len(PROMPT_TEMPLATES)]
                    samples.append(
                        {
                            "image": rel(person, base),
                            "edit_image": [rel(person, base), rel(cloth, base)],
                            "prompt": prompt,
                            "source": "dresscode",
                            "category": cat,
                            "split": "train",
                            "id": f"{folder}-{person.stem}-{cloth.stem}",
                        }
                    )
                    n += 1
                    if limit_per_cat > 0 and n >= limit_per_cat:
                        break
                if limit_per_cat > 0 and n >= limit_per_cat:
                    break
        else:
            cloth_map = {p.stem: p for p in cloth_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
            for person in sorted(image_dir.glob("*")):
                if person.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                # DressCode person filenames often like 00000_0.jpg, cloth 00000_1.jpg
                stem = person.stem
                cloth = cloth_map.get(stem)
                if cloth is None and "_" in stem:
                    cloth = cloth_map.get(stem.rsplit("_", 1)[0] + "_1") or cloth_map.get(stem.replace("_0", "_1"))
                if cloth is None:
                    continue
                prompt = PROMPT_TEMPLATES[n % len(PROMPT_TEMPLATES)]
                samples.append(
                    {
                        "image": rel(person, base),
                        "edit_image": [rel(person, base), rel(cloth, base)],
                        "prompt": prompt,
                        "source": "dresscode",
                        "category": cat,
                        "split": "train",
                        "id": f"{folder}-{person.stem}",
                    }
                )
                n += 1
                if limit_per_cat > 0 and n >= limit_per_cat:
                    break
        print(f"[dresscode] {folder}: {n} samples")
    return samples


def _resolve_image(folder: Path, token: str) -> Optional[Path]:
    token = token.strip()
    direct = folder / token
    if direct.is_file():
        return direct
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = folder / f"{token}{ext}"
        if p.is_file():
            return p
        # token may already include ext
    # stem match
    stem = Path(token).stem
    for p in folder.glob(stem + ".*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/data/agent/hf_models/datasets/qwen_vton/raw")
    ap.add_argument("--out-dir", default="/data/agent/hf_models/datasets/qwen_vton/converted")
    ap.add_argument("--viton-limit", type=int, default=0, help="0=all")
    ap.add_argument("--dresscode-limit-per-cat", type=int, default=0, help="0=all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    args = ap.parse_args()

    raw = Path(args.raw_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # dataset_base_path for DiffSynth should point at raw root so relative paths work.
    # We write metadata with paths relative to raw root.
    base = raw

    samples: List[dict] = []
    viton_root = raw / "viton_hd"
    if viton_root.is_dir():
        for split in ("train", "test"):
            part = collect_viton_hd(viton_root, base, split, limit=args.viton_limit)
            print(f"[viton_hd] {split}: {len(part)}")
            samples.extend(part)

    dress_root = raw / "dresscode"
    if dress_root.is_dir():
        part = collect_dresscode(dress_root, base, limit_per_cat=args.dresscode_limit_per_cat)
        print(f"[dresscode] total: {len(part)}")
        samples.extend(part)

    if not samples:
        raise SystemExit(f"no samples found under {raw}")

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio)) if args.val_ratio > 0 else 0
    val = samples[:n_val]
    train = samples[n_val:]

    def dump(name: str, rows: List[dict]) -> None:
        jsonl = out / f"{name}.jsonl"
        js = out / f"{name}.json"
        with jsonl.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        js.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {jsonl} ({len(rows)})")

    dump("metadata_train", train)
    dump("metadata_val", val)
    # DiffSynth default name
    (out / "metadata.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats = {
        "n_total": len(samples),
        "n_train": len(train),
        "n_val": len(val),
        "by_source": {},
        "by_category": {},
        "dataset_base_path": str(base),
        "metadata_train": str(out / "metadata_train.json"),
        "metadata_val": str(out / "metadata_val.json"),
    }
    for r in samples:
        stats["by_source"][r["source"]] = stats["by_source"].get(r["source"], 0) + 1
        stats["by_category"][r["category"]] = stats["by_category"].get(r["category"], 0) + 1
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
