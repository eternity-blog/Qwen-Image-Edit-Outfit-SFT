#!/usr/bin/env python3
"""Upload all IDM synth data + v2 metadata to a Hugging Face Dataset repo.

HF git limit: <=10000 files per directory. Train images (~11.6k) are sharded into
`images/part-0000/`, `images/part-0001/`, ... (5000 files each). Manifest paths
are rewritten to match.

Example:
  export HF_TOKEN=hf_xxx
  python scripts/upload_all_synth_to_hf.py \\
    --repo-id lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


SHARD_SIZE = 5000

CARD = """---
license: other
license_name: cc-by-nc-sa-derived
pretty_name: Outfit Qwen-Image-Edit-2511 IDM Synth (Kling)
task_categories:
  - image-to-image
tags:
  - virtual-try-on
  - image-editing
  - sft
  - qwen-image-edit
  - idm-vton
---

# Outfit_Qwen-Image-Edit-2511_in_Kling

Synthetic outfit-swap pairs for **Qwen-Image-Edit-2511** SFT (keyframe garment edit).

## Contents

| Path | Description |
|---|---|
| `synth/idm_unpaired/` | IDM-VTON unpaired synth on VITON-HD **test** (~2032) |
| `synth/idm_unpaired_train/` | IDM-VTON unpaired synth on VITON-HD **train** (~11647) |
| `converted_idm_synth_train_v2/` | DiffSynth metadata with **full** Outfit v2 prompts |

Each synth dir has sharded `images/part-XXXX/` + `manifest.jsonl`  
(`part-*` keeps HF's ≤10k files/dir limit).  
Metadata `edit_image` paths assume a local `viton_hd/` tree (download VITON-HD yourself; CC BY-NC).

## License

**Non-commercial research only.** Derived from:

- [VITON-HD](https://github.com/shadow2496/VITON-HD) (CC BY-NC 4.0)
- [IDM-VTON](https://github.com/yisol/IDM-VTON) (CC BY-NC-SA 4.0)

Attribute upstream authors. Do not use for commercial products without rights.
"""


def _rmtree(path: Path) -> None:
    if path.exists():
        subprocess.check_call(["rm", "-rf", str(path)])


def shard_copy_images(src_images: Path, dst_images: Path) -> dict[str, str]:
    """Copy images into part-XXXX shards. Returns basename -> relative path map."""
    files = sorted(p for p in src_images.iterdir() if p.is_file())
    mapping: dict[str, str] = {}
    dst_images.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files):
        part = i // SHARD_SIZE
        part_dir = dst_images / f"part-{part:04d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        rel = f"images/part-{part:04d}/{src.name}"
        shutil.copy2(src, part_dir / src.name)
        mapping[src.name] = rel
        if (i + 1) % 2000 == 0:
            print(f"  copied {i + 1}/{len(files)}")
    print(f"  done {len(files)} files -> {(len(files) + SHARD_SIZE - 1) // SHARD_SIZE} shards")
    return mapping


def rewrite_manifest(src_man: Path, dst_man: Path, mapping: dict[str, str]) -> None:
    n = 0
    with src_man.open(encoding="utf-8") as fin, dst_man.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # primary path field used by our synth manifests
            if "out_rel" in row and isinstance(row["out_rel"], str):
                name = Path(row["out_rel"]).name
                if name in mapping:
                    row["out_rel"] = mapping[name]
            for key in ("image", "output", "result", "path", "synth_image", "out_name"):
                if key == "out_name":
                    continue  # basename only
                if key in row and isinstance(row[key], str):
                    name = Path(row[key]).name
                    if name in mapping and (
                        row[key] == name
                        or row[key].endswith("/" + name)
                        or "/images/" in row[key]
                        or row[key].startswith("images/")
                    ):
                        row[key] = mapping[name]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"  rewritten manifest rows={n}")


def copy_synth(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    man = src / "manifest.jsonl"
    if not man.is_file():
        raise FileNotFoundError(man)
    img = src / "images"
    if not img.is_dir():
        raise FileNotFoundError(img)
    print(f"shard-copy {img} -> {dst / 'images'}")
    _rmtree(dst / "images")
    mapping = shard_copy_images(img, dst / "images")
    rewrite_manifest(man, dst / "manifest.jsonl", mapping)


def copy_converted(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "metadata_train.jsonl",
        "metadata_val.jsonl",
        "metadata_train.json",
        "metadata_val.json",
        "metadata.json",
        "stats.json",
        "prompt_example_full.txt",
        "prompt_audit_sample.json",
    ):
        p = src / name
        if p.is_file():
            shutil.copy2(p, dst / name)
            print("copied", name)


def reshard_existing_flat(synth_dir: Path) -> None:
    """If images/ is flat (no part-*), move into shards and rewrite manifest."""
    images = synth_dir / "images"
    if not images.is_dir():
        return
    parts = list(images.glob("part-*"))
    flat = sorted(p for p in images.iterdir() if p.is_file())
    if parts and not flat:
        print(f"already sharded: {synth_dir}")
        return
    if not flat:
        print(f"no flat images in {synth_dir}")
        return
    print(f"reshard in-place (move) {synth_dir} ({len(flat)} files)")
    mapping: dict[str, str] = {}
    for i, src in enumerate(flat):
        part = i // SHARD_SIZE
        part_dir = images / f"part-{part:04d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        dest = part_dir / src.name
        src.rename(dest)
        mapping[src.name] = f"images/part-{part:04d}/{src.name}"
        if (i + 1) % 2000 == 0:
            print(f"  moved {i + 1}/{len(flat)}")
    man = synth_dir / "manifest.jsonl"
    if man.is_file():
        bak = synth_dir / "manifest.jsonl.bak"
        shutil.copy2(man, bak)
        rewrite_manifest(bak, man, mapping)
        bak.unlink()
    print(f"  done shards={(len(flat) + SHARD_SIZE - 1) // SHARD_SIZE}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-id",
        default="lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling",
    )
    ap.add_argument(
        "--data-root",
        default="/data/agent/hf_models/datasets/qwen_vton",
    )
    ap.add_argument(
        "--staging",
        default="/tmp/hf_upload_outfit_synth",
    )
    ap.add_argument("--skip-stage", action="store_true", help="reuse existing staging")
    ap.add_argument(
        "--reshard-staging",
        action="store_true",
        help="reshard flat images/ under existing staging then upload",
    )
    ap.add_argument("--stage-only", action="store_true", help="only build staging; no upload")
    ap.add_argument("--private", action="store_true", default=False)
    args = ap.parse_args()

    root = Path(args.data_root)
    staging = Path(args.staging)
    train = root / "synth" / "idm_unpaired_train"
    test = root / "synth" / "idm_unpaired"
    converted = root / "converted_idm_synth_train_v2"

    if args.reshard_staging:
        for name in ("idm_unpaired", "idm_unpaired_train"):
            reshard_existing_flat(staging / "synth" / name)
        (staging / "README.md").write_text(CARD, encoding="utf-8")
        print("reshard done:", staging)
    elif not args.skip_stage:
        if staging.exists():
            print("removing old staging", staging)
            _rmtree(staging)
        staging.mkdir(parents=True)
        (staging / "README.md").write_text(CARD, encoding="utf-8")
        (staging / "LICENSE_NOTE.txt").write_text(
            "Derived from VITON-HD (CC BY-NC 4.0) and IDM-VTON (CC BY-NC-SA 4.0). "
            "Research / non-commercial only.\n",
            encoding="utf-8",
        )
        copy_synth(test, staging / "synth" / "idm_unpaired")
        copy_synth(train, staging / "synth" / "idm_unpaired_train")
        copy_converted(converted, staging / "converted_idm_synth_train_v2")
        print("staging ready:", staging)
    else:
        print("reuse staging:", staging)

    if args.stage_only:
        print("stage-only done; skip upload")
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "Set HF_TOKEN (write access) then re-run with --skip-stage. "
            "Create at https://huggingface.co/settings/tokens"
        )

    from huggingface_hub import HfApi, create_repo, login

    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)
    create_repo(
        args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=token,
    )
    print(f"uploading -> https://huggingface.co/datasets/{args.repo_id}")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
    )
    print("DONE")


if __name__ == "__main__":
    main()
