#!/usr/bin/env python3
"""upload_all_synth_to_hf.py — publish IDM synth batches to one Hugging Face dataset.

All batches live in a single repo and are told apart **by directory**:

    synth/idm_unpaired/            batch1, VITON-HD test split  (~2 032)
    synth/idm_unpaired_train/      batch1, VITON-HD train split (~11 647)
    synth/idm_unpaired_train_b2/   batch2, new pairing          (~11 647)
    converted_idm_synth_train_v2/  DiffSynth metadata (full v2 prompts)

Each synth dir carries `images/part-XXXX/` (HF caps a directory at 10 000 files),
`manifest.jsonl` with paths rewritten to match the sharding, and — for batches
produced by `make_pair_batch.py` — the `pairs_*.txt` and `batch_meta_*.json`
provenance files, so anyone can verify a batch does not overlap the others.

Upload one new batch without touching what is already published:

    python scripts/upload_all_synth_to_hf.py \\
        --synth idm_unpaired_train_b2=$QWEN_VTON_DATA/synth/idm_unpaired_train_b2

With no --synth/--converted the original three directories are uploaded.

Set HF_TOKEN (write scope) first.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

SHARD_SIZE = 5000
PROVENANCE_GLOBS = ("pairs_*.txt", "batch_meta_*.json")
CONVERTED_FILES = (
    "metadata_train.jsonl",
    "metadata_val.jsonl",
    "metadata_train.json",
    "metadata_val.json",
    "metadata.json",
    "stats.json",
    "prompt_example_full.txt",
    "prompt_audit_sample.json",
)

CARD_HEADER = """---
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

Synthetic outfit-swap pairs for **Qwen-Image-Edit-2511** SFT (keyframe garment edit),
generated with IDM-VTON as the teacher over VITON-HD.

## Batches

Batches are separate directories in this one repo. Every batch uses a distinct
(person, garment) pairing: no person is paired with the garment they already wear,
and no pair is repeated across batches. `batch_meta_*.json` records the seed and the
dedup counts, `pairs_*.txt` the exact pairing, so the guarantee is checkable.
"""

CARD_FOOTER = """
## Layout

```text
synth/<batch>/
  images/part-XXXX/…jpg   # sharded: HF allows at most 10000 files per directory
  manifest.jsonl          # person / cloth / out_rel, paths match the sharding
  pairs_*.txt             # the pairing this batch consumed (batch2 onward)
  batch_meta_*.json       # seed + dedup proof (batch2 onward)
converted_idm_synth_train_v2/
  metadata_train.json     # DiffSynth training metadata, full-text v2 prompts
```

`edit_image` paths in the metadata assume a local `viton_hd/` tree — download
VITON-HD yourself (CC BY-NC). `scripts/prepare_data_from_hf.sh` in the code repo
does the download, flattens the shards and rebuilds the metadata.

## License

**Non-commercial research only.** Derived from:

- [VITON-HD](https://github.com/shadow2496/VITON-HD) (CC BY-NC 4.0)
- [IDM-VTON](https://github.com/yisol/IDM-VTON) (CC BY-NC-SA 4.0)

Attribute the upstream authors. Do not use commercially without rights.

## Code

Training, evaluation and data tooling:
<https://github.com/eternity-blog/Qwen-Image-Edit-Outfit-SFT>
"""


def _rmtree(p: Path) -> None:
    if p.exists():
        subprocess.check_call(["rm", "-rf", str(p)])


def stage_synth(src: Path, dst: Path) -> int:
    """Shard images into part-XXXX, rewrite the manifest, copy provenance files."""
    man = src / "manifest.jsonl"
    img = src / "images"
    if not man.is_file():
        raise FileNotFoundError(f"{man} (run merge_idm_shard_manifests.sh first?)")
    if not img.is_dir():
        raise FileNotFoundError(img)

    dst.mkdir(parents=True, exist_ok=True)
    _rmtree(dst / "images")
    files = sorted(p for p in img.iterdir() if p.is_file())
    mapping: dict[str, str] = {}
    for i, s in enumerate(files):
        part = i // SHARD_SIZE
        pdir = dst / "images" / f"part-{part:04d}"
        pdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, pdir / s.name)
        mapping[s.name] = f"images/part-{part:04d}/{s.name}"
        if (i + 1) % 2000 == 0:
            print(f"    copied {i + 1}/{len(files)}")

    n = 0
    with man.open(encoding="utf-8") as fin, (dst / "manifest.jsonl").open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("out_rel", "image", "output", "result", "path", "synth_image"):
                v = row.get(key)
                if isinstance(v, str) and Path(v).name in mapping:
                    row[key] = mapping[Path(v).name]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    for pattern in PROVENANCE_GLOBS:
        for f in sorted(src.glob(pattern)):
            shutil.copy2(f, dst / f.name)
            print(f"    provenance: {f.name}")

    shards = (len(files) + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"    {len(files)} images -> {shards} shard(s), manifest rows={n}")
    return len(files)


def stage_converted(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in CONVERTED_FILES:
        f = src / name
        if f.is_file():
            shutil.copy2(f, dst / name)
            print(f"    {name}")


def build_card(api, repo_id: str, token: str) -> str:
    """Regenerate the card from what the repo actually contains."""
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    lines = [CARD_HEADER, "", "| batch | 目录 | 图片数 |", "|---|---|---|"]
    try:
        synth_dirs = [
            x.path
            for x in api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo="synth")
            if isinstance(x, RepoFolder)
        ]
    except Exception as e:  # noqa: BLE001 - card is best-effort, upload already succeeded
        print(f"[warn] could not list repo tree for the card: {type(e).__name__}")
        return CARD_HEADER + CARD_FOOTER
    for d in sorted(synth_dirs):
        name = Path(d).name
        n = sum(
            1
            for x in api.list_repo_tree(
                repo_id, repo_type="dataset", path_in_repo=f"{d}/images", recursive=True
            )
            if isinstance(x, RepoFile)
        )
        label = {
            "idm_unpaired": "batch1 (VITON-HD test split)",
            "idm_unpaired_train": "batch1 (VITON-HD train split)",
        }.get(name, name)
        lines.append(f"| {label} | `{d}/` | {n} |")
    return "\n".join(lines) + "\n" + CARD_FOOTER


def parse_kv(items: list[str], flag: str) -> list[tuple[str, Path]]:
    out = []
    for it in items:
        if "=" not in it:
            raise SystemExit(f"{flag} expects NAME=PATH, got {it}")
        name, path = it.split("=", 1)
        p = Path(path)
        if not p.is_dir():
            raise SystemExit(f"{flag} {name}: not a directory: {p}")
        out.append((name, p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling")
    ap.add_argument("--data-root", default="/data/agent/hf_models/datasets/qwen_vton")
    ap.add_argument("--synth", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--converted", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--staging", default="/tmp/hf_upload_outfit_synth")
    ap.add_argument("--skip-stage", action="store_true")
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--skip-card", action="store_true", help="leave README.md untouched")
    ap.add_argument("--private", action="store_true", default=False)
    args = ap.parse_args()

    root = Path(args.data_root)
    synth = parse_kv(args.synth, "--synth")
    converted = parse_kv(args.converted, "--converted")
    if not synth and not converted:
        synth = [
            ("idm_unpaired", root / "synth" / "idm_unpaired"),
            ("idm_unpaired_train", root / "synth" / "idm_unpaired_train"),
        ]
        converted = [
            ("converted_idm_synth_train_v2", root / "converted_idm_synth_train_v2"),
        ]
        print("no --synth/--converted given; uploading the default three directories")

    staging = Path(args.staging)
    if not args.skip_stage:
        _rmtree(staging)
        staging.mkdir(parents=True)
        (staging / "LICENSE_NOTE.txt").write_text(
            "Derived from VITON-HD (CC BY-NC 4.0) and IDM-VTON (CC BY-NC-SA 4.0). "
            "Research / non-commercial only.\n",
            encoding="utf-8",
        )
        for name, src in synth:
            print(f"staging synth/{name} from {src}")
            stage_synth(src, staging / "synth" / name)
        for name, src in converted:
            print(f"staging {name} from {src}")
            stage_converted(src, staging / name)
        print("staging ready:", staging)
    else:
        print("reusing staging:", staging)

    if args.stage_only:
        print("stage-only; skipping upload")
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN (write scope): https://huggingface.co/settings/tokens")

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True, token=token)
    print(f"uploading -> https://huggingface.co/datasets/{args.repo_id}")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Add " + ", ".join(n for n, _ in synth + converted),
    )

    if not args.skip_card:
        card = build_card(api, args.repo_id, token)
        card_path = staging / "README.md"
        card_path.write_text(card, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Refresh dataset card batch table",
        )
        print("card updated")

    print("DONE")


if __name__ == "__main__":
    main()
