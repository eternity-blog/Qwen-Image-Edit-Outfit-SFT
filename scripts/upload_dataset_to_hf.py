#!/usr/bin/env python3
"""Upload IDM synth + v2 metadata to a personal Hugging Face Dataset repo.

Example:
  huggingface-cli login
  python scripts/upload_dataset_to_hf.py \\
    --repo-id yourname/qwen-outfit-idm-synth-v2 \\
    --synth-dir /path/to/synth/idm_unpaired_train \\
    --converted-dir /path/to/converted_idm_synth_train_v2 \\
    --private
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. username/qwen-outfit-idm-synth-v2")
    ap.add_argument("--synth-dir", required=True, help="dir with images/ + manifest.jsonl")
    ap.add_argument("--converted-dir", required=True, help="converted_idm_synth_train_v2")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--staging", default="", help="local staging dir before upload")
    ap.add_argument(
        "--skip-images",
        action="store_true",
        help="upload metadata only (smaller); users need images elsewhere",
    )
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub") from e

    synth = Path(args.synth_dir)
    converted = Path(args.converted_dir)
    if not (synth / "manifest.jsonl").is_file():
        raise SystemExit(f"missing {synth}/manifest.jsonl")
    if not (converted / "metadata_train.jsonl").is_file():
        raise SystemExit(f"missing {converted}/metadata_train.jsonl")

    staging = Path(args.staging) if args.staging else Path("/tmp") / args.repo_id.replace("/", "__")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # dataset card
    card_src = Path(__file__).resolve().parents[1] / "dataset_card" / "README.md"
    if card_src.is_file():
        shutil.copy(card_src, staging / "README.md")
    else:
        (staging / "README.md").write_text(
            f"# {args.repo_id}\n\nSynthetic outfit-edit pairs for learning SFT. Non-commercial. See NOTICE.\n",
            encoding="utf-8",
        )

    conv_out = staging / "converted_idm_synth_train_v2"
    conv_out.mkdir()
    for name in (
        "metadata_train.jsonl",
        "metadata_val.jsonl",
        "stats.json",
        "prompt_example_full.txt",
        "prompt_audit_sample.json",
    ):
        src = converted / name
        if src.is_file():
            shutil.copy(src, conv_out / name)

    man_out = staging / "idm_unpaired_train"
    man_out.mkdir()
    shutil.copy(synth / "manifest.jsonl", man_out / "manifest.jsonl")
    if not args.skip_images:
        img_src = synth / "images"
        if not img_src.is_dir():
            raise SystemExit(f"missing {img_src}")
        print(f"copying images from {img_src} (may take a while)...")
        shutil.copytree(img_src, man_out / "images")

    (staging / "LICENSE_NOTE.txt").write_text(
        "Derived from VITON-HD (CC BY-NC) + IDM-VTON (CC BY-NC-SA). "
        "Research / non-commercial use only. Attribute upstream authors.\n",
        encoding="utf-8",
    )

    api = HfApi()
    create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    print(f"uploading {staging} -> {args.repo_id}")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"DONE https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
