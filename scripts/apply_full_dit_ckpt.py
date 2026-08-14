#!/usr/bin/env python3
"""Apply a DiffSynth full-DiT checkpoint into a loadable Qwen-Image-Edit-2511 model dir.

DiffSynth full SFT (``--trainable_models dit --remove_prefix_in_ckpt pipe.dit.``)
writes ``epoch-N.safetensors`` containing DiT weights only. This script copies the
base model tree and overwrites ``transformer`` weights so the result can be used
like ``MODEL_DIR`` in eval scripts.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True, help="Base or previously fused model dir")
    ap.add_argument("--ckpt", required=True, help="DiffSynth epoch-*.safetensors (DiT only)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    base = Path(args.base_model)
    ckpt = Path(args.ckpt)
    out = Path(args.out_dir)
    if not base.is_dir():
        raise SystemExit(f"missing base model: {base}")
    if not ckpt.is_file():
        raise SystemExit(f"missing ckpt: {ckpt}")

    if out.exists():
        shutil.rmtree(out)
    print(f"copy {base} -> {out}")
    shutil.copytree(
        base,
        out,
        ignore=shutil.ignore_patterns("*.tmp", "__pycache__"),
    )

    from diffusers import QwenImageEditPlusPipeline
    from safetensors.torch import load_file

    dtype = torch.bfloat16
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        str(out),
        torch_dtype=dtype,
        local_files_only=True,
    )
    sd = load_file(str(ckpt), device=args.device)
    # Keys may be bare DiT names or transformer.* / pipe.dit.*
    remapped = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("pipe.dit."):
            nk = nk[len("pipe.dit.") :]
        if nk.startswith("transformer."):
            nk = nk[len("transformer.") :]
        remapped[nk] = v
    print(f"loading {len(remapped)} tensors into transformer")
    missing, unexpected = pipe.transformer.load_state_dict(remapped, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 20:
        print("sample missing:", missing[:10])
    if len(unexpected) > 20:
        print("sample unexpected:", unexpected[:10])

    print(f"saving pipeline -> {out}")
    pipe.save_pretrained(str(out), safe_serialization=True)
    print("DONE")


if __name__ == "__main__":
    main()
