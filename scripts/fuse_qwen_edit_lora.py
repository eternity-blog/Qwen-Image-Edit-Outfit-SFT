#!/usr/bin/env python3
"""Fuse (拼接) DiffSynth/PEFT LoRA weights into Qwen-Image-Edit-2511 DiT.

Saves a full model directory that can be loaded like the base model, with LoRA
baked into transformer weights (no runtime adapter).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def find_lora_file(lora_path: Path) -> Path:
    if lora_path.is_file():
        return lora_path
    candidates = sorted(lora_path.rglob("*.safetensors"))
    # Prefer epoch/final checkpoints
    preferred = [p for p in candidates if "epoch" in p.name.lower() or "lora" in p.name.lower()]
    pool = preferred or candidates
    if not pool:
        raise FileNotFoundError(f"no .safetensors under {lora_path}")
    # pick largest as heuristic for full adapter
    pool.sort(key=lambda p: p.stat().st_size, reverse=True)
    return pool[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    base = Path(args.base_model)
    out = Path(args.out_dir)
    lora_file = find_lora_file(Path(args.lora_path))
    print(f"base={base}")
    print(f"lora={lora_file}")
    print(f"out={out}")

    from diffusers import QwenImageEditPlusPipeline

    dtype = torch.bfloat16
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        str(base),
        torch_dtype=dtype,
        local_files_only=True,
    )

    # DiffSynth saves with remove_prefix_in_ckpt="pipe.dit." — keys often look like
    # blocks.*.to_q.lora_A.weight. Diffusers loader expects transformer.* prefix.
    try:
        pipe.load_lora_weights(str(lora_file.parent if lora_file.is_file() else lora_file), weight_name=lora_file.name)
    except Exception as e1:
        print(f"direct load_lora_weights failed: {e1}; trying state_dict remap")
        from safetensors.torch import load_file, save_file

        sd = load_file(str(lora_file))
        remapped = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("pipe.dit."):
                nk = nk[len("pipe.dit.") :]
            if not nk.startswith("transformer."):
                nk = "transformer." + nk
            nk = nk.replace(".lora_A.default.", ".lora_A.")
            nk = nk.replace(".lora_B.default.", ".lora_B.")
            remapped[nk] = v
        tmp = Path(args.lora_path) / "_remapped_for_diffusers.safetensors"
        if Path(args.lora_path).is_file():
            tmp = Path(args.lora_path).parent / "_remapped_for_diffusers.safetensors"
        save_file(remapped, str(tmp))
        print(f"wrote remapped {tmp} keys={len(remapped)}")
        pipe.load_lora_weights(str(tmp.parent), weight_name=tmp.name)

    # 拼接：把 LoRA 增量融进底座线性层
    pipe.fuse_lora(lora_scale=args.lora_scale)
    try:
        pipe.unload_lora_weights()
    except Exception as e:
        print(f"unload_lora_weights warn: {e}")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # Copy non-transformer parts from base, save fused transformer via pipeline save
    print("saving fused pipeline...")
    pipe.save_pretrained(str(out), safe_serialization=True)

    meta = {
        "base_model": str(base),
        "lora_file": str(lora_file),
        "lora_scale": args.lora_scale,
        "note": "LoRA fused into transformer weights (拼接), load like base model",
    }
    (out / "fuse_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("DONE fused model at", out)


if __name__ == "__main__":
    main()
