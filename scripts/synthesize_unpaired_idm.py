#!/usr/bin/env python3
"""Batch unpaired try-on synthesis with IDM-VTON (teacher for MF-VITON-style data).

Uses precomputed VITON-HD agnostic-mask + densepose (no online parsing).
Saves:
  <out_dir>/images/<person_stem>__<cloth_stem>.jpg
  <out_dir>/manifest.jsonl
  <out_dir>/grids/  (optional small preview)

Example:
  python synthesize_unpaired_idm.py \
    --data-root /data/agent/hf_models/datasets/qwen_vton/raw/viton_hd \
    --pairs test_pairs.txt --phase test --limit 64 \
    --model-dir /data/agent/hf_models/yisol/IDM-VTON \
    --repo-dir /data/agent/hf_models/modules/IDM-VTON \
    --out-dir /data/agent/hf_models/datasets/qwen_vton/synth/idm_unpaired_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data as torch_data
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL, DDPMScheduler


def parse_pairs(path: Path, unpaired: bool) -> List[Tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        im, c = parts[0], parts[1]
        if not unpaired:
            c = im
        rows.append((im, c))
    return rows


class PairDataset(torch_data.Dataset):
    def __init__(
        self,
        data_root: Path,
        phase: str,
        pairs: List[Tuple[str, str]],
        height: int = 1024,
        width: int = 768,
    ):
        self.root = data_root / phase
        self.pairs = pairs
        self.height = height
        self.width = width
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.to_tensor = transforms.ToTensor()
        self.clip_processor = CLIPImageProcessor()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        im_name, c_name = self.pairs[index]
        cloth = Image.open(self.root / "cloth" / c_name).convert("RGB")
        person = Image.open(self.root / "image" / im_name).convert("RGB").resize(
            (self.width, self.height)
        )
        image = self.transform(person)

        mask = Image.open(
            self.root / "agnostic-mask" / im_name.replace(".jpg", "_mask.png")
        ).resize((self.width, self.height))
        mask = self.to_tensor(mask)[:1]
        mask = 1 - mask

        pose = Image.open(self.root / "image-densepose" / im_name).convert("RGB")
        pose = self.transform(pose)

        caption = "model is wearing a upper body garment"
        caption_cloth = "a photo of a upper body garment"

        return {
            "im_name": im_name,
            "c_name": c_name,
            "image": image,
            "cloth_pure": self.transform(cloth.resize((self.width, self.height))),
            "cloth": self.clip_processor(images=cloth, return_tensors="pt").pixel_values[0],
            "inpaint_mask": 1 - mask,
            "caption": caption,
            "caption_cloth": caption_cloth,
            "pose_img": pose,
        }


def pil_to_tensor(images: Image.Image) -> torch.Tensor:
    arr = np.array(images).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def build_pipe(repo_dir: Path, model_dir: Path, device: torch.device):
    sys.path.insert(0, str(repo_dir))
    from src.unet_hacked_tryon import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

    weight_dtype = torch.float16
    noise_scheduler = DDPMScheduler.from_pretrained(model_dir, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet", torch_dtype=weight_dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_dir, subfolder="image_encoder", torch_dtype=weight_dtype
    )
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        model_dir, subfolder="unet_encoder", torch_dtype=weight_dtype
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        model_dir, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        model_dir, subfolder="text_encoder_2", torch_dtype=weight_dtype
    )
    tokenizer_one = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer_2", use_fast=False)

    for m in (unet, vae, image_encoder, unet_encoder, text_encoder_one, text_encoder_two):
        m.requires_grad_(False)
        m.eval()

    pipe = TryonPipeline.from_pretrained(
        model_dir,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        unet_encoder=unet_encoder,
        torch_dtype=weight_dtype,
    ).to(device)
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pairs", default="test_pairs.txt")
    ap.add_argument("--phase", default="test", choices=["train", "test"])
    ap.add_argument("--unpaired", action="store_true", default=True)
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split pairs into N contiguous shards for multi-GPU (with --shard-id).",
    )
    ap.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="0-based shard index when --num-shards > 1.",
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance-scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-dir", default="/data/agent/hf_models/yisol/IDM-VTON")
    ap.add_argument("--repo-dir", default="/data/agent/hf_models/modules/IDM-VTON")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    unpaired = not args.paired
    data_root = Path(args.data_root)
    pairs_path = data_root / args.pairs
    if not pairs_path.is_file():
        # also allow absolute
        pairs_path = Path(args.pairs)
    pairs = parse_pairs(pairs_path, unpaired=unpaired)
    if args.offset:
        pairs = pairs[args.offset :]
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"--shard-id must be in [0, {args.num_shards})")
    if args.num_shards > 1:
        # Contiguous shard: keeps I/O sequential and makes resume easy.
        n = len(pairs)
        start = (n * args.shard_id) // args.num_shards
        end = (n * (args.shard_id + 1)) // args.num_shards
        pairs = pairs[start:end]
    print(
        f"pairs={len(pairs)} unpaired={unpaired} phase={args.phase} "
        f"shard={args.shard_id}/{args.num_shards}"
    )

    out = Path(args.out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    # Per-shard manifests avoid concurrent append races when multi-GPU.
    if args.num_shards > 1:
        manifest_path = out / f"manifest.shard{args.shard_id:02d}.jsonl"
    else:
        manifest_path = out / "manifest.jsonl"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pipe = build_pipe(Path(args.repo_dir), Path(args.model_dir), device)

    ds = PairDataset(data_root, args.phase, pairs, height=args.height, width=args.width)
    loader = torch_data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    done = set()
    # Load all manifests under out_dir so shards can resume safely.
    for mp in sorted(out.glob("manifest*.jsonl")):
        for line in mp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["out_name"])
    print(f"resume_done={len(done)} manifest={manifest_path.name}")

    mf = manifest_path.open("a", encoding="utf-8")
    generator = torch.Generator(device).manual_seed(args.seed + args.shard_id)

    with torch.no_grad():
        for bi, sample in enumerate(loader):
            out_names = []
            skip = True
            for i in range(len(sample["im_name"])):
                stem_p = Path(sample["im_name"][i]).stem
                stem_c = Path(sample["c_name"][i]).stem
                on = f"{stem_p}__{stem_c}.jpg"
                out_names.append(on)
                if on not in done:
                    skip = False
            if skip:
                print(f"[{bi}] skip batch already done")
                continue

            prompt = list(sample["caption"])
            neg = ["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompt)
            # Keep all vision tensors in fp16 to match the half-precision pipeline.
            dtype = torch.float16
            image_embeds = sample["cloth"].to(device=device, dtype=dtype)
            pose_img = sample["pose_img"].to(device=device, dtype=dtype)
            cloth_pure = sample["cloth_pure"].to(device=device, dtype=dtype)
            person = ((sample["image"] + 1.0) / 2.0).to(device=device, dtype=dtype)
            mask = sample["inpaint_mask"].to(device=device, dtype=dtype)

            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=neg,
            )
            prompt_c = list(sample["caption_cloth"])
            (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                prompt_c,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=neg,
            )

            images = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=args.steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_img,
                text_embeds_cloth=prompt_embeds_c,
                cloth=cloth_pure,
                mask_image=mask,
                image=person,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                ip_adapter_image=image_embeds,
            )[0]

            for i, img in enumerate(images):
                on = out_names[i]
                if on in done:
                    continue
                save_path = img_dir / on
                torchvision.utils.save_image(pil_to_tensor(img), str(save_path))
                rec = {
                    "out_name": on,
                    "out_rel": f"images/{on}",
                    "person": f"{args.phase}/image/{sample['im_name'][i]}",
                    "cloth": f"{args.phase}/cloth/{sample['c_name'][i]}",
                    "phase": args.phase,
                    "source": "idm_vton_unpaired",
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mf.flush()
                done.add(on)
            print(f"[{bi}] wrote {out_names}")

    mf.close()
    print("DONE", out, "n=", len(done))


if __name__ == "__main__":
    main()
