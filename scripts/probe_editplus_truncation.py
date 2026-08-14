#!/usr/bin/env python3
"""Measure whether QwenImageEditPlusPipeline truncates long prompts.

EditPlus differs from T2I:
  - T2I: tokenizer(truncation=True, max_length=tokenizer_max_length+drop_idx)
         then encode_prompt slices embeds to max_sequence_length
  - EditPlus: processor(no truncation); encode_prompt ignores max_sequence_length
  - Both: check_inputs raises if max_sequence_length > 1024 (API guard only)

Example (case02 live v2, 4 condition images after pipeline resize):
  prompt_chars≈2000, prompt_only_tokens≈1333, multimodal input_ids≈2200
  after_drop embeds_len >> 1024 → NO text truncation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image


def calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    width = int((target_area * ratio) ** 0.5)
    height = int(target_area / max(width, 1))
    return width, height


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--images", nargs="+", required=True, help="source then products")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--condition-area", type=int, default=384 * 384)
    args = ap.parse_args()

    from diffusers import QwenImageEditPlusPipeline

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    imgs = [Image.open(p).convert("RGB") for p in args.images]

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, local_files_only=True
    )

    cond = []
    for img in imgs:
        w, h = img.size
        cw, ch = calculate_dimensions(args.condition_area, w / h)
        cond.append(pipe.image_processor.resize(img, ch, cw))

    template = pipe.prompt_template_encode
    drop_idx = pipe.prompt_template_encode_start_idx
    img_t = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
    base = "".join(img_t.format(i + 1) for i in range(len(cond)))
    txt = [template.format(base + prompt)]
    mi = pipe.processor(text=txt, images=cond, padding=True, return_tensors="pt")
    ids = mi["input_ids"][0].tolist()
    tok = pipe.processor.tokenizer
    pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    n_pad = sum(1 for t in ids if t == pad_id)
    prompt_only = int(tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1])

    report = {
        "prompt_chars": len(prompt),
        "n_images": len(cond),
        "condition_sizes": [list(i.size) for i in cond],
        "input_ids_len": len(ids),
        "image_pad_tokens": n_pad,
        "non_image_pad_tokens": len(ids) - n_pad,
        "prompt_only_tokens": prompt_only,
        "after_drop_idx_embeds_len": len(ids) - drop_idx,
        "tokenizer_model_max_length": tok.model_max_length,
        "tokenizer_max_length_attr": pipe.tokenizer_max_length,
        "editplus_slices_to_max_sequence_length": False,
        "check_inputs_cap": 1024,
        "verdict": (
            "NO_TRUNCATION"
            if (len(ids) - drop_idx) > 1024 and prompt_only > 1024
            else "CHECK_MANUALLY"
        ),
        "note": (
            "If after_drop_idx_embeds_len > 1024 and processor did not error, "
            "the full multimodal sequence reaches the DiT. Raising check_inputs "
            "cap alone does not change EditPlus quality."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
