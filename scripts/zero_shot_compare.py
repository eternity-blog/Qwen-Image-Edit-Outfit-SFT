#!/usr/bin/env python3
"""Zero-shot Qwen-Image-Edit-2511 vs GPT Image 2 keyframe compare.

Fair mode (--fair) matches GPT Image 2 inputs as closely as recorded:
  - same source frame
  - same production keyframe_prompt
  - all TestSet product refs in order (image1=source, image2+=products)
  - same output canvas (default 1088x1920)
  - empty negative prompt
  - max_sequence_length (EditPlus: API check only; encode path does NOT truncate —
    unlike T2I which slices embeds to max_sequence_length)

Uses an existing outfit_v2 run:
  real_frames/  keyframes/  outfit_spec_v2.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL_CANDIDATES = [
    "/data/agent/hf_models/Qwen-Image-Edit-2511",
    "/data/agent/hf_models/Qwen/Qwen-Image-Edit-2511",
]


def resolve_model_dir(explicit: Optional[str]) -> Path:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(DEFAULT_MODEL_CANDIDATES)
    for c in candidates:
        p = Path(c)
        if (p / "model_index.json").is_file():
            return p
    raise FileNotFoundError(
        "model_index.json not found in: " + ", ".join(str(x) for x in candidates)
    )


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def garment_prompt_short() -> str:
    """Same short locality prompt used in VTON LoRA training metadata."""
    return (
        "Edit image 1 only. Replace the clothing on the person in image 1 with the "
        "garment shown in image 2. Keep the person's identity, face, hair, pose, "
        "hands, body shape, camera, and background unchanged. Do not redraw the scene."
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _engine_root() -> Path:
    # Optional: sibling kling-aigc-engine for shot helpers (facing / overlay).
    return Path(__file__).resolve().parents[2] / "kling-aigc-engine"


def _overlay_and_facing_from_shot(shot: dict, *, is_tail: bool) -> tuple[str, str]:
    """Best-effort shot metadata; falls back when production engine is absent."""
    overlay, facing = "none", ""
    engine = _engine_root()
    if engine.is_dir():
        if str(engine) not in sys.path:
            sys.path.insert(0, str(engine))
        try:
            from kling_aigc_base.prompt import outfit as eng  # noqa: WPS433

            overlay = eng.promotional_overlay_placement(shot if isinstance(shot, dict) else {})
            facing = eng.keyframe_facing_from_shot(
                shot if isinstance(shot, dict) else {},
                is_tail=is_tail,
            )
            return overlay, facing
        except Exception:
            pass
    # Heuristics without engine
    if isinstance(shot, dict):
        text = " ".join(
            str(shot.get(k) or "")
            for k in ("keyframe_caption", "directive", "facing", "view")
        ).lower()
        if "back" in text or "背面" in text:
            facing = "back"
        elif "side" in text or "侧面" in text:
            facing = "side"
        elif "front" in text or "正面" in text:
            facing = "front"
        if shot.get("overlay_placement"):
            overlay = str(shot["overlay_placement"])
    return overlay, facing


def build_live_v2_prompt(
    *,
    product_text: str,
    product_visual_facts: str,
    shot: dict,
    role: str,
    n_product_refs: int,
    canvas_size: str,
) -> str:
    """Build Outfit v2 garment-only keyframe prompt (vendored full template)."""
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from prompts import outfit_v2 as prompts  # noqa: WPS433

    is_tail = role == "end"
    selling_point = prompts.product_selling_point_from_text(product_text)
    overlay_placement, facing = _overlay_and_facing_from_shot(
        shot if isinstance(shot, dict) else {},
        is_tail=is_tail,
    )
    roles = prompts.default_garment_roles(n_product_refs=max(1, n_product_refs))
    prompt = prompts.outfit_garment_only_keyframe_prompt(
        product_text,
        product_visual_facts,
        selling_point,
        roles,
        is_tail=is_tail,
        overlay_placement=overlay_placement,
        facing=facing,
    )
    prompt += (
        f"\n\n【输出画布】严格输出 {canvas_size}，内容铺满整个画布。"
        "禁止黑边、白边、彩色边框、信箱条、留白或把竖图缩在横向画布中央；"
        "需要扩展构图时自然补全背景，但不得裁掉人物主体或商品关键结构。"
    )
    return prompt


def garment_prompt_simple(product_text: str, directive: str = "") -> str:
    product_text = " ".join(product_text.strip().split())
    directive = directive.strip() or "按原帧可见部位和遮挡关系完成换装。"
    return f"""Edit image 1 only. Do not create a new poster or advertisement.
Image 1 is the only base frame (composition / pose / background / camera / person identity ground truth).
Image 2+ are product references for garment appearance only (style, color, fabric, cut, logos on the garment).
Task: replace the clothing on the person in image 1 with the product shown in the product reference images ({product_text}).
Keep everything else in image 1 unchanged: same person, same pose, same hands, same background, same camera, same framing.
Do not redraw the background. Do not change aspect ratio or crop. Do not invent ice/studio backdrops.
Do not add new Chinese marketing copy, price tags, or brand posters; remove platform watermarks/UI if present.
Directive: {directive}"""


def parse_size(text: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", text or "")
    if not m:
        raise ValueError(f"bad --size {text!r}, expected like 1088x1920")
    return int(m.group(1)), int(m.group(2))


def round_size(w: int, h: int, multiple: int = 16) -> Tuple[int, int]:
    w = max(multiple, int(round(w / multiple) * multiple))
    h = max(multiple, int(round(h / multiple) * multiple))
    return w, h


def discover_product_images(testset_dir: Path, case_id: str, limit: int = 99) -> List[str]:
    paths: List[Path] = []
    for ext in ("png", "jpg", "jpeg", "webp", "PNG", "JPG", "JPEG", "WEBP"):
        paths.extend(sorted(testset_dir.glob(f"{case_id}_*.{ext}")))
    seen = set()
    out: List[str] = []
    for p in paths:
        if not re.fullmatch(rf"{re.escape(case_id)}_\d+\.[A-Za-z0-9]+", p.name):
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(str(p))
        if len(out) >= limit:
            break
    return out


def read_product_text(testset_dir: Path, case_id: str) -> str:
    p = testset_dir / f"{case_id}_text.md"
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return f"case {case_id} product"


@dataclass
class Sample:
    case_id: str
    shot_index: int
    role: str
    source_path: str
    gpt_path: str
    product_paths: List[str]
    prompt: str
    product_text: str
    out_width: int
    out_height: int


@dataclass
class MetricRow:
    case_id: str
    shot_index: int
    role: str
    source: str
    gpt: str
    qwen: str
    mad_gpt: float
    mad_qwen: float
    hist_gpt: float
    hist_qwen: float
    size_gpt: str
    size_qwen: str
    elapsed_sec: float
    prompt_mode: str
    n_products: int
    prompt_chars: int


def samples_from_outfit_run(
    run_dir: Path,
    testset_dir: Path,
    case_id: str,
    roles: Sequence[str],
    shot_indices: Optional[Sequence[int]],
    max_samples: int,
    prompt_mode: str,
    product_limit: int,
    out_width: int,
    out_height: int,
    match_gpt_size: bool,
    product_visual_facts: str = "",
) -> List[Sample]:
    spec_path = run_dir / "outfit_spec_v2.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"missing {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    shots = spec.get("shots") or []
    product_paths = discover_product_images(testset_dir, case_id, limit=product_limit)
    product_text = read_product_text(testset_dir, case_id)
    if not product_paths:
        raise FileNotFoundError(f"no product images for case {case_id} in {testset_dir}")
    facts = (product_visual_facts or "").strip()
    if not facts:
        meta = spec.get("meta") if isinstance(spec.get("meta"), dict) else {}
        facts = str(meta.get("product_visual_facts") or "").strip()
    if not facts:
        facts = "只以图2主商品图中清晰可见的商品本体为准"

    wanted = set(int(x) for x in shot_indices) if shot_indices else None
    out: List[Sample] = []
    for shot in shots:
        idx = int(shot.get("index", -1))
        if wanted is not None and idx not in wanted:
            continue
        shot_obj = shot.get("shot") if isinstance(shot.get("shot"), dict) else shot
        for role in roles:
            src_key = "source_start_frame" if role == "start" else "source_end_frame"
            gpt_key = "keyframe_start" if role == "start" else "keyframe_end"
            src = shot.get(src_key) or ""
            gpt = shot.get(gpt_key) or ""
            if not (src and gpt and Path(src).is_file() and Path(gpt).is_file()):
                continue
            ow, oh = out_width, out_height
            if match_gpt_size:
                with Image.open(gpt) as gim:
                    ow, oh = gim.size
            ow, oh = round_size(ow, oh)

            if prompt_mode == "production":
                prompt = str(shot.get("keyframe_prompt") or "").strip()
                if not prompt:
                    prompt = garment_prompt_simple(product_text)
            elif prompt_mode == "short":
                prompt = garment_prompt_short()
            elif prompt_mode == "v2":
                prompt = build_live_v2_prompt(
                    product_text=product_text,
                    product_visual_facts=facts,
                    shot=shot_obj if isinstance(shot_obj, dict) else {},
                    role=role,
                    n_product_refs=len(product_paths),
                    canvas_size=f"{ow}x{oh}",
                )
            else:
                prompt = garment_prompt_simple(product_text)

            out.append(
                Sample(
                    case_id=case_id,
                    shot_index=idx,
                    role=role,
                    source_path=src,
                    gpt_path=gpt,
                    product_paths=list(product_paths),
                    prompt=prompt,
                    product_text=product_text,
                    out_width=ow,
                    out_height=oh,
                )
            )
            if len(out) >= max_samples:
                return out
    return out


def mad_and_hist(a: Image.Image, b: Image.Image) -> Tuple[float, float]:
    ra = a.convert("RGB")
    rb = b.convert("RGB").resize(ra.size, Image.Resampling.LANCZOS)
    ga = np.asarray(ra.convert("L"), dtype=np.float32)
    gb = np.asarray(rb.convert("L"), dtype=np.float32)
    mad = float(np.mean(np.abs(ga - gb)))
    ha = cv2.cvtColor(np.asarray(ra), cv2.COLOR_RGB2HSV)
    hb = cv2.cvtColor(np.asarray(rb), cv2.COLOR_RGB2HSV)
    hist_a = cv2.calcHist([ha], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hb], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    corr = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
    return mad, corr


def label_image(img: Image.Image, label: str, width: int = 360) -> Image.Image:
    im = img.convert("RGB").copy()
    w, h = im.size
    nh = max(1, int(round(h * (width / float(w)))))
    im = im.resize((width, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, nh + 28), (245, 245, 245))
    canvas.paste(im, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, 28], fill=(30, 30, 30))
    font = ImageFont.load_default()
    draw.text((8, 8), label, fill=(255, 255, 255), font=font)
    return canvas


def make_grid(source: Image.Image, gpt: Image.Image, qwen: Image.Image, title: str) -> Image.Image:
    panels = [
        label_image(source, "source"),
        label_image(gpt, "GPT Image 2"),
        label_image(qwen, "Qwen-2511 fair"),
    ]
    gap = 8
    total_w = sum(p.size[0] for p in panels) + gap * (len(panels) - 1)
    total_h = max(p.size[1] for p in panels) + 36
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    x = 0
    for p in panels:
        canvas.paste(p, (x, 36))
        x += p.size[0] + gap
    return canvas


def load_pipeline(model_dir: Path, device: str, dtype: torch.dtype, cpu_offload: bool):
    from diffusers import QwenImageEditPlusPipeline

    print(
        f"loading pipeline from {model_dir} dtype={dtype} device={device} "
        f"cpu_offload={cpu_offload}",
        flush=True,
    )
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    elif device.startswith("cuda"):
        pipe.to(device)
    else:
        pipe.to("cpu")
    pipe.set_progress_bar_config(disable=None)
    return pipe


def run_edit(
    pipe,
    source: Image.Image,
    products: Sequence[Image.Image],
    prompt: str,
    steps: int,
    seed: int,
    true_cfg_scale: float,
    width: int,
    height: int,
    negative_prompt: str,
    max_sequence_length: int,
) -> Image.Image:
    images = [source] + list(products)
    inputs = {
        "image": images,
        "prompt": prompt,
        "generator": torch.Generator(device="cpu").manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
        "width": width,
        "height": height,
        "max_sequence_length": max_sequence_length,
    }
    print(
        f"edit size={width}x{height} n_images={len(images)} "
        f"prompt_chars={len(prompt)} max_seq={max_sequence_length}",
        flush=True,
    )
    with torch.inference_mode():
        out = pipe(**inputs)
    return out.images[0]


def parse_shot_list(text: Optional[str]) -> Optional[List[int]]:
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip() != ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--testset-dir", required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--true-cfg-scale", type=float, default=4.0)
    ap.add_argument(
        "--prompt-mode",
        choices=["short", "simple", "production", "v2"],
        default="short",
        help="short=LoRA-train; simple=medium; production=stored keyframe_prompt; v2=live Outfit v2 template",
    )
    ap.add_argument(
        "--product-visual-facts",
        default="",
        help="For --prompt-mode v2: product visual facts string (else meta or fallback)",
    )
    ap.add_argument("--roles", default="start")
    ap.add_argument("--shots", default="")
    ap.add_argument("--max-samples", type=int, default=4)
    ap.add_argument("--product-limit", type=int, default=2)
    ap.add_argument("--size", default="", help="WxH; empty + --match-gpt-size uses GPT canvas")
    ap.add_argument("--match-gpt-size", action="store_true", default=False)
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--max-sequence-length", type=int, default=512)
    ap.add_argument("--cpu-offload", action="store_true", default=False)
    ap.add_argument(
        "--fair",
        action="store_true",
        help="Match GPT inputs: production prompt, all products, GPT canvas, no neg prompt, long seq",
    )
    args = ap.parse_args()

    if args.fair:
        args.prompt_mode = "production"
        args.product_limit = max(args.product_limit, 99)
        args.match_gpt_size = True
        args.negative_prompt = " "
        # EditPlus check_inputs rejects >1024 but does not slice embeds; keep <=1024
        # so the call succeeds. Live v2 ~1333 text tokens still fully encoded.
        args.max_sequence_length = min(max(args.max_sequence_length, 1024), 1024)
        if not args.size:
            args.size = "1088x1920"

    if args.prompt_mode == "v2":
        # Same as fair: satisfy API check only; EditPlus encode ignores this for truncation.
        args.max_sequence_length = min(max(args.max_sequence_length, 1024), 1024)

    out_w, out_h = (1088, 1920)
    if args.size:
        out_w, out_h = parse_size(args.size)

    run_dir = Path(args.run_dir)
    testset_dir = Path(args.testset_dir)
    out_dir = Path(args.out_dir)
    qwen_dir = out_dir / "qwen"
    grid_dir = out_dir / "grids"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    model_dir = resolve_model_dir(args.model_dir or None)
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    samples = samples_from_outfit_run(
        run_dir=run_dir,
        testset_dir=testset_dir,
        case_id=args.case_id,
        roles=roles,
        shot_indices=parse_shot_list(args.shots),
        max_samples=args.max_samples,
        prompt_mode=args.prompt_mode,
        product_limit=args.product_limit,
        out_width=out_w,
        out_height=out_h,
        match_gpt_size=args.match_gpt_size,
        product_visual_facts=args.product_visual_facts,
    )
    if not samples:
        raise SystemExit("no samples found; check --run-dir / --shots / --roles")

    parity = {
        "fair": bool(args.fair),
        "aligned": {
            "source_frame": "same real_frames path as GPT run",
            "prompt": (
                "live Outfit v2 garment_only template"
                if args.prompt_mode == "v2"
                else (
                    "exact outfit_spec keyframe_prompt"
                    if args.prompt_mode == "production"
                    else ("short VTON-train style" if args.prompt_mode == "short" else "simple")
                )
            ),
            "product_images": samples[0].product_paths,
            "image_order": ["source", *[f"product[{i}]" for i in range(len(samples[0].product_paths))]],
            "output_size": f"{samples[0].out_width}x{samples[0].out_height}",
            "negative_prompt": args.negative_prompt,
            "max_sequence_length": args.max_sequence_length,
        },
        "known_gaps": [
            "GPT may have used refine attempts / delivery_fallback; stored keyframe_prompt is the recorded final prompt, not necessarily attempt-1.",
            "GPT API may internally resize/compress uploads; Qwen reads local files directly.",
            "Seed / sampler not shared with GPT Image 2.",
            "EditPlus max_sequence_length>1024 is rejected by check_inputs only; encode_prompt does NOT slice (unlike T2I). Case02 live v2 (~1333 prompt tokens) is fully encoded — raising the cap is a no-op for quality.",
            "IDM LoRA was trained on short VTON-style prompts; domain gap vs live v2 prompts is the main quality limiter (retrain with v2-style prompts next).",
        ],
    }
    (out_dir / "input_parity.json").write_text(
        json.dumps(parity, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta = {
        "model_dir": str(model_dir),
        "run_dir": str(run_dir),
        "case_id": args.case_id,
        "prompt_mode": args.prompt_mode,
        "fair": bool(args.fair),
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device,
        "max_sequence_length": args.max_sequence_length,
        "negative_prompt": args.negative_prompt,
        "samples": [asdict(s) for s in samples],
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # High-res fair canvas often needs offload on shared GPUs.
    cpu_offload = args.cpu_offload or (args.fair and max(samples[0].out_width, samples[0].out_height) >= 1536)
    pipe = load_pipeline(model_dir, args.device, dtype, cpu_offload=cpu_offload)

    metrics_path = out_dir / "metrics.jsonl"
    rows: List[MetricRow] = []
    with metrics_path.open("w", encoding="utf-8") as mf:
        for i, sample in enumerate(samples):
            tag = f"{sample.shot_index:02d}_{sample.role}"
            print(f"\n=== [{i+1}/{len(samples)}] {tag} ===", flush=True)
            source = load_rgb(sample.source_path)
            gpt = load_rgb(sample.gpt_path)
            products = [load_rgb(p) for p in sample.product_paths]
            t0 = time.time()
            qwen = run_edit(
                pipe,
                source=source,
                products=products,
                prompt=sample.prompt,
                steps=args.steps,
                seed=args.seed,
                true_cfg_scale=args.true_cfg_scale,
                width=sample.out_width,
                height=sample.out_height,
                negative_prompt=args.negative_prompt if args.negative_prompt else " ",
                max_sequence_length=args.max_sequence_length,
            )
            elapsed = time.time() - t0
            qwen_path = qwen_dir / f"{tag}.png"
            qwen.save(qwen_path)

            mad_gpt, hist_gpt = mad_and_hist(source, gpt)
            mad_qwen, hist_qwen = mad_and_hist(source, qwen)
            row = MetricRow(
                case_id=sample.case_id,
                shot_index=sample.shot_index,
                role=sample.role,
                source=sample.source_path,
                gpt=sample.gpt_path,
                qwen=str(qwen_path),
                mad_gpt=round(mad_gpt, 3),
                mad_qwen=round(mad_qwen, 3),
                hist_gpt=round(hist_gpt, 4),
                hist_qwen=round(hist_qwen, 4),
                size_gpt=f"{gpt.size[0]}x{gpt.size[1]}",
                size_qwen=f"{qwen.size[0]}x{qwen.size[1]}",
                elapsed_sec=round(elapsed, 2),
                prompt_mode=args.prompt_mode,
                n_products=len(sample.product_paths),
                prompt_chars=len(sample.prompt),
            )
            rows.append(row)
            mf.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            mf.flush()

            grid = make_grid(
                source,
                gpt,
                qwen,
                title=(
                    f"FAIR case{sample.case_id} shot{sample.shot_index} {sample.role} | "
                    f"MAD gpt={row.mad_gpt} qwen={row.mad_qwen} | "
                    f"products={row.n_products} prompt_chars={row.prompt_chars}"
                ),
            )
            grid.save(grid_dir / f"{tag}_compare.jpg", quality=92)
            print(
                f"saved {qwen_path.name} elapsed={elapsed:.1f}s "
                f"MAD gpt/qwen={row.mad_gpt}/{row.mad_qwen} "
                f"size={row.size_qwen}",
                flush=True,
            )

    lines = [
        "# Fair compare: Qwen-Image-Edit-2511 vs GPT Image 2",
        "",
        f"- case: `{args.case_id}`",
        f"- fair: `{bool(args.fair)}`",
        f"- model: `{model_dir}`",
        f"- prompt_mode: `{args.prompt_mode}`",
        f"- products: `{len(samples[0].product_paths)}`",
        f"- canvas: `{samples[0].out_width}x{samples[0].out_height}`",
        f"- max_sequence_length: `{args.max_sequence_length}`",
        f"- steps/seed: `{args.steps}` / `{args.seed}`",
        "",
        "| shot | role | MAD GPT | MAD Qwen | hist GPT | hist Qwen | products | prompt_chars | sec |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.shot_index} | {r.role} | {r.mad_gpt} | {r.mad_qwen} | "
            f"{r.hist_gpt} | {r.hist_qwen} | {r.n_products} | {r.prompt_chars} | {r.elapsed_sec} |"
        )
    lines.extend(
        [
            "",
            "See `input_parity.json` for exact alignment and known gaps.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nDONE", out_dir, flush=True)


if __name__ == "__main__":
    main()
