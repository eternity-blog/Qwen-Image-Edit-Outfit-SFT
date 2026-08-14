#!/usr/bin/env python3
"""Build Outfit v2 garment-only prompts for IDM synth training (full template).

Uses vendored `prompts.outfit_v2` (same text as production garment-only path).
Training pairs use 2 images (person + cloth); multi-ref is deferred — see TODO.md.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_repo_on_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def build_train_v2_prompt(
    *,
    product_text: str,
    product_visual_facts: str,
    facing: str = "",
    overlay_placement: str = "none",
    is_tail: bool = False,
    n_product_refs: int = 1,
) -> str:
    ensure_repo_on_path()
    from prompts import outfit_v2 as prompts  # noqa: WPS433

    if n_product_refs < 1:
        raise ValueError("n_product_refs must be >= 1")
    if n_product_refs > 1:
        raise ValueError(
            "multi-ref prompts deferred (see TODO.md); pass n_product_refs=1"
        )

    selling_point = prompts.product_selling_point_from_text(product_text)
    roles = prompts.default_garment_roles(n_product_refs=1)
    return prompts.outfit_garment_only_keyframe_prompt(
        product_text,
        product_visual_facts,
        selling_point,
        roles,
        is_tail=is_tail,
        overlay_placement=overlay_placement,
        facing=facing,
    )


def default_product_text(category: str = "upper") -> str:
    if category in ("upper", "upper_body", "tops"):
        return "商品名称：上装\n品类：上装\n核心卖点：按主商品图还原款式与颜色"
    if category in ("lower", "bottoms"):
        return "商品名称：下装\n品类：下装\n核心卖点：按主商品图还原款式与颜色"
    return "商品名称：目标服装\n品类：服装\n核心卖点：按主商品图还原款式与颜色"


def default_visual_facts(category: str = "upper", facing: str = "") -> str:
    base = "只以图2主商品图中清晰可见的商品本体为准"
    if category in ("upper", "upper_body", "tops"):
        base = f"上装；{base}"
    elif category in ("lower", "bottoms"):
        base = f"下装；{base}"
    if facing == "back":
        return f"{base}；背面未知"
    if facing == "front":
        return f"{base}；正面展示"
    return f"{base}；背面未知"


def infer_facing_for_viton() -> str:
    return "front"
