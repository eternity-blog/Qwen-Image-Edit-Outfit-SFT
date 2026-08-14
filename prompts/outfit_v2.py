#!/usr/bin/env python3
"""Outfit v2 garment-only keyframe prompt (vendored for standalone repo).

Source of truth historically lived in the production outfit pipeline
(`outfit_garment_only_keyframe_prompt`). This module is a self-contained
copy of the garment-only path so this repo does not depend on kling-aigc-engine.
"""

from __future__ import annotations

import re
from typing import Sequence


def product_name_from_text(product_text: str) -> str:
    text = str(product_text or "").strip()
    match = re.search(
        r"(?:商品名称|产品名称|名称)\s*[:：]\s*([^\n。；;]+)",
        text,
    )
    value = match.group(1).strip() if match else ""
    if not value:
        value = next(
            (
                line.strip(" #*-。")
                for line in text.splitlines()
                if line.strip(" #*-。")
            ),
            "目标商品",
        )
    return value.rstrip("。；;")[:36]


def product_selling_point_from_text(product_text: str) -> str:
    """Deterministic short selling-point string from product copy (no LLM)."""
    text = str(product_text or "").strip()
    if not text:
        return product_name_from_text(text)

    lines = [
        re.sub(
            r"^\s*(?:(?:[#>*+\-]+\s*)|(?:\d+\s*[.)、]\s*))*",
            "",
            line,
        ).strip()
        for line in text.splitlines()
        if line.strip()
    ]
    labels = r"核心卖点|商品卖点|产品卖点|卖点|商品亮点|产品亮点"
    candidates: list[str] = []

    for line in lines:
        match = re.match(rf"(?:{labels})\s*[:：]\s*(.+)$", line)
        if match:
            candidates.append(match.group(1))
            break

    if not candidates:
        for index, line in enumerate(lines):
            if not re.fullmatch(rf"(?:{labels})\s*[:：]?", line):
                continue
            for following in lines[index + 1 : index + 4]:
                if re.match(r"[^:：]{1,12}[:：]", following):
                    break
                candidates.append(following)
            break

    if not candidates:
        fact_labels = r"材质|面料|版型|特点|功能|优势|设计亮点|商品描述|产品描述"
        for line in lines:
            match = re.match(rf"(?:{fact_labels})\s*[:：]\s*(.+)$", line)
            if match:
                candidates.append(match.group(1))
            if len(candidates) >= 3:
                break

    fragments: list[str] = []
    for candidate in candidates:
        for part in re.split(r"[，、；;。|/]+", candidate):
            value = re.sub(r"\s+", "", part).strip("“”\"'[]【】()（）")
            if value and value not in fragments:
                fragments.append(value)

    if not fragments:
        return product_name_from_text(text)

    selected: list[str] = []
    for fragment in fragments:
        proposed = " · ".join([*selected, fragment])
        if selected and len(proposed) > 30:
            break
        selected.append(fragment[:30] if not selected else fragment)
        if len(selected) == 3:
            break
    return " · ".join(selected) or product_name_from_text(text)


def _outfit_reference_block(roles: Sequence[str], *, replace_model: bool) -> str:
    lines = ["【输入图片】"]
    lines.extend(f"图{index}：{role}" for index, role in enumerate(roles, 1))
    scope = (
        "图1是画面布局与动作的基准；只允许替换主模特和目标服装，"
        "以及处理指定的商品爆点特效文案与底部字幕。"
        if replace_model
        else "图1是画面和人物的基准；只允许替换目标服装，"
        "以及处理指定的商品爆点特效文案与底部字幕。"
    )
    lines.append(scope)
    lines.append("其他参考图只提供上面标明的特征，不复制其姿势、背景、机位或拼图边框。")
    return "\n".join(lines)


def _outfit_product_fidelity_block() -> str:
    return """【商品真值与服装还原】
图2主商品图是本次要替换的商品、套装组成、基础颜色、色块比例、图案和 Logo 位置的唯一真值。
如果图2是商品展示卡，只读取服装商品本身；卡片背景、边框、模特、道具、广告标题、背景品牌字和装饰不属于商品。
其他商品参考可能是不同视角、细节图或其他配色：只有与图2一致的结构信息可以补充；任何与图2冲突的颜色、图案、文字、Logo 或部件一律忽略。

准确还原图2中服装的版型、轮廓、比例、颜色、材质、结构、领口、袖型、缝线及清晰可见的图案、文字和 Logo。
商品颜色硬锁定为图2：光照只能产生自然高光和阴影，不得漂白、变灰、偏色或用其他参考图的配色。
如果图2是套装或多件装，必须逐件替换图1中对应的上衣、下装、外搭或手持服饰，不得只替换其中一件。

【Logo、图案与正反面硬约束】
只有商品参考中清晰画在服装本体上的 Logo、文字和图案才允许出现，不得把商品卡背景上的品牌字或广告元素贴到服装上。
每个 Logo、文字和图案都按“服装部件 + 正面/背面 + 人物自身左右”锁定，禁止镜像、换边、移到中央或搬到另一面。
当图1展示服装背面时，商品参考中只在正面可见的胸前 Logo、肩部图案、前裤腿 Logo 必须完全不可见；参考图没有清晰展示的背面图案一律不得猜测。
图1旧服装的颜色、图案、文字和 Logo 必须随旧服装彻底移除；禁止保留旧图案的位置作为模板，再把新商品 Logo 填到该位置。
参考未展示的布料区域只按已知版型、颜色和材质自然延展，保持素净；不得新增图案、文字、Logo、口袋、配饰或装饰。"""


def _outfit_current_frame_block(facing: str) -> str:
    if facing == "back":
        return """【当前帧朝向与换装结果】
图1当前展示人物背面。输出必须仍是同一个背面姿势：上衣和下装都替换为图2主商品套装中对应部件的背面，并分别严格使用图2对应部件的基础颜色、版型和材质。
主商品图视觉事实明确写为“背面未知”的部件，其背面保持纯净无标识；如果视觉事实明确描述了背面特征，则只复刻被明确描述的内容。
不得把图2仅在正面可见的胸前 Logo、肩部条纹、前裤腿 Logo 搬到背面，也不得保留图1旧衣服背部大图案的位置或轮廓。"""
    if facing == "front":
        return """【当前帧朝向与换装结果】
图1当前展示人物正面。输出必须仍是同一个正面姿势，并完整替换为图2主商品图的整套配色和部件。
上衣与下装必须同时替换；每一件都严格使用图2对应部件的基础颜色，不得保留或生成图1旧服装的颜色。Logo 和图案只按图2正面的部件与人物自身左右复刻。"""
    if facing == "side":
        return """【当前帧朝向与换装结果】
图1当前展示人物侧面。输出必须仍是同一个侧面姿势，并完整替换为图2主商品图的整套配色和部件。
只显示该侧面和当前遮挡关系下本来可见的商品特征，不得把正面 Logo 搬到侧面或背面。"""
    return """【当前帧朝向与换装结果】
保持图1当前身体朝向和姿势，完整替换为图2主商品图的整套配色和部件；上衣与下装必须同时替换，并分别严格使用图2对应部件的基础颜色。"""


def _outfit_copy_edit_block(
    selling_point: str,
    *,
    overlay_placement: str,
) -> str:
    if overlay_placement != "none":
        placement_rule = (
            "旧花字位于画面下方；将整组新花字移到画面中上部的干净区域，"
            "避开人物脸部和商品主体。"
            if overlay_placement == "move_up"
            else "在源图花字区域完成替换，保持原位置，不得向画面下方延伸。"
        )
        promotional_rule = f"""【商品爆点特效文案（不是字幕）】
将源图商品花字整体替换为“{selling_point}”。{placement_rule}
新文字必须逐字准确；保持原花字的排版层级、对齐、逐行颜色和特效风格。将整段文字作为连续广告标题，不添加项目符号、图标、编号、底板或新特效。
新文案不得超过原花字总行数；优先缩小字号、字距和行距使文字完整。只保留一组商品花字。"""
    else:
        promotional_rule = """【商品爆点特效文案（不是字幕）】
图1没有需要替换的商品卖点花字、功能标签或营销文案。禁止生成商品名、卖点、广告标题、功能标签或任何新的营销文字。"""
    return promotional_rule + """

【字幕与水印】
画面下方的对白字幕、歌词字幕或说明字幕不是商品爆点特效文案：如果存在，将其完整移除并用周围画面自然补全。
最终画面不得生成新的下方字幕。同时移除图1中的平台 Logo、账号、用户名、水印和 UI，但不要误删图2服装商品本身清晰可见的 Logo 或文字。"""


def outfit_garment_only_keyframe_prompt(
    product_text: str,
    product_visual_facts: str,
    selling_point: str,
    roles: Sequence[str],
    *,
    is_tail: bool = False,
    overlay_placement: str = "none",
    facing: str = "",
) -> str:
    """Full Outfit v2 garment-only keyframe prompt (production style, not compressed)."""
    tail_rule = (
        "\n这是同一镜头的尾帧。图1仍是尾时刻姿势、身体朝向、取景、背景和机位的唯一基准；"
        "已生成的同镜首帧只用于对齐人物身份、服装基础颜色、版型、材质和光影连续性。"
        "不复制首帧的姿势、构图或正反面 Logo；尾帧为背面时，首帧正面 Logo 必须消失。"
        if is_tail
        else ""
    )
    return f"""{_outfit_reference_block(roles, replace_model=False)}

【编辑任务】
编辑图1，将人物当前穿着、手持展示或搭在手臂/肩部的目标服装，替换为商品参考图中的服装。
目标商品名称：{product_name_from_text(product_text)}
主商品图视觉事实：{str(product_visual_facts or '只以图2主商品图中清晰可见的商品本体为准').strip()[:1200]}

{_outfit_current_frame_block(facing)}

{_outfit_product_fidelity_block()}

让目标服装自然贴合原人物的身体结构和现有姿势，呈现真实的面料厚度、垂坠、褶皱、拉伸、缝线、遮挡和接触阴影。
保持手指、手臂、头发和其他前景物体与服装之间正确的前后关系，不要出现穿模、粘连、悬空或原服装残留。

【人物保持】
只替换服装，不替换人物。保持图1中人物的身份、脸型、五官、肤色、发型、妆容、年龄感、体型、身体比例、
表情、视线、姿势、动作、手势、身体朝向、人物数量和画面位置不变。

{_outfit_copy_edit_block(selling_point, overlay_placement=overlay_placement)}

【整体保持】
除目标服装和上述文字处理外，保持图1的背景、场景、道具、构图、机位、视角、裁切范围和画面比例不变。
匹配图1的光照方向、阴影、色温、透视和摄影质感，使服装看起来是人物真实穿着并在同一时刻拍摄的，不要有贴图感。{tail_rule}"""


def default_garment_roles(n_product_refs: int = 1) -> list[str]:
    roles = [
        "源视频真实帧；只锁定构图、人物姿态、肢体几何、遮挡、背景与光线。",
        "主商品参考图；只提供商品款式、颜色、材质、结构、Logo 与文字。",
    ]
    for _ in range(max(0, n_product_refs - 1)):
        roles.append(
            "候选商品补充参考，可能是其他视角、细节图或其他配色；"
            "只补充与主商品一致的结构信息。"
        )
    return roles
