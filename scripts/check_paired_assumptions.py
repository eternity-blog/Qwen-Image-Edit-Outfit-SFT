#!/usr/bin/env python3
"""check_paired_assumptions.py — 检验配对比较的分布假设，并做稳健性交叉验证。

KNOWLEDGE.md 第 10.3 节引用的检验统计量全部由本脚本产出，可直接重跑核对。

回答三个问题：
  1. 配对差值 d_i 是否近似正态（t 检验的前提）——偏度/峰度/Shapiro-Wilk
  2. 若不正态，换成不假设分布的 Wilcoxon 符号秩检验，结论是否改变
  3. 在任意子样本量上（如当年那 6 条），两种检验是否给出不同答案

用法:
    python scripts/check_paired_assumptions.py \\
        --metrics outputs/qwen_kf_zeroshot/holdout_n200_0817/metrics.json \\
        --model lora_v2_lr1e-4 --ref full_sft_b1 --subset 6

依赖 scipy（仅本脚本需要）: pip install scipy
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    raise SystemExit("需要 scipy: pip install scipy")

METRICS = ("mad_vs_teacher", "hist_vs_teacher", "mad_vs_person")


def paired(results: dict, model: str, ref: str, key: str):
    """返回 (ids, 差值列表)，跳过任一侧缺失该指标的样本。"""
    ids, diffs = [], []
    for sid, rec in results[ref].items():
        a, b = results[model].get(sid, {}).get(key), rec.get(key)
        if a is None or b is None:
            continue
        ids.append(sid)
        diffs.append(a - b)
    return ids, diffs


def report(d: list[float], label: str) -> None:
    n = len(d)
    if n < 3:
        print(f"  {label}: 样本量 {n} 过小，跳过")
        return

    mean, sd = st.mean(d), st.stdev(d)
    t, p_t = stats.ttest_1samp(d, 0.0)
    # Wilcoxon 在全为 0 差值时会报错，兜一下
    try:
        w, p_w = stats.wilcoxon(d)
    except ValueError:
        w, p_w = float("nan"), float("nan")

    print(f"\n  --- {label} (n={n}) ---")
    print(f"    均值 {mean:+.4f}   中位数 {st.median(d):+.4f}   标准差 {sd:.4f}")
    print(f"    极差 {min(d):+.3f} ~ {max(d):+.3f}")
    print(f"    Cohen's d_z = {mean / sd:+.3f}   (= t/sqrt(n))")

    print("    正态性:")
    print(f"      偏度 {stats.skew(d):+.3f}   超额峰度 {stats.kurtosis(d):+.3f}")
    if n >= 3:
        sw_W, sw_p = stats.shapiro(d)
        verdict = "拒绝正态" if sw_p < 0.05 else "不拒绝正态"
        print(f"      Shapiro-Wilk W={sw_W:.4f}  p={sw_p:.3e}  -> {verdict}")

    print("    显著性（两种方法交叉验证）:")
    print(f"      配对 t 检验      t={t:+.3f}   p={p_t:.5f}   {'显著' if p_t < 0.05 else '不显著'}")
    print(f"      Wilcoxon 符号秩  W={w:.1f}   p={p_w:.5f}   {'显著' if p_w < 0.05 else '不显著'}")

    agree = (p_t < 0.05) == (p_w < 0.05)
    print(f"      -> 两种方法结论{'一致' if agree else '冲突'}"
          f"{'' if agree else '（此时应以非参数结果为准）'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True, help="eval_viton_holdout.py 产出的 metrics.json")
    ap.add_argument("--model", required=True, help="待检验模型名")
    ap.add_argument("--ref", required=True, help="参照模型名")
    ap.add_argument("--keys", nargs="*", default=list(METRICS), help=f"指标，默认 {METRICS}")
    ap.add_argument("--subset", type=int, default=0,
                    help="额外在前 N 条子集上复算（用于复现小样本结论），0=关闭")
    args = ap.parse_args()

    blob = json.loads(args.metrics.read_text())
    results = blob.get("results", blob)

    missing = [m for m in (args.model, args.ref) if m not in results]
    if missing:
        raise SystemExit(f"metrics 里没有模型 {missing}；可用: {sorted(results)}")

    print(f"配对比较: {args.model}  vs  {args.ref}   (差值 = 前者 − 后者)")

    for key in args.keys:
        ids, d = paired(results, args.model, args.ref, key)
        if not d:
            print(f"\n[{key}] 无可用数据，跳过")
            continue
        print(f"\n[{key}]")
        report(d, "全量")
        if args.subset and args.subset < len(d):
            report(d[: args.subset], f"前 {args.subset} 条子集")


if __name__ == "__main__":
    main()
