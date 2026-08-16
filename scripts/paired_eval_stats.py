#!/usr/bin/env python3
"""paired_eval_stats.py — decide whether eval differences are real or noise.

The holdout eval scores every model on the *same* samples, so a paired comparison
is the right test: for each sample take the per-sample difference against a
reference model, then look at the mean difference relative to its standard error.
Comparing two independent means would throw away that pairing and badly
under-power a 6-sample run.

    python scripts/paired_eval_stats.py \\
        --metrics $OUT/metrics.json --reference full_sft --metric mad_vs_teacher
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--reference", default="full_sft")
    ap.add_argument("--metric", default="mad_vs_teacher")
    ap.add_argument("--exclude", default="base", help="comma list omitted from pairing")
    args = ap.parse_args()

    m = json.loads(Path(args.metrics).read_text())
    res = m["results"]
    ids = [r["id"] for r in m["rows"]]
    names = [n for n in res if all(i in res[n] for i in ids)]
    excl = {x.strip() for x in args.exclude.split(",") if x.strip()}

    def val(n: str, i: str) -> float:
        return res[n][i][args.metric]

    width = max(len(n) for n in names) + 2
    print(f"per-sample {args.metric}")
    print("sample".ljust(24) + "".join(n.rjust(width) for n in names))
    for i in ids:
        print(i.ljust(24) + "".join(f"{val(n, i):{width}.2f}" for n in names))
    print("mean".ljust(24) + "".join(f"{st.fmean(val(n, i) for i in ids):{width}.2f}" for n in names))
    print("stdev".ljust(24) + "".join(f"{st.stdev([val(n, i) for i in ids]):{width}.2f}" for n in names))

    ref = args.reference
    if ref not in res:
        raise SystemExit(f"reference {ref} not in metrics")
    print(f"\npaired vs {ref}  (negative = the other model is closer to the teacher)")
    print(f"  n = {len(ids)} samples; |t| > 2.57 is p<0.05 two-sided at df=5")
    for n in names:
        if n == ref or n in excl:
            continue
        d = [val(n, i) - val(ref, i) for i in ids]
        mean_d = st.fmean(d)
        sd = st.stdev(d)
        se = sd / len(d) ** 0.5
        t = mean_d / se if se else float("inf")
        if abs(t) > 2.57:
            verdict = "significant"
        elif abs(t) > 2.0:
            verdict = "marginal"
        else:
            verdict = "not distinguishable from noise"
        print(f"  {n:<16s} delta={mean_d:+6.2f}  sd={sd:5.2f}  se={se:5.2f}  t={t:+6.2f}  {verdict}")


if __name__ == "__main__":
    main()
