#!/usr/bin/env python3
"""make_pair_batch.py — generate a fresh (person, garment) pairing for a new synth batch.

VITON-HD ships one fixed pairing in `train_pairs.txt`, which is exactly what batch 1
consumed, so every person appears once. To grow the dataset we generate additional
permutations, guaranteeing:

1. **no self-pair** — a person is never paired with the garment they already wear
   (same id would make the task reconstruction, not swap; see docs/KNOWLEDGE.md)
2. **no repeat across batches** — every (person, garment) tuple is new, checked
   against the pairs already present in previous batches

Reads previous batches from their `manifest.jsonl` (fields `person`, `cloth`) and/or
plain pairs files, so it stays correct even if a batch was produced elsewhere.

Example:
    python scripts/make_pair_batch.py \\
        --viton-root $QWEN_VTON_DATA/raw/viton_hd \\
        --prev $QWEN_VTON_DATA/synth/idm_unpaired_train \\
        --out $QWEN_VTON_DATA/synth/idm_unpaired_train_b2/pairs_b2.txt \\
        --batch-id b2 --seed 2
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path


def load_base_pairs(pairs_file: Path) -> list[tuple[str, str]]:
    rows = []
    for line in pairs_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))
    if not rows:
        raise SystemExit(f"no pairs parsed from {pairs_file}")
    return rows


def load_used_pairs(prev_dirs: list[Path]) -> set[tuple[str, str]]:
    """Collect (person_basename, cloth_basename) already synthesised."""
    used: set[tuple[str, str]] = set()
    for d in prev_dirs:
        man = d / "manifest.jsonl"
        if man.is_file():
            n0 = len(used)
            for line in man.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                used.add((Path(rec["person"]).name, Path(rec["cloth"]).name))
            print(f"  {d.name}/manifest.jsonl -> +{len(used) - n0} pairs")
            continue
        for pf in sorted(d.glob("pairs_*.txt")):
            n0 = len(used)
            for p, c in load_base_pairs(pf):
                used.add((Path(p).name, Path(c).name))
            print(f"  {pf.name} -> +{len(used) - n0} pairs")
    return used


def build_batch(
    persons: list[str],
    garments: list[str],
    used: set[tuple[str, str]],
    rng: random.Random,
    max_rounds: int = 200,
) -> tuple[list[tuple[str, str]], dict]:
    """Assign each person one garment: no self-pair, no reuse of an existing pair."""
    shuffled = garments[:]
    rng.shuffle(shuffled)
    assign = dict(zip(persons, shuffled))

    def bad(p: str, c: str) -> bool:
        return Path(p).stem == Path(c).stem or (p, c) in used

    rounds = 0
    while rounds < max_rounds:
        conflicts = [p for p in persons if bad(p, assign[p])]
        if not conflicts:
            break
        # Swap each conflicting person's garment with a random other person's.
        for p in conflicts:
            q = rng.choice(persons)
            assign[p], assign[q] = assign[q], assign[p]
        rounds += 1

    pairs = [(p, assign[p]) for p in persons if not bad(p, assign[p])]
    dropped = len(persons) - len(pairs)
    stats = {
        "requested": len(persons),
        "produced": len(pairs),
        "dropped_unresolvable": dropped,
        "resolve_rounds": rounds,
    }
    return pairs, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viton-root", required=True, help="…/raw/viton_hd")
    ap.add_argument("--phase", default="train")
    ap.add_argument("--base-pairs", default="", help="defaults to <viton-root>/<phase>_pairs.txt")
    ap.add_argument(
        "--prev",
        action="append",
        default=[],
        help="previous batch dir(s) with manifest.jsonl or pairs_*.txt; repeatable",
    )
    ap.add_argument("--out", required=True, help="pairs file to write")
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap produced pairs")
    args = ap.parse_args()

    viton = Path(args.viton_root)
    base_pairs_file = (
        Path(args.base_pairs) if args.base_pairs else viton / f"{args.phase}_pairs.txt"
    )
    base = load_base_pairs(base_pairs_file)
    persons = sorted({p for p, _ in base})
    garments = sorted({c for _, c in base})
    print(f"base pairs file: {base_pairs_file}")
    print(f"  persons={len(persons)} garments={len(garments)}")

    print("previous batches:")
    used = load_used_pairs([Path(p) for p in args.prev]) if args.prev else set()
    print(f"  total already-used pairs: {len(used)}")

    rng = random.Random(args.seed)
    pairs, stats = build_batch(persons, garments, used, rng)
    if args.limit:
        pairs = pairs[: args.limit]
        stats["produced"] = len(pairs)

    # hard verification before writing
    for p, c in pairs:
        assert Path(p).stem != Path(c).stem, f"self-pair leaked: {p} {c}"
        assert (p, c) not in used, f"duplicate leaked: {p} {c}"
    assert len({(p, c) for p, c in pairs}) == len(pairs), "intra-batch duplicate"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{p}\t{c}\n" for p, c in pairs), encoding="utf-8")

    meta = {
        "batch_id": args.batch_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "phase": args.phase,
        "base_pairs_file": str(base_pairs_file),
        "pairs_file": str(out),
        "prev_batches": [str(p) for p in args.prev],
        "n_prev_used_pairs": len(used),
        **stats,
        "guarantees": [
            "no person paired with their own garment id",
            "no (person, garment) tuple repeated from a previous batch",
            "no duplicate within this batch",
        ],
    }
    (out.parent / f"batch_meta_{args.batch_id}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nwrote {out} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
