#!/usr/bin/env bash
# Merge per-shard IDM manifests into one manifest.jsonl (dedupe by out_name).
set -euo pipefail

SYNTH="${1:?usage: merge_idm_shard_manifests.sh <synth_dir>}"
OUT="$SYNTH/manifest.jsonl"
TMP="$SYNTH/manifest.merged.tmp"

shopt -s nullglob
shards=("$SYNTH"/manifest.shard*.jsonl)
if [[ ${#shards[@]} -eq 0 ]]; then
  echo "no shard manifests under $SYNTH"
  if [[ -f "$OUT" ]]; then
    echo "existing $OUT lines=$(wc -l < "$OUT")"
  fi
  exit 0
fi

python3 - <<PY
from pathlib import Path
synth = Path("$SYNTH")
seen = {}
order = []
# Prefer existing merged/main first, then shards.
paths = []
main = synth / "manifest.jsonl"
if main.is_file():
    paths.append(main)
paths.extend(sorted(synth.glob("manifest.shard*.jsonl")))
for p in paths:
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        import json
        rec = json.loads(line)
        on = rec["out_name"]
        if on not in seen:
            seen[on] = line
            order.append(on)
out = synth / "manifest.merged.tmp"
with out.open("w", encoding="utf-8") as f:
    for on in order:
        f.write(seen[on] + "\n")
print(f"merged={len(order)} from {len(paths)} files -> {out}")
PY

mv -f "$TMP" "$OUT"
echo "wrote $OUT lines=$(wc -l < "$OUT")"
# Keep shard files as backup; do not delete automatically.
