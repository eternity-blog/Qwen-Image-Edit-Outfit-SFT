#!/usr/bin/env bash
# Wait for a synth batch to finish, verify it, then publish it to Hugging Face.
#
# Runs unattended after run_idm_synth_batch.sh: polls until the batch reaches its
# expected image count, merges the per-shard manifests, checks the batch really is
# clean (count matches, no self-pairs, no overlap with earlier batches), and only
# then uploads. A batch that fails verification is never published.
#
# Usage:
#   BATCH_ID=b2 EXPECT=11647 HF_TOKEN_FILE=/root/.hf_token_lx \
#     bash scripts/finish_and_publish_batch.sh
#
# Env:
#   BATCH_ID        batch suffix (default b2)
#   EXPECT          expected image count; 0 = derive from the pairs file
#   PREV            comma list of earlier batch dirs to check overlap against
#   MAX_WAIT_H      give up waiting after this many hours (default 12)
#   HF_TOKEN_FILE   file holding the write token (default /root/.hf_token_lx)
#   SKIP_UPLOAD=1   verify only
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_env.sh"

BATCH_ID="${BATCH_ID:-b2}"
DATA_ROOT_VTON="${QWEN_VTON_DATA:-${DATA_ROOT}/datasets/qwen_vton}"
BATCH_DIR="${BATCH_DIR:-$DATA_ROOT_VTON/synth/idm_unpaired_train_$BATCH_ID}"
PREV="${PREV:-$DATA_ROOT_VTON/synth/idm_unpaired_train}"
EXPECT="${EXPECT:-0}"
MAX_WAIT_H="${MAX_WAIT_H:-12}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/root/.hf_token_lx}"
REPO_ID="${REPO_ID:-lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling}"

if [[ -n "${ENV_DIR:-}" && -x "$ENV_DIR/bin/python" ]]; then
  PY="$ENV_DIR/bin/python"
else
  PY="${PYTHON:-python3}"
fi

PAIRS="$BATCH_DIR/pairs_$BATCH_ID.txt"
if [[ "$EXPECT" == "0" && -f "$PAIRS" ]]; then
  EXPECT="$(wc -l < "$PAIRS" | tr -d ' ')"
fi
[[ "$EXPECT" -gt 0 ]] || { echo "ERROR: cannot determine EXPECT (no $PAIRS)"; exit 1; }

echo "[$(date -Is)] publish pipeline for batch=$BATCH_ID"
echo "  dir=$BATCH_DIR  expect=$EXPECT  repo=$REPO_ID"

# ---- 1. wait ---------------------------------------------------------------
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
last=-1
stall_since=$(date +%s)
while :; do
  n="$(ls "$BATCH_DIR/images" 2>/dev/null | wc -l | tr -d ' ')"
  now=$(date +%s)
  if [[ "$n" -ge "$EXPECT" ]]; then
    echo "[$(date -Is)] synthesis complete: $n/$EXPECT"
    break
  fi
  if [[ "$now" -gt "$deadline" ]]; then
    echo "[$(date -Is)] TIMEOUT after ${MAX_WAIT_H}h at $n/$EXPECT — not publishing"
    exit 1
  fi
  if [[ "$n" -ne "$last" ]]; then
    last="$n"
    stall_since="$now"
    echo "[$(date -Is)] progress $n/$EXPECT ($(( n * 100 / EXPECT ))%)"
  elif [[ $(( now - stall_since )) -gt 1800 ]]; then
    # no new image for 30 min: shards likely died, check before waiting further
    if ! tmux ls 2>/dev/null | grep -q "idm-synth-$BATCH_ID"; then
      echo "[$(date -Is)] shards gone and stalled at $n/$EXPECT — not publishing"
      exit 1
    fi
    echo "[$(date -Is)] WARNING stalled 30min at $n/$EXPECT (shards still alive)"
    stall_since="$now"
  fi
  sleep 120
done

# ---- 2. merge shard manifests ----------------------------------------------
echo "[$(date -Is)] merging shard manifests"
bash "$SCRIPT_DIR/merge_idm_shard_manifests.sh" "$BATCH_DIR" || exit 1

# ---- 3. verify --------------------------------------------------------------
echo "[$(date -Is)] verifying batch"
"$PY" - "$BATCH_DIR" "$PREV" <<'PY' || exit 1
import json, sys
from pathlib import Path

batch = Path(sys.argv[1])
prev_dirs = [Path(p) for p in sys.argv[2].split(",") if p.strip()]

rows = [json.loads(l) for l in (batch / "manifest.jsonl").read_text().splitlines() if l.strip()]
imgs = {p.name for p in (batch / "images").iterdir() if p.is_file()}
print(f"  manifest rows: {len(rows)}   images on disk: {len(imgs)}")

missing = [r["out_name"] for r in rows if r["out_name"] not in imgs]
if missing:
    sys.exit(f"  FAIL: {len(missing)} manifest rows have no image, e.g. {missing[:3]}")

pairs = {(Path(r["person"]).name, Path(r["cloth"]).name) for r in rows}
if len(pairs) != len(rows):
    sys.exit(f"  FAIL: intra-batch duplicate pairs ({len(rows) - len(pairs)})")

self_pairs = [p for p in pairs if Path(p[0]).stem == Path(p[1]).stem]
if self_pairs:
    sys.exit(f"  FAIL: {len(self_pairs)} self-pairs, e.g. {self_pairs[:3]}")

for d in prev_dirs:
    man = d / "manifest.jsonl"
    if not man.is_file():
        print(f"  (skip overlap check, no manifest in {d})")
        continue
    prev = {
        (Path(r["person"]).name, Path(r["cloth"]).name)
        for r in (json.loads(l) for l in man.read_text().splitlines() if l.strip())
    }
    dup = pairs & prev
    if dup:
        sys.exit(f"  FAIL: {len(dup)} pairs overlap {d.name}, e.g. {list(dup)[:3]}")
    print(f"  no overlap with {d.name} ({len(prev)} pairs)")

print("  VERIFY_OK")
PY

if [[ "${SKIP_UPLOAD:-0}" == "1" ]]; then
  echo "[$(date -Is)] SKIP_UPLOAD=1, stopping after verification"
  exit 0
fi

# ---- 4. upload --------------------------------------------------------------
[[ -f "$HF_TOKEN_FILE" ]] || { echo "ERROR: token file $HF_TOKEN_FILE missing"; exit 1; }
export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"

echo "[$(date -Is)] uploading synth/idm_unpaired_train_$BATCH_ID"
for attempt in 1 2 3; do
  if "$PY" "$SCRIPT_DIR/upload_all_synth_to_hf.py" \
      --repo-id "$REPO_ID" \
      --synth "idm_unpaired_train_$BATCH_ID=$BATCH_DIR" \
      --staging "/tmp/hf_upload_$BATCH_ID"; then
    echo "[$(date -Is)] upload succeeded on attempt $attempt"
    break
  fi
  echo "[$(date -Is)] upload attempt $attempt failed"
  [[ "$attempt" == "3" ]] && { echo "giving up"; exit 1; }
  sleep 120
done

# ---- 5. verify remote -------------------------------------------------------
echo "[$(date -Is)] verifying remote"
"$PY" - "$REPO_ID" "idm_unpaired_train_$BATCH_ID" "$EXPECT" <<'PY'
import os, sys
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

repo, name, expect = sys.argv[1], sys.argv[2], int(sys.argv[3])
api = HfApi(token=os.environ["HF_TOKEN"])
n = sum(
    1
    for x in api.list_repo_tree(
        repo, repo_type="dataset", path_in_repo=f"synth/{name}/images", recursive=True
    )
    if isinstance(x, RepoFile)
)
print(f"  remote images under synth/{name}: {n} (expected {expect})")
print("  REMOTE_OK" if n == expect else "  REMOTE_MISMATCH")
PY

rm -rf "/tmp/hf_upload_$BATCH_ID"
echo "[$(date -Is)] PIPELINE_DONE https://huggingface.co/datasets/$REPO_ID/tree/main/synth/idm_unpaired_train_$BATCH_ID"
