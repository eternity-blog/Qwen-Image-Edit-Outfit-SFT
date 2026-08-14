# CLAUDE.md — Agent bootstrap for this repo

**Audience:** Claude Code (or any coding agent) on a **fresh multi-GPU Linux machine**.  
**Mission:** Rebuild the full experiment stack so a human can start training with one script.  
**Do not** redesign the training stack. Prefer existing scripts under `scripts/` and docs under `docs/`.

Human-facing narrative: `docs/REPRODUCE.md`. This file is the **executable runbook**.

---

## 0. Goal and definition of done

When finished, **all** of the following must be true:

1. Repo cloned; `configs/env.local.sh` exists and is sourced by scripts via `scripts/lib_env.sh`.
2. Conda/venv at `ENV_DIR` has: `torch`, `accelerate`, `deepspeed`, `diffusers`, `transformers`, `peft`, `safetensors`, `huggingface_hub`, and **editable** DiffSynth-Studio.
3. `MODEL_DIR` is a complete local `Qwen-Image-Edit-2511` tree (`transformer/`, `text_encoder/`, `vae/`, `tokenizer/`, `processor/`).
4. `QWEN_VTON_DATA/converted_idm_synth_train_v2/{metadata_train.json,dataset_base}` exists; random metadata paths resolve to real files under `dataset_base`.
5. `bash -n scripts/train_full_sft_zero3.sh` passes; a **dry preflight** (imports + path checks) succeeds.
6. You print the exact train command for this machine (GPU count + `DS_PROFILE`) and **do not** start a long training job unless the user explicitly asks.

**Default training target:** full-parameter DiT SFT via ZeRO-3 (`scripts/train_full_sft_zero3.sh`).  
LoRA (`scripts/train_idm_lora_multigpu.sh`) is optional / lighter.

---

## 1. Hard rules

- Never commit `configs/env.local.sh`, `.env`, tokens, or large weights/data.
- Never re-synthesize IDM data if HF dataset can be used (`lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling`).
- Never launch full DiT with plain DDP / `accelerate launch --multi_gpu` without DeepSpeed ZeRO — it will OOM.
- Always keep `--zero_cond_t` for Qwen-Image-Edit-2511.
- Training entrypoint is DiffSynth’s  
  `$DIFFSYNTH_DIR/examples/qwen_image/model_training/train.py`  
  wrapped by this repo’s shell scripts — do not invent a new `train.py`.
- If the machine needs HTTP proxy for HF/GitHub, set `http_proxy`/`https_proxy` before downloads.
- Long jobs: use `tmux`/`screen` or the cluster scheduler; do not leave multi-hour train attached to a flaky SSH session.
- License: VITON-HD / IDM-VTON derived data are **non-commercial research only** (`NOTICE.md`).

---

## 2. Ask the user once (if unknown)

Collect before guessing:

| Variable | Why |
|---|---|
| Absolute data disk path (large, fast) | `DATA_ROOT` / `QWEN_VTON_DATA` / `OUTPUT_ROOT` |
| GPU count × VRAM | Choose `NUM_PROCESSES` + `DS_PROFILE` |
| HF token (if dataset private) | `HF_TOKEN` |
| Whether VITON-HD already exists on disk | Skip download; only need path |
| Proxy host:port (if any) | HF / pip / git |

If user says “use defaults on this box”, probe with `df -h`, `nvidia-smi -L`, `pwd`, and place bulky data on the largest writable filesystem.

---

## 3. Phase A — Code and env

```bash
# A1. Clone
git clone https://github.com/eternity-blog/Qwen-Image-Edit-Outfit-SFT.git
cd Qwen-Image-Edit-Outfit-SFT
REPO_ROOT="$PWD"

# A2. Pick paths (EDIT THESE)
export DATA_ROOT=/CHANGE_ME/qwen_outfit_data          # large disk
export MODEL_DIR=$DATA_ROOT/models/Qwen-Image-Edit-2511
export DIFFSYNTH_DIR=$DATA_ROOT/modules/DiffSynth-Studio
export QWEN_VTON_DATA=$DATA_ROOT/datasets/qwen_vton
export OUTPUT_ROOT=$DATA_ROOT/outputs
export ENV_DIR=/CHANGE_ME/conda/envs/qwen-image-edit  # or conda env path

mkdir -p "$DATA_ROOT" "$MODEL_DIR" "$DIFFSYNTH_DIR" "$QWEN_VTON_DATA" "$OUTPUT_ROOT"

# A3. Write env.local.sh (gitignored)
cp configs/env.example.sh configs/env.local.sh
cat > configs/env.local.sh <<EOF
export DATA_ROOT=$DATA_ROOT
export MODEL_DIR=$MODEL_DIR
export DIFFSYNTH_DIR=$DIFFSYNTH_DIR
export QWEN_VTON_DATA=$QWEN_VTON_DATA
export OUTPUT_ROOT=$OUTPUT_ROOT
export ENV_DIR=$ENV_DIR
EOF

# A4. Python env (conda example)
# conda create -p "$ENV_DIR" python=3.11 -y
# conda activate "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install -U pip
"$ENV_DIR/bin/python" -m pip install -r requirements.txt
"$ENV_DIR/bin/python" -m pip install deepspeed

# A5. DiffSynth-Studio (required; train.py lives here)
if [[ ! -f "$DIFFSYNTH_DIR/examples/qwen_image/model_training/train.py" ]]; then
  git clone https://github.com/modelscope/DiffSynth-Studio.git "$DIFFSYNTH_DIR"
fi
"$ENV_DIR/bin/python" -m pip install -e "$DIFFSYNTH_DIR"
```

### Verify A

```bash
source configs/env.local.sh
"$ENV_DIR/bin/python" - <<'PY'
import torch, accelerate, deepspeed, diffusers, transformers, peft, safetensors, huggingface_hub
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
import diffsynth
print("diffsynth ok", diffsynth.__file__)
PY
test -f "$DIFFSYNTH_DIR/examples/qwen_image/model_training/train.py"
test -f configs/accelerate_zero3.yaml
```

---

## 4. Phase B — Base model

Download **Qwen/Qwen-Image-Edit-2511** into `MODEL_DIR` (HF CLI or `huggingface_hub.snapshot_download`). Offline mirrors OK if complete.

### Verify B

```bash
source configs/env.local.sh
"$ENV_DIR/bin/python" - <<'PY'
import glob
from pathlib import Path
md = Path("$MODEL_DIR".replace("$MODEL_DIR", __import__("os").environ["MODEL_DIR"]))
need = [
    list((md/"transformer").glob("diffusion_pytorch_model*.safetensors")),
    list((md/"text_encoder").glob("model*.safetensors")),
    list((md/"vae").glob("diffusion_pytorch_model*.safetensors")),
]
assert all(need), need
assert (md/"tokenizer").exists() or True
print("model ok", md)
PY
# Prefer:
ls "$MODEL_DIR/transformer"/diffusion_pytorch_model*.safetensors | head
ls "$MODEL_DIR/text_encoder"/model*.safetensors | head
ls "$MODEL_DIR/vae"/diffusion_pytorch_model*.safetensors | head
```

---

## 5. Phase C — Dataset

### C1. VITON-HD (user-provided, CC BY-NC)

Place (or symlink) so that these exist:

```text
$QWEN_VTON_DATA/raw/viton_hd/train/image/
$QWEN_VTON_DATA/raw/viton_hd/train/cloth/
# test/ optional but recommended
```

**Do not** scrape illegally. If missing, stop and ask the user for the path.

### C2. Synthetic pairs from Hugging Face

```bash
source configs/env.local.sh
export HF_TOKEN="${HF_TOKEN:-}"   # if needed
# Default repo is baked into the script:
#   lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling
bash scripts/prepare_data_from_hf.sh
```

This will:

1. Download HF dataset → `$QWEN_VTON_DATA/from_hf`
2. Flatten `images/part-*` → `synth/*/images/` via symlinks
3. Run `run_convert_idm_v2.sh` → full Outfit v2 prompts + `dataset_base` symlinks

If VITON was missing when the script ran, fix VITON then:

```bash
bash scripts/run_convert_idm_v2.sh
```

### Verify C (mandatory)

```bash
source configs/env.local.sh
"$ENV_DIR/bin/python" - <<'PY'
import json, os
from pathlib import Path
base = Path(os.environ["QWEN_VTON_DATA"]) / "converted_idm_synth_train_v2"
meta = json.load(open(base / "metadata_train.json"))
db = base / "dataset_base"
assert len(meta) > 1000, len(meta)
row = meta[0]
assert (db / row["image"]).is_file(), row["image"]
for p in row["edit_image"]:
    assert (db / p).is_file(), p
assert len(row["prompt"]) > 1000, len(row["prompt"])
print("DATA OK", "n=", len(meta), "prompt_chars=", len(row["prompt"]))
print("dataset_base", db.resolve())
PY
```

Expected ballpark: ~11415 train rows; prompt length ~1592 chars (full v2).

---

## 6. Phase D — Training profile selection

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv
```

| GPUs (80GB class) | Export |
|---|---|
| ≥8 | `NUM_PROCESSES=8` `DS_PROFILE=zero3` |
| 4 | `NUM_PROCESSES=4` `DS_PROFILE=zero2_offload` |
| &lt;4 | Prefer LoRA script instead of full SFT |

Optional warm start from a LoRA-fused tree:

```bash
export INIT_MODEL_DIR=/path/to/qwen_idm_lora_fused
```

Default: `INIT_MODEL_DIR=$MODEL_DIR`.

---

## 7. Phase E — Preflight (do this before real train)

```bash
source configs/env.local.sh
export METADATA=$QWEN_VTON_DATA/converted_idm_synth_train_v2/metadata_train.json
export DATASET_BASE=$QWEN_VTON_DATA/converted_idm_synth_train_v2/dataset_base
export NUM_PROCESSES=8          # adjust
export DS_PROFILE=zero3         # or zero2_offload
export LR=1e-5
export NUM_EPOCHS=1

# Syntax + dependency checks inside the train script’s early exits:
bash -n scripts/train_full_sft_zero3.sh
"$ENV_DIR/bin/python" -c "import deepspeed,accelerate; print('ok')"
test -f "$METADATA"
test -d "$DATASET_BASE"
test -d "$DIFFSYNTH_DIR/examples/qwen_image/model_training"
```

**Do not** start full training in preflight unless asked.

---

## 8. Phase F — Start training (only when user asks)

```bash
source configs/env.local.sh
export METADATA=$QWEN_VTON_DATA/converted_idm_synth_train_v2/metadata_train.json
export DATASET_BASE=$QWEN_VTON_DATA/converted_idm_synth_train_v2/dataset_base
export NUM_PROCESSES=8
export DS_PROFILE=zero3
export LR=1e-5
export NUM_EPOCHS=1

# Recommended: tmux
tmux new -d -s qwen_full_sft \
  "cd $REPO_ROOT && bash scripts/train_full_sft_zero3.sh 2>&1 | tee $OUTPUT_ROOT/qwen_vton_full_sft_launch.log"
```

Artifacts:

- Log: `$OUTPUT_ROOT/qwen_vton_full_sft/logs/train_full_sft.log`
- DiT ckpt: `$OUTPUT_ROOT/qwen_vton_full_sft/dit_full/epoch-*.safetensors`

After train completes:

```bash
"$ENV_DIR/bin/python" scripts/apply_full_dit_ckpt.py \
  --base-model "$MODEL_DIR" \
  --ckpt "$OUTPUT_ROOT/qwen_vton_full_sft/dit_full/epoch-0.safetensors" \
  --out-dir "$OUTPUT_ROOT/qwen_full_sft_fused"
```

### LoRA alternative

```bash
export METADATA=$QWEN_VTON_DATA/converted_idm_synth_train_v2/metadata_train.json
export DATASET_BASE=$QWEN_VTON_DATA/converted_idm_synth_train_v2/dataset_base
export NUM_PROCESSES=4
bash scripts/train_idm_lora_multigpu.sh
```

---

## 9. Final report template (print to user)

```text
ENV READY
  REPO_ROOT=...
  ENV_DIR=...
  MODEL_DIR=...
  DIFFSYNTH_DIR=...
  QWEN_VTON_DATA=...
  OUTPUT_ROOT=...
  ngpu=...
  train_rows=...
  prompt_chars≈...

START FULL SFT WITH:
  source configs/env.local.sh
  export METADATA=$QWEN_VTON_DATA/converted_idm_synth_train_v2/metadata_train.json
  export DATASET_BASE=$QWEN_VTON_DATA/converted_idm_synth_train_v2/dataset_base
  export NUM_PROCESSES=...
  export DS_PROFILE=...
  bash scripts/train_full_sft_zero3.sh
```

---

## 10. Failure cheat sheet

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: deepspeed` | `pip install deepspeed` into `ENV_DIR` |
| OOM on full SFT | `DS_PROFILE=zero2_offload`, fewer pixels, or LoRA |
| `missing dataset_base` / FileNotFound in loader | Re-run `prepare_data_from_hf.sh` / `run_convert_idm_v2.sh`; confirm VITON path |
| HF images only under `part-*` | Always use `prepare_data_from_hf.sh` (flattens) |
| `train.py` not found | Clone DiffSynth to `DIFFSYNTH_DIR` |
| Accidental DDP full train | Stop; use `train_full_sft_zero3.sh` only |
| Proxy / HF 403 | Set proxy + `HF_TOKEN` |
| epoch ckpt not loadable as MODEL_DIR | Run `apply_full_dit_ckpt.py` |

---

## 11. Key file map

| Path | Role |
|---|---|
| `scripts/lib_env.sh` | Loads `configs/env.local.sh` |
| `scripts/prepare_data_from_hf.sh` | HF download + flatten + v2 convert |
| `scripts/run_convert_idm_v2.sh` | Full v2 metadata only |
| `scripts/train_full_sft_zero3.sh` | Full DiT + Accelerate DeepSpeed |
| `scripts/train_idm_lora_multigpu.sh` | LoRA DDP |
| `scripts/apply_full_dit_ckpt.py` | DiT ckpt → full model dir |
| `configs/accelerate_zero3.yaml` | ZeRO-3 template |
| `configs/accelerate_zero2_offload.yaml` | ZeRO-2 + CPU offload |
| `prompts/outfit_v2.py` | Live garment-only prompt template |
| `docs/REPRODUCE.md` | Human reproduce guide |
| `docs/TRAINING.md` | Training notes |
| `docs/KNOWLEDGE.md` | Model / ZeRO / prompt knowledge |

---

## 12. Out of scope (unless user asks)

- Re-running IDM-VTON teacher synthesis from scratch  
- Multi-reference product views  
- Business TestSet / case02 assets (not in this git repo)  
- Changing license to commercial  

End of runbook.
