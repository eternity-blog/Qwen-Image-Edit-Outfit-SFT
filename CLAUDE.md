# CLAUDE.md — 本仓库 Agent 环境重建手册

**读者：** 在一台**全新多卡 Linux 机器**上工作的 Claude Code（或其它编码 Agent）。  
**任务：** 把完整实验栈重建好，让人可以用一条脚本启动训练。  
**禁止**重新设计训练链路。优先使用本仓 `scripts/` 与 `docs/` 里已有内容。

给人看的说明见 `docs/REPRODUCE.md`。本文件是**可执行 runbook**。

---

## 0. 目标与完成标准

全部满足才算完成：

1. 仓库已 clone；存在 `configs/env.local.sh`，且脚本经 `scripts/lib_env.sh` 能 source 到它。
2. `ENV_DIR` 对应的 conda/venv 已安装：`torch`、`accelerate`、`deepspeed`、`diffusers`、`transformers`、`peft`、`safetensors`、`huggingface_hub`，以及 **editable** 安装的 DiffSynth-Studio。
3. `MODEL_DIR` 是完整的本地 `Qwen-Image-Edit-2511` 目录（含 `transformer/`、`text_encoder/`、`vae/`、`tokenizer/`、`processor/`）。
4. 存在 `QWEN_VTON_DATA/converted_idm_synth_train_v2/{metadata_train.json,dataset_base}`；metadata 里的路径在 `dataset_base` 下能落到真实文件。
5. `bash -n scripts/train_full_sft_zero3.sh` 通过；**预检**（import + 路径检查）成功。
6. 打印适合本机的完整训练命令（含 GPU 数与 `DS_PROFILE`）；**除非用户明确要求，不要启动长时间训练**。

**默认训练目标：** 全参 DiT SFT + ZeRO-3（`scripts/train_full_sft_zero3.sh`）。  
LoRA（`scripts/train_idm_lora_multigpu.sh`）为可选/轻量路径。

---

## 1. 硬性规则

- 禁止提交 `configs/env.local.sh`、`.env`、token、大权重或大数据。
- 若可用 HF 数据集（`lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling`），禁止重新跑 IDM 合成。
- 禁止用纯 DDP / 无 DeepSpeed ZeRO 的 `accelerate launch --multi_gpu` 跑全参 DiT——会 OOM。
- Qwen-Image-Edit-2511 必须始终带 `--zero_cond_t`。
- 训练入口是 DiffSynth 的  
  `$DIFFSYNTH_DIR/examples/qwen_image/model_training/train.py`，  
  由本仓 shell 脚本封装——不要自造新的 `train.py`。
- 若机器访问 HF/GitHub 需要代理，下载前先设 `http_proxy` / `https_proxy`。
- 长任务用 `tmux` / `screen` 或集群调度；不要把数小时训练挂在不稳定的 SSH 前台。
- 许可：VITON-HD / IDM-VTON 衍生数据**仅限非商用研究**（见 `NOTICE.md`）。

---

## 2. 未知信息先问用户一次

猜之前先收集：

| 变量 | 用途 |
|---|---|
| 大数据盘绝对路径（大、快） | `DATA_ROOT` / `QWEN_VTON_DATA` / `OUTPUT_ROOT` |
| GPU 数量 × 显存 | 决定 `NUM_PROCESSES` + `DS_PROFILE` |
| HF token（数据集若私有） | `HF_TOKEN` |
| 本机是否已有 VITON-HD | 有则只需要路径，跳过下载 |
| 代理 host:port（如有） | HF / pip / git |

若用户说「用这台机器默认」，用 `df -h`、`nvidia-smi -L`、`pwd` 探测，把大文件放到最大可写磁盘。

---

## 3. 阶段 A — 代码与环境

```bash
# A1. Clone
git clone https://github.com/eternity-blog/Qwen-Image-Edit-Outfit-SFT.git
cd Qwen-Image-Edit-Outfit-SFT
REPO_ROOT="$PWD"

# A2. 选定路径（改这里）
export DATA_ROOT=/CHANGE_ME/qwen_outfit_data          # 大盘
export MODEL_DIR=$DATA_ROOT/models/Qwen-Image-Edit-2511
export DIFFSYNTH_DIR=$DATA_ROOT/modules/DiffSynth-Studio
export QWEN_VTON_DATA=$DATA_ROOT/datasets/qwen_vton
export OUTPUT_ROOT=$DATA_ROOT/outputs
export ENV_DIR=/CHANGE_ME/conda/envs/qwen-image-edit  # 或 conda env 路径

mkdir -p "$DATA_ROOT" "$MODEL_DIR" "$DIFFSYNTH_DIR" "$QWEN_VTON_DATA" "$OUTPUT_ROOT"

# A3. 写 env.local.sh（已被 gitignore）
cp configs/env.example.sh configs/env.local.sh
cat > configs/env.local.sh <<EOF
export DATA_ROOT=$DATA_ROOT
export MODEL_DIR=$MODEL_DIR
export DIFFSYNTH_DIR=$DIFFSYNTH_DIR
export QWEN_VTON_DATA=$QWEN_VTON_DATA
export OUTPUT_ROOT=$OUTPUT_ROOT
export ENV_DIR=$ENV_DIR
EOF

# A4. Python 环境（conda 示例）
# conda create -p "$ENV_DIR" python=3.11 -y
# conda activate "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install -U pip
"$ENV_DIR/bin/python" -m pip install -r requirements.txt
"$ENV_DIR/bin/python" -m pip install deepspeed

# A5. DiffSynth-Studio（必需；train.py 在这里）
if [[ ! -f "$DIFFSYNTH_DIR/examples/qwen_image/model_training/train.py" ]]; then
  git clone https://github.com/modelscope/DiffSynth-Studio.git "$DIFFSYNTH_DIR"
fi
"$ENV_DIR/bin/python" -m pip install -e "$DIFFSYNTH_DIR"
```

### 校验 A

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

## 4. 阶段 B — 底座模型

把 **Qwen/Qwen-Image-Edit-2511** 下到 `MODEL_DIR`（HF CLI 或 `huggingface_hub.snapshot_download`）。离线镜像可以，但目录必须完整。

### 校验 B

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
# 推荐再看一眼：
ls "$MODEL_DIR/transformer"/diffusion_pytorch_model*.safetensors | head
ls "$MODEL_DIR/text_encoder"/model*.safetensors | head
ls "$MODEL_DIR/vae"/diffusion_pytorch_model*.safetensors | head
```

---

## 5. 阶段 C — 数据

### C1. VITON-HD（用户自备，CC BY-NC）

放置或软链，保证存在：

```text
$QWEN_VTON_DATA/raw/viton_hd/train/image/
$QWEN_VTON_DATA/raw/viton_hd/train/cloth/
# test/ 建议也有
```

**禁止**非法爬取。若缺失，停下来向用户要路径。

### C2. 从 Hugging Face 拉合成 pair

```bash
source configs/env.local.sh
export HF_TOKEN="${HF_TOKEN:-}"   # 需要时再设
# 脚本内默认仓库：
#   lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling
bash scripts/prepare_data_from_hf.sh
```

脚本会：

1. 下载 HF 数据集 → `$QWEN_VTON_DATA/from_hf`
2. 把 `images/part-*` 扁平化为 `synth/*/images/`（symlink）
3. 跑 `run_convert_idm_v2.sh` → 全文 Outfit v2 prompt + `dataset_base` 软链

若当时还没有 VITON，先补上再执行：

```bash
bash scripts/run_convert_idm_v2.sh
```

### 校验 C（必须）

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

量级预期：约 11415 条 train；prompt 约 1592 字符（全文 v2）。

---

## 6. 阶段 D — 选择训练配置

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv
```

| GPU（约 80GB 级） | 导出变量 |
|---|---|
| ≥8 | `NUM_PROCESSES=8` `DS_PROFILE=zero3` |
| 4 | `NUM_PROCESSES=4` `DS_PROFILE=zero2_offload` |
| &lt;4 | 优先走 LoRA 脚本，不要硬上全参 |

可选：从 LoRA fuse 后的目录热启动：

```bash
export INIT_MODEL_DIR=/path/to/qwen_idm_lora_fused
```

默认：`INIT_MODEL_DIR=$MODEL_DIR`。

---

## 7. 阶段 E — 预检（真开训前必做）

```bash
source configs/env.local.sh
export METADATA=$QWEN_VTON_DATA/converted_idm_synth_train_v2/metadata_train.json
export DATASET_BASE=$QWEN_VTON_DATA/converted_idm_synth_train_v2/dataset_base
export NUM_PROCESSES=8          # 按机器改
export DS_PROFILE=zero3         # 或 zero2_offload
export LR=1e-5
export NUM_EPOCHS=1

# 语法 + 训练脚本开头会检查的依赖：
bash -n scripts/train_full_sft_zero3.sh
"$ENV_DIR/bin/python" -c "import deepspeed,accelerate; print('ok')"
test -f "$METADATA"
test -d "$DATASET_BASE"
test -d "$DIFFSYNTH_DIR/examples/qwen_image/model_training"
```

预检阶段**不要**开全量训练，除非用户要求。

---

## 8. 阶段 F — 启动训练（仅当用户明确要求）

```bash
source configs/env.local.sh
export METADATA=$QWEN_VTON_DATA/converted_idm_synth_train_v2/metadata_train.json
export DATASET_BASE=$QWEN_VTON_DATA/converted_idm_synth_train_v2/dataset_base
export NUM_PROCESSES=8
export DS_PROFILE=zero3
export LR=1e-5
export NUM_EPOCHS=1

# 推荐 tmux
tmux new -d -s qwen_full_sft \
  "cd $REPO_ROOT && bash scripts/train_full_sft_zero3.sh 2>&1 | tee $OUTPUT_ROOT/qwen_vton_full_sft_launch.log"
```

产物：

- 日志：`$OUTPUT_ROOT/qwen_vton_full_sft/logs/train_full_sft.log`
- DiT ckpt：`$OUTPUT_ROOT/qwen_vton_full_sft/dit_full/epoch-*.safetensors`

训练结束后：

```bash
"$ENV_DIR/bin/python" scripts/apply_full_dit_ckpt.py \
  --base-model "$MODEL_DIR" \
  --ckpt "$OUTPUT_ROOT/qwen_vton_full_sft/dit_full/epoch-0.safetensors" \
  --out-dir "$OUTPUT_ROOT/qwen_full_sft_fused"
```

### LoRA 备选

```bash
export METADATA=$QWEN_VTON_DATA/converted_idm_synth_train_v2/metadata_train.json
export DATASET_BASE=$QWEN_VTON_DATA/converted_idm_synth_train_v2/dataset_base
export NUM_PROCESSES=4
bash scripts/train_idm_lora_multigpu.sh
```

---

## 9. 完成后向用户汇报（模板）

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

## 10. 故障速查

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: deepspeed` | 在 `ENV_DIR` 里 `pip install deepspeed` |
| 全参 OOM | `DS_PROFILE=zero2_offload`、降低分辨率，或改 LoRA |
| `missing dataset_base` / loader FileNotFound | 重跑 `prepare_data_from_hf.sh` / `run_convert_idm_v2.sh`；检查 VITON 路径 |
| HF 图只在 `part-*` 下 | 必须用 `prepare_data_from_hf.sh`（会扁平化） |
| 找不到 `train.py` | 把 DiffSynth clone 到 `DIFFSYNTH_DIR` |
| 误用 DDP 跑全参 | 立刻停；只用 `train_full_sft_zero3.sh` |
| 代理 / HF 403 | 设代理 + `HF_TOKEN` |
| epoch ckpt 不能当 MODEL_DIR 用 | 跑 `apply_full_dit_ckpt.py` |

---

## 11. 关键文件

| 路径 | 作用 |
|---|---|
| `scripts/lib_env.sh` | 加载 `configs/env.local.sh` |
| `scripts/prepare_data_from_hf.sh` | 下 HF + 扁平化 + v2 转换 |
| `scripts/run_convert_idm_v2.sh` | 只重建全文 v2 metadata |
| `scripts/train_full_sft_zero3.sh` | 全参 DiT + Accelerate DeepSpeed |
| `scripts/train_idm_lora_multigpu.sh` | LoRA DDP |
| `scripts/apply_full_dit_ckpt.py` | DiT ckpt → 完整模型目录 |
| `configs/accelerate_zero3.yaml` | ZeRO-3 模板 |
| `configs/accelerate_zero2_offload.yaml` | ZeRO-2 + CPU offload |
| `prompts/outfit_v2.py` | 线上 garment-only 提示词模板 |
| `docs/REPRODUCE.md` | 人读复现指南 |
| `docs/TRAINING.md` | 训练说明 |
| `docs/KNOWLEDGE.md` | 模型 / ZeRO / prompt 知识 |

---

## 12. 默认不做（除非用户要求）

- 从头重跑 IDM-VTON teacher 合成  
- 多参考商品图增强  
- 业务 TestSet / case02 资产（不在本 git 仓）  
- 改成可商用许可  

手册结束。
