# 完整复现指南

从零到 **Qwen-Image-Edit-2511 换装 SFT**（LoRA 或全参 ZeRO-3）的可执行清单。  
硬件建议：全参 **≥8×80GB**；LoRA **≥1×80GB**（本仓默认 4 卡 DDP）。

相关链接：

- 代码：https://github.com/eternity-blog/Qwen-Image-Edit-Outfit-SFT  
- 合成数据：https://huggingface.co/datasets/lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling  
- 底座：[`Qwen/Qwen-Image-Edit-2511`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)  
- 训练框架：[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)

---

## 0. 你要得到什么

| 产物 | 说明 |
|---|---|
| `converted_idm_synth_train_v2/` | 全文 Outfit v2 prompt 的 DiffSynth metadata + `dataset_base` |
| LoRA ckpt / fused 模型 | `train_idm_lora_multigpu.sh` → `fuse_qwen_edit_lora.py` |
| 全参 DiT ckpt / fused 模型 | `train_full_sft_zero3.sh` → `apply_full_dit_ckpt.py` |
| 评测拼图 | `run_case02_v2_prompt_eval.sh`（需自备 TestSet） |

许可：合成数据衍生自 VITON-HD（CC BY-NC）与 IDM-VTON（CC BY-NC-SA），**仅研究/非商用**。见 [NOTICE.md](../NOTICE.md)。

---

## 1. 环境

```bash
git clone https://github.com/eternity-blog/Qwen-Image-Edit-Outfit-SFT.git
cd Qwen-Image-Edit-Outfit-SFT

cp configs/env.example.sh configs/env.local.sh
# 编辑 env.local.sh：MODEL_DIR / DIFFSYNTH_DIR / QWEN_VTON_DATA / OUTPUT_ROOT / ENV_DIR

# Python env（示例）
conda create -n qwen-image-edit python=3.11 -y
conda activate qwen-image-edit
pip install -r requirements.txt
pip install deepspeed          # 全参 ZeRO 必需；LoRA 可跳过

# DiffSynth（训练入口在其 examples/ 下）
git clone https://github.com/modelscope/DiffSynth-Studio.git "$DIFFSYNTH_DIR"
pip install -e "$DIFFSYNTH_DIR"
```

下载底座到 `MODEL_DIR`（完整含 `transformer/` `text_encoder/` `vae/` `tokenizer/` `processor/`）。

集群出公网时请自备代理（`http_proxy` / `https_proxy`）。

---

## 2. 数据

### 2.1 推荐：一条命令从公开 HF 拉齐

```bash
# 自动下载：合成集 + VITON-HD + 重建全文 v2 metadata
bash scripts/prepare_data_from_hf.sh
```

公开来源：

- 合成：`lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling`
- VITON-HD：`skush1/viton-hd`（CC BY-NC）

完成后应有：

```text
$QWEN_VTON_DATA/
  raw/viton_hd/
  synth/idm_unpaired/            # test ~2k
  synth/idm_unpaired_train/      # train ~11k（images 已扁平化）
  converted_idm_synth_train_v2/
    dataset_base/{viton_hd,idm_synth}   # symlink
    metadata_train.json                 # DiffSynth 读这个
    metadata_val.json
    stats.json
```

### 2.2 备选：本地从零合成

```bash
bash scripts/setup_idm_vton.sh
bash scripts/download_vton_datasets.sh   # 或按 DATA.md 手动放好
# 合成 train（耗时长、占 GPU）
bash scripts/run_idm_train_multigpu.sh   # 或 synthesize_unpaired_idm.py
# 只改 prompt、不重跑 teacher
bash scripts/run_convert_idm_v2.sh
```

细节见 [DATA.md](DATA.md)。

### 2.3 开训前自检

```bash
python - <<'PY'
import json
from pathlib import Path
base = Path("...")  # converted_idm_synth_train_v2
meta = json.load(open(base / "metadata_train.json"))
row = meta[0]
db = base / "dataset_base"
assert (db / row["image"]).is_file()
for p in row["edit_image"]:
    assert (db / p).is_file(), p
print("ok", len(meta), "prompt_chars", len(row["prompt"]))
PY
```

---

## 3. LoRA SFT（可选，轻量）

```bash
export METADATA=$QWEN_VTON_DATA/converted_idm_synth_train_v2/metadata_train.json
export DATASET_BASE=$QWEN_VTON_DATA/converted_idm_synth_train_v2/dataset_base
export NUM_PROCESSES=4
bash scripts/train_idm_lora_multigpu.sh

# 若脚本未自动 fuse：
python scripts/fuse_qwen_edit_lora.py \
  --base-model "$MODEL_DIR" \
  --lora-path "$OUTPUT_ROOT/qwen_vton_lora/lora_idm_train" \
  --out-dir "$OUTPUT_ROOT/qwen_idm_lora_fused"
```

要点：`--zero_cond_t`（2511 必需）、bf16、gradient checkpointing、全文 v2 指令。

---

## 4. 全参 DiT SFT（ZeRO-3）— 主路径

官方 DiffSynth 全参示例用 **Accelerate + DeepSpeed**，不是手写 `deepspeed --num_gpus`。本仓脚本与之对齐。

### 4.1 显存与卡数

| 配置 | 适用 |
|---|---|
| `DS_PROFILE=zero3`（默认） | **8×80GB**，无 CPU offload |
| `DS_PROFILE=zero2_offload` | **4×80GB** 或 ZeRO-3 OOM 时 |

只训 DiT；TE/VAE 冻结。DDP 全参不可行（单副本静态约 280+ GiB）。原理见 [TRAINING.md](TRAINING.md) / [KNOWLEDGE.md](KNOWLEDGE.md)。

### 4.2 启动

```bash
# 建议在 tmux / 作业系统里跑
export METADATA=$QWEN_VTON_DATA/converted_idm_synth_train_v2/metadata_train.json
export DATASET_BASE=$QWEN_VTON_DATA/converted_idm_synth_train_v2/dataset_base
export NUM_PROCESSES=8
export DS_PROFILE=zero3
export LR=1e-5
export NUM_EPOCHS=1
# 可选：从 LoRA fuse 后的权重起步
# export INIT_MODEL_DIR=$OUTPUT_ROOT/qwen_idm_lora_fused

bash scripts/train_full_sft_zero3.sh
```

4 卡 fallback：

```bash
export NUM_PROCESSES=4
export DS_PROFILE=zero2_offload
bash scripts/train_full_sft_zero3.sh
```

日志：`$OUTPUT_ROOT/qwen_vton_full_sft/logs/train_full_sft.log`  
权重：`$OUTPUT_ROOT/qwen_vton_full_sft/dit_full/epoch-*.safetensors`（仅 DiT）

### 4.3 合成可加载完整模型

```bash
python scripts/apply_full_dit_ckpt.py \
  --base-model "$MODEL_DIR" \
  --ckpt "$OUTPUT_ROOT/qwen_vton_full_sft/dit_full/epoch-0.safetensors" \
  --out-dir "$OUTPUT_ROOT/qwen_full_sft_fused"
```

之后评测把 `IDM_MODEL` / 模型路径指到该目录即可。

---

## 5. 评测

```bash
# 准备 TestSet（本仓不附带业务样例）
export BASE_MODEL=$MODEL_DIR
export IDM_MODEL=$OUTPUT_ROOT/qwen_full_sft_fused   # 或 LoRA fused
bash scripts/run_case02_v2_prompt_eval.sh
```

须使用 **live 全文 v2 prompt**（与训练同分布）。见 [EVAL.md](EVAL.md)、[PROMPT_V2.md](PROMPT_V2.md)。

---

## 6. 推荐复现顺序（最短路径）

1. §1 环境 + 底座 + DiffSynth + `deepspeed`  
2. §2.1 HF 数据 + VITON + `prepare_data_from_hf.sh`  
3. §4 全参 ZeRO-3（8 卡）或先 §3 LoRA 再 §4（`INIT_MODEL_DIR`=fused）  
4. `apply_full_dit_ckpt.py`  
5. §5 评测  

---

## 7. 检查清单

- [ ] `env.local.sh` 路径可用；`pip show deepspeed diffsynth`（或 editable DiffSynth）正常  
- [ ] `metadata_train.json` 样本 `image` / `edit_image` 在 `dataset_base` 下真实存在  
- [ ] prompt 为全文 v2（约 1.5k+ 字符），非旧短指令  
- [ ] 全参：`accelerate_*.yaml` 的 `num_processes` 与可见 GPU 数一致  
- [ ] 训练打开 `--zero_cond_t`  
- [ ] 评测模型与训练指令同分布  

---

## 8. 常见问题

**OOM**  
→ 换 `DS_PROFILE=zero2_offload`、减 `MAX_PIXELS`、确认未误开 DDP 全参。

**`deepspeed` import 失败**  
→ 在同一 `ENV_DIR` 内 `pip install deepspeed`。

**HF 图在 `part-*` 下找不到**  
→ 必须跑 `prepare_data_from_hf.sh`（会扁平 symlink）再 `run_convert_idm_v2.sh`。

**ckpt 不能当完整模型目录用**  
→ 全参产出是 DiT-only；用 `apply_full_dit_ckpt.py`。

**长 prompt 是否被截断**  
→ EditPlus 训练/推理路径不按 T2I 的 1024 硬截断；见 [KNOWLEDGE.md](KNOWLEDGE.md)。
