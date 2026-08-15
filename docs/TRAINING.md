# 训练：LoRA SFT 与多卡全参 ZeRO-3

完整逐步清单见 [REPRODUCE.md](REPRODUCE.md)；原理见 [KNOWLEDGE.md](KNOWLEDGE.md)。

---

## 1. LoRA SFT

```bash
source configs/env.local.sh
export METADATA=$QWEN_VTON_DATA/converted_idm_synth_train_v2/metadata_train.json
export DATASET_BASE=$QWEN_VTON_DATA/converted_idm_synth_train_v2/dataset_base
bash scripts/train_idm_lora_multigpu.sh
```

| 项 | 值 |
|---|---|
| 框架 | DiffSynth + Accelerate DDP |
| 可训 | DiT 上 LoRA |
| `zero_cond_t` | 开 |
| GC | 开 |
| 精度 | bf16 |
| 数据指令 | 全文 v2 |

**显存：** 底座复制 ~54GiB + LoRA/Adam ~2GiB + 激活 → 单卡约 60–75GiB（80GB 可跑）。  
**融合：** `scripts/fuse_qwen_edit_lora.py`。

---

## 2. 全参 SFT + ZeRO-3

### 2.1 为什么要上 ZeRO-3

只训 DiT 时，DDP 单副本静态大约：

```text
权重 bf16 ~54 GiB（含冻结 TE/VAE）
+ DiT grad fp32 ~76 GiB
+ Adam m+v ~152 GiB
≈ 280+ GiB   ← 一张 80GB 卡不可能；4 卡 DDP 也只是 4 份复制
```

ZeRO-3 把 **参数、梯度、优化器** 切到 N 卡。建议 **≥8×80GB** 先不开 offload；不够再用 ZeRO-2 + CPU offload。

### 2.2 启动（本仓脚本）

对齐 DiffSynth 官方 `examples/qwen_image/model_training/full/Qwen-Image-Edit-2511.sh`：  
**Accelerate + DeepSpeed config**，`--trainable_models dit`，`--zero_cond_t`。

```bash
export NUM_PROCESSES=8
export DS_PROFILE=zero3          # 或 zero2_offload
export LR=1e-5
export NUM_EPOCHS=1
# export INIT_MODEL_DIR=...      # 默认 MODEL_DIR；可改为 LoRA fused
bash scripts/train_full_sft_zero3.sh
```

配置模板：

- `configs/accelerate_zero3.yaml`
- `configs/accelerate_zero2_offload.yaml`

脚本会按 `NUM_PROCESSES` 生成实际 yaml 到 `$OUTPUT_ROOT/qwen_vton_full_sft/`。

### 2.3 推荐设定

- 可训：**仅 DiT**；TE/VAE 冻结  
- LR：1e-5～5e-5（低于 LoRA）  
- 数据：同一套全文 v2 metadata  
- GC：开  
- 初始化：底座或 LoRA fuse 后权重  

### 2.4 导出完整模型

```bash
python scripts/apply_full_dit_ckpt.py \
  --base-model "$MODEL_DIR" \
  --ckpt "$OUTPUT_ROOT/qwen_vton_full_sft/dit_full/epoch-0.safetensors" \
  --out-dir "$OUTPUT_ROOT/qwen_full_sft_fused"
```

### 2.5 注意点

1. 不要用纯 DDP 跑全参 DiT  
2. GC 与 ZeRO-3 叠加会增加通信，但通常仍优于存满激活  
3. `zero3_save_16bit_model: true` 便于汇总 16-bit 权重；最终评测仍建议走 `apply_full_dit_ckpt.py`

---

## 3. 对照实验：同数据 LoRA（`train_lora_v2_multigpu.sh`）

**为什么需要它：** 早先那版 LoRA（`train_idm_lora_multigpu.sh`）训的是 v1 metadata
（2 条英文短指令、76–223 字符），而全参 SFT 用的是 v2（1 条 1592 字符中文模板）。
评测统一用 live v2 prompt，等于只把 LoRA 放在了分布之外，因此那组数据**无法回答
「LoRA vs 全参」**。详见 [EVAL_RESULTS_20260815.md](EVAL_RESULTS_20260815.md) 第 3 节。

`scripts/train_lora_v2_multigpu.sh` 把数据固定成与全参完全相同的 v2，只留「训练方式」一个变量。

| 保持一致 | 值 |
|---|---|
| metadata / dataset_base | `converted_idm_synth_train_v2`（同一批文件） |
| epochs / dataset_repeat | 1 / 1 |
| max_pixels | 1048576 |
| gradient checkpointing | 开 |
| `zero_cond_t` | 开 |
| 精度 | bf16 |
| **NUM_PROCESSES** | **8** → 有效 batch 8，**optimizer 步数同为 1427** |

| 必然不同（方法本身决定，结论里要注明） | LoRA | 全参 |
|---|---|---|
| 可训参数 | DiT LoRA r16（约 0.118B） | 全部 DiT（20.43B） |
| 学习率 | 1e-4 | 1e-5 |
| 分布式 | DDP | DeepSpeed ZeRO-3 |

```bash
source configs/env.local.sh
bash scripts/train_lora_v2_multigpu.sh      # 约 3–4h @ 8×H100，含 fuse
```

脚本启动前会打印 `samples / prompt_chars / expected_steps` 自检；
若 `prompt_chars` 小于 1000 会告警——那说明误用了 v1 metadata，对照即失效。

**显存提示：** LoRA 走 DDP，每张卡仍要放完整的冻结底座（约 55 GiB 静态 + 激活），
所以 8×80GB 才跑得动；脚本会对空闲显存不足的卡发出警告（`MIN_FREE_MIB`，默认 70000）。

训练结束后按脚本末尾提示跑 **base / lora_v2 / full_sft 三方评测**（同 seed、同步数）。

---

## 4. 环境依赖

```bash
pip install -r requirements.txt
pip install -e $DIFFSYNTH_DIR
pip install deepspeed   # 全参
```
