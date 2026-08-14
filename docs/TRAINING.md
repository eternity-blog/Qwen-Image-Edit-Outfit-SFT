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

## 3. 环境依赖

```bash
pip install -r requirements.txt
pip install -e $DIFFSYNTH_DIR
pip install deepspeed   # 全参
```
