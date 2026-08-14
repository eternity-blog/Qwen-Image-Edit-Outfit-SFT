# 训练：LoRA SFT 与多卡全参 ZeRO-3

操作清单见 [REPRODUCE.md](REPRODUCE.md)；原理见 [KNOWLEDGE.md](KNOWLEDGE.md)。

---

## 1. LoRA SFT（本仓已有脚本）

```bash
source configs/env.local.sh   # 或依赖 scripts/lib_env.sh
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

ZeRO-3 把 **参数、梯度、优化器** 切到 N 卡。建议 **≥8×80GB** 先不开 offload；不够再 `optimizer/param` CPU offload。

### 2.2 推荐第一枪设定

- 可训：**仅 DiT**；TE/VAE 冻结（或预计算 embed/latent，进阶优化）  
- LR：比 LoRA 低（如 1e-5～5e-5）  
- 数据：同一套全文 v2 metadata  
- GC：开  
- 初始化：底座或 LoRA fuse 后权重  

示例配置：`configs/ds_zero3_bf16_example.json`。

### 2.3 待实现脚本

- [ ] `scripts/train_full_sft_zero3.sh`  
- [ ] 确认当前 DiffSynth 的 `--trainable_models dit`（无 LoRA）启动方式  
- [ ] ZeRO-3 ckpt 汇总为可 `diffusers` 加载的流程文档  

伪代码层面：

```bash
deepspeed --num_gpus 8 train.py \
  --trainable_models dit \
  --learning_rate 2e-5 \
  --use_gradient_checkpointing \
  --zero_cond_t \
  --deepspeed_config configs/ds_zero3_bf16_example.json \
  ...
```

### 2.4 注意点

1. DDP 与 ZeRO-3 适用场景不同  
2. 全参显存需按参数量估算后再选卡数  
3. GC 与 ZeRO-3 叠加会增加通信，但仍常优于存全量激活  

---

## 3. 环境依赖

```bash
pip install -r requirements.txt
pip install -e $DIFFSYNTH_DIR
pip install deepspeed   # 全参
```

`scripts/setup_env.sh` 可作集群装环境参考（路径请用 env.local 覆盖）。
