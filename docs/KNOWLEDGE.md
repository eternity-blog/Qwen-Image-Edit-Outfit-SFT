# 知识详解：图像编辑模型、SFT 与多卡训练

配合本仓库实验阅读。标注 **【实验】** 的条目来自本项目实测；其余为通用原理及与本项目的对应关系。

---

## 1. 任务与设定

### 关键帧换装 vs 经典 VTON

| | 经典 VTON | 关键帧局部换装 |
|---|---|---|
| 底图 | 棚拍人 | 视频实拍帧（姿势/背景/遮挡要锁死） |
| 参考 | 常 1 张衣 | 商品图，线上可多张 |
| 指令 | 短 | 长编辑指令（locality、Logo、朝向） |
| 输出 | 单图 | 视频管线中的关键帧 |

**【实验】** 用短指令训出的 LoRA，在长指令评测下会整幕重画或崩溃 → **训练指令分布需与推理对齐**。

### SFT 的监督信号

多图条件编辑的 **flow-matching / 扩散式** 损失：给定 `prompt` + `edit_image`（源人+衣），在 latent 轨迹上靠近目标图 `image`（合成 GT）。不是自回归 next-token SFT。

---

## 2. Base Model：Qwen-Image-Edit-2511

### 2.1 结构

```text
Text Encoder (Qwen2.5-VL)  →  文本 + 条件图语义
VAE                        →  像素 ↔ latent
DiT (Transformer)          →  条件生成/编辑主干  ← LoRA / 全参主要更新这里
```

**【实验】参数量（bf16 权重累加）**

- DiT ≈ 20.43B（~38 GiB）  
- TE ≈ 8.29B（~15.5 GiB）  
- VAE ≈ 0.13B  
- 本配置 LoRA rank16 ≈ 118M  

### 2.2 扩散 / Flow Matching（与编辑模型）

经典 DDPM/DDIM：在噪声图上学习 ε 或 v。  
许多现代图像模型（含 Qwen-Image 系）采用 **Flow Matching / 整流流**：学习噪声到数据的速度场，推理逐步积分。

编辑模型额外条件包括：源图 latent、参考图（商品）、文本指令。

**【实验】** Edit-2511 训练需 `--zero_cond_t`（条件时间步处理与旧版不同）。

### 2.3 EditPlus 与 T2I：长 prompt 行为

| | T2I pipeline | EditPlus |
|---|---|---|
| tokenize | 常 `truncation=True` + max_length | processor 不截断 |
| encode | 按 `max_sequence_length` **切片** | **不切片** |
| check_inputs | >1024 报错 | 同左，仅 API 护栏 |

**【实验】** 约 2000 字长指令可全文进入模型；质量问题主因是 **短训长推的域差**，不是 1024 截断。

---

## 3. 数据：Teacher 合成与指令对齐

### 为何需要 IDM-VTON

公开 VITON 同 id 的 person/cloth 若目标仍是「原穿着图」，学的是重建而非换装。  
IDM unpaired try-on 得到 `人穿 A + 衣 B → 人穿 B`，形成真编辑 pair。

### 为何训练要用长模板

推理使用长编辑指令时，SFT 需拟合同一指令分布。本仓库：`prompts/outfit_v2.py` + `convert_idm_synth_to_qwen_edit_v2.py`。

### 许可

VITON：CC BY-NC；IDM：CC BY-NC-SA。派生数据默认非商用。

---

## 4. LoRA SFT

### 机制

对选定线性层：\(W' = W + BA\)（或带 scale），只训低秩矩阵。本项目插在 DiT 的 attention / MLP / mod 相关层。

### 显存直觉

DDP **每卡复制整模**。底座 bf16 已占约 54 GiB；LoRA 可训参数约 0.1B 量级，**梯度与 Adam 只为这部分付账**。全参则要对约 20B DiT 支付 fp32 梯度与 Adam 状态。

**【实验】LoRA 单卡静态粗算**

```text
冻结权重 bf16 ≈ 53.7 GiB
LoRA + grad + Adam ≈ 1.5–2 GiB
+ GC 后激活        ≈ 数～十几 GiB
峰值               ≈ 60–75 GiB / 80GB
```

### Fuse

将 LoRA bake 进 \(W\)，推理与底座加载方式一致。

---

## 5. 多卡并行

### DDP

每卡一份模型 + 不同 micro-batch；backward 后梯度 all-reduce。**不节省参数显存**，提高吞吐。

### ZeRO

| Stage | 分片 | 作用 |
|---|---|---|
| ZeRO-1 | 优化器状态 | 降低 Adam 占用 |
| ZeRO-2 | + 梯度 | 进一步降低 |
| **ZeRO-3** | + 参数 | 前向按层 all-gather |

全参 20B 级 DiT 用 DDP 复制时，优化状态可达上百 GiB；需 ZeRO-3 / FSDP。LoRA 阶段可训参数小，DDP 通常足够。

### ZeRO-3 与 Gradient Checkpointing

GC：少存激活，backward **重算** forward。  
ZeRO-3 下重算可能再次 gather 参数 → 通信增加，但通常仍优于存满激活。  
重算应与首次 forward 同输入同权重（Dropout 需复用 RNG）。

---

## 6. 显存公式

对 GPU 上参数量 \(P_{\mathrm{gpu}}\)、可训 \(P_t\)：

```text
权重(bf16) = P_gpu × 2
梯度(fp32) = P_t × 4
AdamW      = P_t × 8        # m, v
静态       = 权重 + 梯度 + Adam
峰值       ≈ 静态 + 激活
```

口算：bf16 下「约 2×参数量(B) ≈ GiB」。激活依赖分辨率与是否 GC，需估算或 `max_memory_allocated` 实测。

---

## 7. 编辑训练一步（流程）

1. VAE 将目标图编码为 latent  
2. 采样时刻 t，构造中间状态  
3. DiT 在文本/多图条件下预测速度或噪声  
4. 与目标做回归损失  
5. 反传更新 LoRA 或全参 DiT  

训练侧常 `cfg_scale=1`；guidance 多在推理使用。

---

## 8. 评测要点

- 主看长指令下的 locality（是否重画、朝向是否变）  
- 对照：源帧 | 参考方法 | base | SFT  
- 消融：短指令 vs 长指令，区分截断假设与域差  

---

## 9. 要点速记

1. Edit = DiT + VL + VAE；主要更新 DiT  
2. SFT 指令需与推理同分布  
3. EditPlus 长文不按 1024 切片  
4. LoRA 主要减小优化器状态，不取消整模复制  
5. 全参需要 ZeRO-3；DDP 不够  
6. GC 用计算换激活显存  
7. Teacher 合成提供真编辑对；注意 NC 许可  

工程步骤见 [TRAINING.md](TRAINING.md) / [DATA.md](DATA.md) / [REPRODUCE.md](REPRODUCE.md)。
