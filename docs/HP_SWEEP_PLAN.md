# 超参数对比实验方案（HP Sweep Plan）

> 起因：2026-08-15 已完成全参 SFT（LR=1e-5, 1 epoch）并在 [EVAL_RESULTS_20260815.md](EVAL_RESULTS_20260815.md) 留下 base-vs-full_sft 双轨评测；LoRA v2（rank16, LR=1e-4）也跑完但尚未评测。本文设计**同一方案内的单变量超参扫描**——一次只动一个旋钮，其余钉死在基线，用同一套 held-out 评测判断每个配置是否优于基线。
>
> **不是** SFT 与 LoRA 之间的对比：两者同时变了 LR / 可训参数 / 更新方向 / 分布式策略，受多变量混淆（EVAL_RESULTS 第 3 节已论证不可直接比较）。本文只在**各自方案内部**做单变量消融。
>
> **本版按反馈精简过**：全参 Tier 2（epochs/wd）降级为条件性、Tier 3（batch）降级为理解性；LoRA Tier 3 中 lora_scale 因与 LR 数学冗余直接删除，2-epoch 降级、target 消融重分类。主路径只剩 LR 与 rank 两类扫参（见第 7 节）。

---

## 1. 基线（每个新 run 必须超过的标尺）

### 1.1 全参 SFT 基线（已有评测，直接当 LR-sweep 的基线点复用）

| 维度 | 值 | 来源 |
|---|---|---|
| LR / epochs / wd | 1e-5 / 1 / 0.01 | training_args.json |
| 有效 batch / 步数 | 8 / 1427 | launch log + TB |
| 训练 loss 末 100 步 | 0.03118（首 100 步 0.04085，↓23.7%） | TB 事件 |
| 峰值显存 | 74.8 GB（91.7%） | nvidia-smi |
| LR 调度器 | ConstantLR（runner.py:66，不绑 epoch） | 源码 |

**评测基线**（EVAL_RESULTS_20260815.md，`seed=0 / steps=40`）：

| 轨道 | 指标 | base | **full_sft（基线）** | 标尺 |
|---|---|---|---|---|
| 域内 VITON-HD 留出（6 样本，768×1024，prompt 1592） | MAD vs 人物 | 35.65 | **20.04** | 18.64（teacher） |
| | MAD vs teacher | 28.94 | **9.31** | 0 |
| | hist corr vs teacher | 0.8036 | **0.9380** | 1.0 |
| 业务域 case02（2 shot，720×1280，prompt 2000/1922） | MAD shot00 | 72.77 | **32.16** | 48.04（GPT Image2） |
| | MAD shot01 | 86.77 | **26.22** | 37.28 |

> 每个 SFT 新配置要在**同一评测**上跑出数字，与上表 full_sft 行正面对比。**超过基线才算该 HP 更优**；不如基线则该 HP 设置更差。

### 1.2 LoRA v2 基线（尚无评测 → 第一步先补评测）

| 维度 | 值 |
|---|---|
| rank / LR / epochs | 16 / 1e-4 / 1 |
| 可训参数 / 步数 | 118M / 1427 |
| 训练 loss 末 100 步 | 0.04081（首 100 步 0.04160，**仅 ↓1.9%**，几乎平台） |

**评测基线：缺。** LoRA sweep 的第一个动作不是扫参，而是**把现有 rank16/LR1e-4 配置送进三方评测**（`eval_viton_holdout.py` + `run_case02_v2_prompt_eval.sh`，同 seed/steps），拿到它自己的 in-domain / case02 数字，作为 LoRA sweep 的 "to beat"。没有这个锚点，LoRA sweep 无法判断好坏。尤其 LoRA loss 只降 1.9%，**LR sweep（第 4.2 节）正是为了回答"是 LR 偏低还是方法本身平台化"**——没有评测锚点这道题没法判。

---

## 2. 前提（不满足则 sweep 无意义）

1. **先评测再扫参。** 全参已有基线评测可直接用；LoRA 必须先补基线评测。无评测 = 盲调。
2. **多 seed 评测，否则比的是噪声。** 扩散单步噪声 ≫ 趋势：全参末 100 步单步 stdev 0.025、极差 0.10；LoRA 末 100 步 stdev 0.033、极差 0.12。单 seed 的指标差很可能落在噪声带内。**每个配置至少 3 seed（或 6 样本取 mean±std），只有差值超出噪声带才判"更好"。**
3. **一次一变量。** 每个新 run 只改一个旋钮，其余钉死基线，否则归因不清。
4. **同评测协议。** 固定 `--seed 0 --steps 40`，同 6 留出 + 同 2 case02 shot，每个新 run 带上 base 作对照。
5. **隔离输出目录。** 每 run 用 `FULL_SFT_OUT=` / `LORA_V2_OUT=` 独立目录，不覆盖基线 ckpt（否则后跑的覆盖先跑的，无法回评）。

---

## 3. 全参 SFT 单变量扫描

> 每 run ~2.5h（1427 步 × 5.87 s/it）+ 评测 ~45min。**标 ✓ 的配置已跑过，直接当基线点复用，不重跑。**

### 3.1 Tier 1 — learning_rate（最高优先级）

| 配置 | LR | 固定其余 | 目的 | 状态 |
|---|---|---|---|---|
| A0 | **1e-5** | epochs=1, wd=0.01, bs=8 | 基线点 | ✓ 已跑 + 已评 |
| A1 | 5e-6 | 同上 | 是否欠拟合（loss 降不够） | 新 run |
| A2 | 2e-5 | 同上 | 是否更快收敛 / 过拟合 | 新 run |
| A3 | 5e-5 | 同上 | 是否毁预训练表示（灾难遗忘） | 新 run |

**判据：** A1–A3 的域内 MAD-teacher 是否 < 9.31、hist-corr > 0.9380，且 case02 MAD < 32.16/26.22（超基线且超噪声带）。A3 预期 case02 崩（颜色 / 身份漂移加剧）——若如此，正是"LR 过高毁表示"的实锤。

### 3.2 条件性 — num_epochs & weight_decay（已从主路径降级）

这两项都**不进主扫参面**，理由是它们都是**二阶旋钮、动不了真正瓶颈**：

- **B1（2 epoch）**：`FULL_SFT_RUN` 第 9 节已从理论上判过——ConstantLR 不绑 epoch，2-epoch 是干净平坦延伸，**无 schedule 副作用**；但参数/数据比极端（20B/11.4k ≈ 1.8M 参数/样本），第 2 遍把同一批唯一图再喂一次主要加 memorization 风险，且 LoRA loss 已平台化暗示全参 2-epoch 大概率也平。**已有强先验说它会 hurt**，再跑只是经验确认本对话开头那个"2 epoch 会不会更好"的问题。→ 仅在你想要经验收口时跑，不排期。
- **C1/C2（weight_decay 0 / 0.1）**：wd 在 1-epoch 扩散 SFT 上效应通常很小；更要命的是它**动不了 case02**——case02 崩是域差 / 多样性坍缩，不是正则不够。→ 改为**条件性**：只有当 P1 评测真的在 case02 或域内上看到过拟合膝点（train 继续降、eval 反降）时，才回头扫 wd 验证"正则能否压住膝点"。没看到就跳过。

| 配置 | 变量 | 值 | 何时做 | 状态 |
|---|---|---|---|---|
| B1 | epochs | 2 | 仅当想经验收口"2-epoch"问题 | 条件性 |
| C1 | weight_decay | 0 | 仅当评测出现明显过拟合膝点 | 条件性 |
| C2 | weight_decay | 0.1 | 同上 | 条件性 |

> 一句话：过拟合的解法是**加数据**（见 [DATA_SCALING_PLAN](DATA_SCALING_PLAN)），不是加 wd 也不是加 epoch。

### 3.3 理解性 — effective batch（不进主路径，且与 LR 混淆）

**这个旋钮的作用（你问的"是什么作用"）：**

有效 batch = num_gpu × micro_batch × grad_accum，本配置 = 8 × 1 × grad_accum。它调节的是**梯度的信噪比**：

1. **大 batch**：梯度在更多样本上平均 → 方差更低、方向估计更干净、优化更稳；但每步更慢（要攒更多样本）。
2. **小 batch**：梯度噪声大 → 能逃尖锐极小、有正则化效果；但也可能抖动、难收敛。
3. **关键张力（在本数据集上）**：固定 11.4k / 1-epoch 时，**翻倍有效 batch 直接砍半 optimizer 步数**（bs16→713 步，bs32→476 步）。你是在用"更新次数"换"每步质量"——不保证赚，476 步更新极少、很可能欠拟合。
4. **致命的 LR 混淆**：最优 LR 随 batch 近似线性缩放（linear scaling rule）。**固定 LR 改 batch = 两个变量一起动**，你分不清结果是 batch 的功劳还是 LR 现在失配了。要干净隔离 batch 这个变量，必须连 LR 一起按比例缩放——那又不再是单变量实验。这是 batch 扫最难做干净的地方。
5. **显存**：grad_accum 不改峰值显存（仍是一次 micro-batch 的 forward/backward），是"在不加显存的前提下模拟大 batch"的手段；这里 micro-batch 已是 1，grad_accum 是唯一能加大 batch 的杠杆。

| 配置 | grad_accum | 有效 batch | 步数 | 目的 | 状态 |
|---|---|---|---|---|---|
| D0 | 1 | 8 | 1427 | 基线点 | ✓ |
| D1 | 2 | 16 | 713 | 梯度更稳 vs 更少更新 | 理解性，可选 |
| D2 | 4 | 32 | 476 | 同上极限（欠拟合风险高） | 理解性，可选 |

**结论：** batch 是真旋钮，但**二阶（次于数据瓶颈）、与 LR 混淆（要干净就得加变量）、在 11.4k 上还欠拟合风险**。只在你确实想理解"梯度质量 vs 更新数谁更是瓶颈"时才跑，且跑时务必注明"LR 未随 batch 缩放，结论含混淆"。不进主路径。

---

## 4. LoRA v2 单变量扫描

> 每 run ~2h（1427 步 × 4.60 s/it）+ fuse 1.5min + 评测 ~45min。**rank16/LR1e-4 已跑过，复用为基线点。**

### 4.1 Tier 1 — lora_rank

| 配置 | rank | 固定其余 | 目的 | 状态 |
|---|---|---|---|---|
| E0 | **16** | LR=1e-4, ep=1, scale=1.0 | 基线点 | ✓ 已跑（待评） |
| E1 | 8 | 同上 | 容量不足？欠拟合？ | 新 run |
| E2 | 32 | 同上 | 容量↑，过拟合？ | 新 run |
| E3 | 64 | 同上 | 极限容量，记忆化？ | 新 run |

**判据：** 与 LoRA 基线评测（补评后）正面对比。rank 越大可训参数越多（64 ≈ 4×118M），对 20B/11.4k 更易记忆化。

### 4.2 Tier 2 — learning_rate（LoRA loss 仅降 1.9%，这组最该跑）

| 配置 | LR | 固定其余 | 目的 | 状态 |
|---|---|---|---|---|
| F0 | **1e-4** | rank=最优E, ep=1 | 基线点 | ✓（待评） |
| F1 | 5e-5 | 同上 | LoRA loss 平台是否因 LR 偏低 | 新 run |
| F2 | 2e-4 | 同上 | 更大增量 | 新 run |
| F3 | 5e-4 | 同上 | 过大→不稳 / 毁 | 新 run |

> 基线 loss 仅降 1.9%，可能 LR 偏低（增量起不来）也可能方法本身平台化。F1/F2 能区分这两者——这是 LoRA sweep 里信息量最大的一组，必做。

### 4.3 不进主路径 — lora_scale / 2-epoch / target 消融（三个理由各不同）

- **lora_scale（α/r）：直接删除。** 数学上 `W' = W + (α/r)·BA`，训练时灌进 BA 的梯度随 α/r 缩放——**改 scale 等价于改 LR**，是同一自由度的重参数化。同时扫 scale 和 LR = 重新发现同一条曲线，纯冗余。已被第 4.2 节 LR sweep（F1–F3）覆盖，不单列。
- **H1（LoRA 2-epoch）：降级为条件性。** 比 full-param 2-epoch 风险低（LoRA 只更新 118M 低秩，记忆化容量远小于 20B），但 loss 已平台（↓1.9%），大概率也平。→ 同全参 B1，仅在你想要经验确认时跑。
- **I1（target 消融：attn-only vs 基线 13 模块）：重分类为结构性消融，不是 HP。** 它回答的是"换装信号需要改 DiT 的哪些层"——attn-only 是经典 LoRA 目标，加 MLP/mod（`img_mlp/img_mod/txt_mlp/txt_mod`）是个选择。三个里它最有趣（真·理解性问题），但不在"找最优 config"的路径上，不混进 HP 表。挪到下方脚注。

| 配置 | 变量 | 值 | 何时做 | 状态 |
|---|---|---|---|---|
| H1 | epochs | 2 | 仅当想经验确认 LoRA 2-epoch | 条件性 |
| I1 | target modules | attn-only | 仅当想理解"哪些层吃换装信号" | 结构性消融 |

> **结构性消融脚注（I1，可选）：** 把 LoRA 只插在 attention 投影（`to_q/k/v, add_q/k/v_proj, to_out.0, to_add_out`），去掉 MLP/mod 层，与基线 13 模块对比。若 attn-only 掉点明显，说明双塔的 mod/mlp 层对换装（图像条件注入）重要；若差不多，说明 attention 已足够承载换装信号，MLP/mod 是冗余开销。这是**理解模型**的实验，不优化 config，只在有余力时做。

---

## 5. 评测协议（每个 run 都跑同一套）

| 步 | 脚本 | 素材 | 回答 |
|---|---|---|---|
| 1 域内留出 | `eval_viton_holdout.py` | VITON-HD test（训练未见）6 样本 | 该 HP 下换装能力本身 |
| 2 业务域 | `run_case02_v2_prompt_eval.sh` | case02 2 shot + GPT 参照 | 该 HP 下真实帧能不能用 |
| 3 可视化 | `visualize_metrics.py` | 评测目录 | 差异分布在哪 |

固定 `--seed 0 --steps 40`，每个配置带 base 作对照，**并至少 3 seed 取 mean±std**。归档指标 + 拼图。

**"更好"的定义：** 域内 MAD-teacher 与 case02 MAD **同时**优于基线且超出噪声带（mean 差 > 各自 seed 间 stdev）。只一项好不算——出现过"域内↑但业务域↓"的过拟合。

---

## 6. 反面清单（不要这么扫）

| 做法 | 为什么不行 |
|---|---|
| 跨方法比（SFT 最优 LR vs LoRA 最优 rank） | 仍受 LR/参数/更新方向多变量混淆，不是单变量 |
| 单 seed 下结论 | 扩散噪声 ≫ HP 间差异，单点比的是运气 |
| 没补 LoRA 基线评测就扫 LoRA | 无锚点，判不了好坏 |
| 同时变 LR 和 epochs | 两个变量动，归因不清 |
| **同时扫 lora_scale 与 LR** | 两者是同一自由度的重参数化（`W'=W+(α/r)·BA`，梯度随 α/r 缩放），扫两个 = 重复发现同一条曲线 |
| **固定 LR 改 batch 当单变量** | 最优 LR 随 batch 线性缩放，固定 LR 改 batch = LR 失配混淆，分不清是 batch 还是 LR 的锅 |
| 在 case02 上指望 HP sweep 修域差 | HP 调的是域内拟合上限；case02 崩是数据/域差，只有加数据能修（见 [DATA_SCALING_PLAN](DATA_SCALING_PLAN)） |
| 把 train loss 当判据 | 已论证 train loss 判不了（FULL_SFT_RUN 7.3） |

---

## 7. 执行顺序与成本（按反馈精简后）

**主路径（核心，~25h 含评测）：**

| 阶段 | 内容 | 新 run | 成本（8×H100） | 状态 |
|---|---|---|---|---|
| P0 | LoRA 基线补评测 + 全参基线复用 | 0 | ~1h 评测 | 必做（前置） |
| P1 | 全参 LR sweep（A1,A2,A3）+ 评测 | 3 | ~9h | 核心 |
| P2 | LoRA rank sweep（E1,E2,E3）+ 评测 | 3 | ~7.5h | 核心 |
| P4 | LoRA LR sweep（F1,F2,F3）+ 评测 | 3 | ~7.5h | 核心（LoRA loss 仅降 1.9%，必查 LR 是否偏低） |

回答"全参 / LoRA 各自最优 LR 与 rank 在哪"。P1 定全参最优 LR 后，P2/P4 的 rank/LR 扫在该结论上择优收敛。

**非主路径（条件性 / 理解性，不默认排期）：**

| 阶段 | 内容 | 何时做 |
|---|---|---|
| P3 | 全参 epochs/wd（B1,C1,C2） | 仅当 P1 评测出现明显过拟合膝点，或想经验收口"2-epoch"问题时 |
| P5 | 全参 batch（D1,D2）+ LoRA 2-epoch（H1）+ target 消融（I1） | 仅当想理解"梯度质量 vs 更新数"或"哪些层吃换装信号"时；batch 须注明 LR 未随缩放、含混淆 |

P3/P5 不在主路径：前者受数据瓶颈主导、非 HP 能解；后者要么与 LR 混淆、要么已分析过、要么是理解性而非优化性。**要扩 ROI，先扩数据（见 [DATA_SCALING_PLAN](DATA_SCALING_PLAN)），不要先扩 HP 扫参面。**

---

## 8. 与数据扩充的关系（设定期望）

HP sweep 优化的是**域内拟合上限**与**方法本身的样本效率**。但 EVAL_RESULTS 已显示 case02 崩的病因是**域差 / 数据多样性坍缩**（1 条 prompt、仅正面上装、无真实帧），不是 HP。所以：

- HP sweep **可能**让域内 MAD-teacher 从 9.31 再降一点、case02 略改善；
- HP sweep **不会**修掉 case02 的颜色错 / 字幕崩 / 身份漂移——那需要 DATA_SCALING_PLAN 的真实帧 + GPT teacher + DressCode 多类目；
- 正确排序：**先 HP sweep 把现有数据榨到最优（定基线 config），再加数据看增益归因到数据而非 HP。**

---

## 9. 复现命令（全参 LR sweep 示例）

```bash
source configs/env.local.sh
export METADATA=$QWEN_VTON_DATA/converted_idm_synth_train_v2/metadata_train.json
export DATASET_BASE=$QWEN_VTON_DATA/converted_idm_synth_train_v2/dataset_base
export NUM_PROCESSES=8 DS_PROFILE=zero3 NUM_EPOCHS=1

# A1: LR=5e-6（隔离输出目录，不覆盖基线 ckpt）
FULL_SFT_OUT=$OUTPUT_ROOT/hp_sweep/A1_lr5e6 LR=5e-6 bash scripts/train_full_sft_zero3.sh
# fuse（A1 的 ckpt 在它自己的目录下）
"$ENV_DIR/bin/python" scripts/apply_full_dit_ckpt.py \
  --base-model "$MODEL_DIR" \
  --ckpt "$OUTPUT_ROOT/hp_sweep/A1_lr5e6/dit_full/epoch-0.safetensors" \
  --out-dir "$OUTPUT_ROOT/hp_sweep/A1_lr5e6_fused"
# 评测（3 seed，带 base + 基线对照）
for s in 0 1 2; do
  "$ENV_DIR/bin/python" scripts/eval_viton_holdout.py \
    --model base="$MODEL_DIR" \
    --model lr5e6="$OUTPUT_ROOT/hp_sweep/A1_lr5e6_fused" \
    --model baseline="$OUTPUT_ROOT/qwen_full_sft_fused" \
    --out-dir "$OUTPUT_ROOT/hp_sweep/A1_eval_s$s" --n 6 --steps 40 --seed $s
done
```

LoRA sweep 同理：`LORA_V2_OUT=$OUTPUT_ROOT/hp_sweep/E1_rank8 LORA_RANK=8 bash scripts/train_lora_v2_multigpu.sh`（脚本自带 fuse 到该目录的 `fused/`）。**注意：不要单独扫 `LORA_SCALE`——它与 LR 数学等价，已被 LR sweep 覆盖。**

---

## 10. 诚实口径

- 本方案主张：**以 held-out 评测为判据、一次一变量、多 seed、复用已跑 run 作基线点**的方法论。
- **不主张**任何具体 HP 在跑出来之前更优；A3 / F3 等极端值预期会劣化，但需实锤。
- 主动降级 / 删除的项各有诚实理由：epochs/wd 是二阶且动不了域差；batch 与 LR 混淆且在 11.4k 上欠拟合；lora_scale 与 LR 数学冗余；target 消融是理解性非优化性。**把这些"不扫"的判断写进简历，本身比"扫了一堆"更能体现理解深度。**
- SFT sweep 与 LoRA sweep 是**两套独立实验、两个独立基线**，不跨方法比绝对值；"两方法谁更优"需各自最优 config 出来后，在承认 LR 不可拉平的前提下比较评测——即便那样仍是方法整体对比而非单变量消融。
