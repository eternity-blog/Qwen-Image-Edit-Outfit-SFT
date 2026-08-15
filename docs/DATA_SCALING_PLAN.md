# 数据扩充方案（Data Scaling Plan）

> 起因：2026-08-15 完成首次 8 卡全参 SFT（见 [FULL_SFT_RUN_20260815.md](FULL_SFT_RUN_20260815.md)）后，
> 在真实业务关键帧（case02）上 base 与 LoRA 均出现整图重绘。本文把「数据到底缺什么、怎么补」写清楚。

---

## 1. 现状实测

| 维度 | 实测值 | 数据来源 |
|---|---|---|
| 训练样本 | **11 415**（+232 val，合成总数 11 647） | `converted_idm_synth_train_v2/stats.json` |
| 优化步数 | 1 427 步（1 epoch，有效 batch 8） | 训练日志 |
| **prompt 种类** | **1 条**（`prompt_chars` min=max=p50=**1592**） | `stats.json` |
| 每个人物搭配服装数 | **1.00**（最多 1） | `synth/idm_unpaired_train/manifest.jsonl` |
| VITON-HD train 素材 | 11 647 人物 × 11 647 服装 = **1.357 亿**理论组合 | `raw/viton_hd/train/{image,cloth}` |
| facing | 仅 `front` | 转换脚本固定参数 |
| category | 仅 `upper` | 同上 |
| overlay_placement | 仅 `none` | 同上 |
| 商品参考图 | 仅 1 张，无多视图 | `n_product_refs=1`, `multi_ref=false` |
| 标签来源 | IDM-VTON teacher 输出 | — |

## 2. 诊断：问题不是「量少」，是「多样性坍缩到 1」

20.43B 全参配 1.1 万样本，量确实小；但更致命的是**每个语义维度都只有一个取值**：

1. **指令分布不相交。** 训练集只存在 1592 字符这一条 prompt；线上 case02 实测是 **2000 / 1922** 字符的不同模板。
   模型没见过任何指令变化，遇到新模板即退化为「按描述生成新图」。
2. **域不相交。** VITON 是正面站姿 + 干净背景 + 上半身平铺衣服；业务帧是运动姿态 + 复杂背景 + 遮挡 + 字幕。
3. **能力覆盖缺口。** 下装 / 连衣裙、非正面、花字与字幕、多参考图融合，训练集里**完全没有**监督信号。
4. **天花板 = teacher。** 标签是 IDM-VTON 输出，teacher 的瑕疵会被一并学进去。

与此一致的两个观测：训练 loss 窗口均值只降 **23.7%**（几乎没动）；case02 上完全失效。

**结论：优先补「多样性」和「目标域」，而不是把 VITON 单一组合从 1 万堆到 10 万。**

---

## 3. 扩充方案（按性价比排序）

### 方案 1 — Prompt 表层增广（零新增图片，优先级最高）

**做什么：** 让同一批图片配上**语义等价但表层不同**的指令。

`prompts/outfit_v2.py` 的 `build_train_v2_prompt` 已经接受
`facing / category / overlay_placement / n_product_refs / is_tail`，
但 `run_convert_idm_v2.sh` 把它们钉死成单一组合。改为按样本采样。

安全的增广轴（不改变「正确答案」，因此不会教坏模型）：

| 增广方式 | 说明 |
|---|---|
| 段落顺序置换 | 输入说明 / 约束 / 输出要求等区块顺序随机 |
| 同义改写 | 「只允许替换目标服装」↔「除目标服装外不得改动」 |
| 冗长度变化 | 生成 800 / 1600 / 2500 字符三档，覆盖线上 1.9k–2k 区间 |
| 中英混排比例 | 线上模板含中英混排，随机调整比例 |
| 约束子集抽样 | 随机保留 60–100% 的非关键约束条目 |

**必须遵守的一条铁律：增广后的指令必须与目标图一致。**
例如目标图没有字幕，就不能把 `overlay_placement` 改成 `bottom`——那会训练模型**忽略**字幕指令。
同理 VITON 全是正面图，`facing` 不能随机成 `back`。

**产出预期：** 指令种类 1 → 数十至上百；图片零新增；只需重跑一次 metadata 转换（分钟级）。

**验收：** `stats.json` 里 `prompt_chars_min != prompt_chars_max`，且 p50 落在 1.6k–2.0k。

---

### 方案 2 — 同素材扩 pair（低成本，已实施 batch2）

VITON-HD 自带的 `train_pairs.txt` 只给了**一种**固定配对，batch1 正是它，所以每个人物只出现一次。
扩量 = 生成新的配对排列。

| k | 累计 pairs | 新增 | 2 卡耗时 | 8 卡耗时 |
|---|---|---|---|---|
| 1（batch1） | 11 647 | — | — | — |
| 2（batch2） | 23 294 | 11 647 | 6.3 h | 1.6 h |
| 3 | 34 941 | 23 294 | 12.7 h | 3.2 h |
| 5 | 58 235 | 46 588 | 25.3 h | 6.3 h |

**实测吞吐：920 张/小时/卡**（batch1 4 卡 3.17h 反推，batch2 2 卡实测复现同一数字）。
单图约 51.7 KB，磁盘可忽略。

**组合上限 vs 多样性上限（重要）：** 排除同 id 自配后组合数约 1.356 亿，看似无限；
但视觉素材始终是那 11 647 个人 + 11 647 件衣服，扩 k 只是重新排列，
**不新增任何人物或服装本体**，收益递减。真正增加视觉多样性要靠方案 3（DressCode）与方案 4。

#### 批次组织

```text
$QWEN_VTON_DATA/synth/
├── idm_unpaired_train/          # batch1
├── idm_unpaired_train_b2/       # batch2
│   ├── images/
│   ├── manifest.jsonl
│   ├── pairs_b2.txt
│   └── batch_meta_b2.json       # seed / 去重统计 / 保证项
└── ...
```

上传到**同一个 HF 仓库**并列存放，沿用 `images/part-XXXX/` 分片（HF 单目录 ≤1 万文件）。

```bash
# 1) 生成配对：自动排除同 id 自配，并与所有既往批次去重（写盘前断言校验）
python scripts/make_pair_batch.py \
  --viton-root $QWEN_VTON_DATA/raw/viton_hd \
  --prev $QWEN_VTON_DATA/synth/idm_unpaired_train \
  --out $QWEN_VTON_DATA/synth/idm_unpaired_train_b2/pairs_b2.txt \
  --batch-id b2 --seed 2

# 2) 多卡合成（可断点续跑，已存在的图会跳过）
BATCH_ID=b2 GPU_LIST=0,5 bash scripts/run_idm_synth_batch.sh

# 3) 合并各 shard 的 manifest
bash scripts/merge_idm_shard_manifests.sh $QWEN_VTON_DATA/synth/idm_unpaired_train_b2
```

再加批次时把每个既往批次都用 `--prev` 传进去，即可保证全局零重复。

---

### 方案 3 — 接入 DressCode（补类目缺口）

仓库已有 `scripts/download_vton_datasets.sh` 与 `scripts/extract_dresscode_zip.py`（约 72GB 分卷）。
DressCode 含 **upper / lower / dresses** 三类，直接解决「只有上装」的覆盖缺口，
并让 `category` 这个 prompt 轴变成**有真实标签支撑**的可增广维度。

**顺带收益：** 电商类素材更可能存在同款多视图 → 可能解开 TODO 里挂着的多参考增强。

---

### 方案 4 — 真实业务帧 + GPT 作第二 teacher（最关键）

这是当前**最短的板**，也是 case02 失效的直接原因：训练集里没有一张真实视频关键帧。

**关键观察：** 2026-08-15 的 case02 对照图显示，**GPT Image 2 在真实帧上是成功的**——
它正确保留了构图、姿态、走廊背景与字幕，只替换了服装。
也就是说，**GPT 在真实帧上的输出，正是我们缺的那部分监督信号**。

**做法：**

1. 从 `kling-aigc-engine` 的历史任务里收集 (真实关键帧, 商品图集, 线上 v2 指令, GPT 产出) 四元组
2. 用与线上完全一致的 prompt（而不是重建的模板）作为训练指令 —— 天然解决方案 1 想模拟的分布问题
3. 人工或规则过滤：剔除 GPT 自身失败样本（构图漂移、Logo 错误、字幕丢失）
4. 与 VITON 合成数据**混采**（建议先 1:3 真实:合成，再按验证结果调）

**注意点：**

- 这条路线的天花板变成 GPT，而项目目标本就是「在自研骨干上逼近 GPT」，因此是合理的
- 需确认 GPT 产出用于训练自研模型在**许可与合规**上没有问题（外部 API 产出物的使用条款）
- 真实帧含真人肖像，注意授权范围

---

### 方案 5 — 多参考图（解锁线上真实入参）

线上一次最多传 3 张商品参考图（不同视角）。当前训练全是单张，模型没有多图融合能力。
VITON 无同款多视图，故此项一直 deferred；DressCode 或电商商品图（主图 + 细节图 + 平铺图）可解。

---

## 4. 反面清单（不要这么扩）

| 做法 | 为什么不行 |
|---|---|
| 调大 `dataset_repeat` / 多跑 epoch | prompt 完全相同时，重复只加速记忆化，不改善泛化 |
| 无脑把 VITON 堆到 10 万条 | 域和指令仍单一，只是把同一个分布看更多遍 |
| 语义轴随机化但不改目标图 | 会训练模型**忽略指令**（如说要字幕却没字幕） |
| 同 id person/cloth 自配 | 退化为图像重建，学不到换装 |
| 把 test split 也并入训练 | 失去留出集，后续无法判断是否真的学到了 |

---

## 5. 建议执行顺序

| 阶段 | 内容 | 成本 | 预期效果 |
|---|---|---|---|
| P0 | 方案 1 prompt 表层增广 + 重训 | 分钟级转换 + 一次训练 | 直接检验「指令单一」是否为 case02 失效主因 |
| P1 | 方案 4 真实帧 + GPT teacher（先几百条打通链路） | 数据收集为主 | 补目标域，预期对 case02 提升最大 |
| P2 | 方案 2 扩到 k=3~5 | teacher GPU 时间 | 提升换装保真度上限 |
| P3 | 方案 3 DressCode + 方案 5 多参考 | 下载 + 合成 | 补类目与多图融合 |

算力不是瓶颈：6 万 pairs 跑 1 epoch ≈ 7 500 步 × 5.87 s ≈ **12 小时**（8×H100，ZeRO-3）。

---

## 6. 评测协议（每轮扩充都要跑同一套）

只看训练 loss 会误判（见 FULL_SFT_RUN 第 7.3 节），必须双轨评测：

| 轨道 | 脚本 | 素材 | 回答什么问题 |
|---|---|---|---|
| 域内留出 | `scripts/eval_viton_holdout.py` | VITON-HD **test**（训练未见） | 有没有学到换装能力本身 |
| 目标域 | `scripts/run_case02_v2_prompt_eval.sh` | case02 业务帧 + GPT 参照 | 在真实关键帧上能不能用 |

固定 `--seed 0 --steps 40`，并保留 base 与上一轮模型作对照，指标与拼图一并归档。
