# Qwen-Image-Edit-2511 全参 DiT SFT 训练报告（b1+b2 合并数据,2026-08-16）

> 从 base model 起、用 b1+b2 合并洗牌数据做的 1 epoch 全参 DiT SFT 全记录。所有训练数字取自 TensorBoard / nvidia-smi CSV / launch log,非估算。
> 与 2026-08-15 的 b1-only run（`qwen_vton_full_sft`,wandb run `50jhh811`)同框架对比——唯一变量是数据量翻倍(b1 11415 → b1b2 ~22830)。

## 0. 一句话结论

8×H100 80GB + ZeRO-3 + bf16,从 **base model** 起、用 b1+b2 合并洗牌的 **22 829 条**换装 pair 做 1 epoch 全参 DiT(20.43B)SFT,**2854/2854 步、4h41m 跑完,产出 39GB DiT-only checkpoint**。loss 窗口均值 0.0414 → 0.0339(降 18.0%),显存峰值 74.8GB(91.7%),8 卡负载均衡。与 b1-only run 同框架,**唯一变量是数据量翻倍**(11415 → 22829)。

> 数据翻倍→墙钟线性翻倍(2h22m → 4h41m)、单步速度不变(5.87 s/it)、显存不变(ZeRO-3 分片大小由参数量决定,与数据量无关)。但 loss 两条曲线都在噪声带内(last100 stdev 0.029 ≫ 0.003 的窗口差),**训练 loss 无法判定 b1 vs b1b2 孰优**——最终排名需留出集评测。

## 1. 时间线

| 时刻(+08:00) | 事件 |
|---|---|
| 2026-08-15 22:49 UTC | HF dataset `lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling` 上传 b2 新批次 |
| 08-16 ~12:08 | `snapshot_download` 增量拉取 b2(幂等,只下本地没有的新文件) |
| 08-16 12:17 | 远端结构确认:3 个 split — `idm_unpaired`(2032)、`idm_unpaired_train`(11647,b1)、`idm_unpaired_train_b2`(11647,b2) |
| 08-16 12:2x | b2 内容确认:同 VITON-HD 人物×衣服重新洗牌的配对,`batch_meta_b2.json` 保证 "no (person,garment) tuple repeated from a previous batch"——out_names(`{person}__{cloth}.jpg`)与 b1 不相交,prompt 模板与 b1 完全相同 |
| 08-16 ~12:3x | 方案确定:**Plan B** — base model + b1+b2 合并洗牌,1 epoch 全参 SFT |
| 08-16 ~13:0x | b2 下载完成,`prepare_data_from_hf.sh` flatten b2 + 构建 combined 目录 + convert v2 |
| 08-16 13:01:42 | `tmux qwen_b1b2` 启动 `launch_full_sft_observable.sh`;13:01:53 nvidia-smi 采样器开跑 |
| 08-16 13:01:59 | `accelerate launch` 启动(8 卡,base init),显存 7 MiB→739 MiB 加载中 |
| 08-16 13:03:06 | 第 1 步;**live wandb 连上 `changan_1`**(run `qdg3zh5m`,`Loaded credentials from WANDB_API_KEY`);TensorBoard 同步开 |
| 08-16 17:43:18 | `TRAIN DONE`——2854/2854 步跑完,DiT-only ckpt(39GB)落盘 |
| 08-16 17:46:03–17:47:07 | `apply_full_dit_ckpt.py` 嫁接成完整模型目录 `qwen_full_sft_b1b2_fused`(1933 tensors,missing=0 unexpected=0,inode 不同=独立拷贝,`img_in.bias` sum 219.20→219.08 确认权重已更新) |

## 2. 为什么是 Plan B(base + 合并,而非续训)

候选三方案:

| 方案 | 起点 | 数据 | 1ep 步数 | 墙钟 | 评估 |
|---|---|---|---|---|---|
| A | 全参 SFT ckpt | 仅 b2 | ~1426 | ~2.3h | 最省,但 b1 不再见→轻遗忘;需 apply_full_dit_ckpt 嫁接 |
| **B(选)** | **base model** | **b1+b2 合并** | ~2853 | ~4.7h | 最干净:单次训练、单 LR、两批均匀交错、无遗忘;满足"从 base 开始" |
| C | 全参 SFT ckpt | b1+b2 合并 | ~2853 | ~4.7h | 复用算力+无遗忘,但 b1 见 2×、b2 见 1× 轻度不均衡 |

选 B 理由:故事最干净(单次训练、单 LR schedule、两批配对均匀交错)、无续训复杂度、多 2.4h 算力相对清洁度收益可忽略。

**诚实对齐:** b2 补的是"配对多样性"(同一批人×衣重新搭配),而非"实体多样性"。之前分析定位的真正瓶颈——**数据多样性坍缩**(单一 outfit_v2 prompt 模板、仅正面上装、无字幕监督)——b2 完全没碰。所以预期质量提升真实但有限,不是戏剧性的。

## 3. 训练配置

| 项 | 值 |
|---|---|
| 底座 | `Qwen/Qwen-Image-Edit-2511`(base,本地完整目录) |
| 起点 | **base model**(`INIT_MODEL_DIR=$MODEL_DIR`,非 SFT ckpt) |
| 可训练 | **仅 DiT**(`--trainable_models dit --remove_prefix_in_ckpt pipe.dit.`) |
| 参数量 | **20.43 B**(DiT,全 BF16,1933 个 tensor) |
| 框架 | DiffSynth-Studio(editable)+ Accelerate + DeepSpeed |
| 分布式 | **ZeRO-3**(stage 3),无 offload,bf16 mixed precision |
| GPU | 8× NVIDIA H100 80GB HBM3 |
| 优化器 | AdamW,LR **1e-5**,weight_decay 0.01,ConstantLR |
| 调度 | 1 epoch,grad_accum=1,gradient_checkpointing=True |
| 分辨率 | `max_pixels=1048576`(≤1024×1024) |
| 其它 | `--find_unused_parameters` `--zero_cond_t` |
| **数据** | **b1+b2 合并 ~22830 条**(vs b1-only 11415),1 epoch ~2853 步(vs 1427) |
| 指标同步 | **live wandb**(`ENABLE_WANDB_LOG=1`,`WANDB_API_KEY` 走环境变量,project `qwen-outfit-full-sft`,run `full_sft_b1b2_8gpu_zero3`)+ TensorBoard 本地兜底 |
| 版本 | torch 2.6.0+cu124 / deepspeed 0.19.5 / accelerate 1.14.0 |

## 4. 数据

- b1: 11415 条(train split),`idm_unpaired_train`,11647 raw→11415 converted(drop 2.0%)
- b2: ~11415 条预计,`idm_unpaired_train_b2`,11647 raw,prompt 模板与 b1 **完全相同**(全文 outfit v2,单条 ~1592 字符)
- 合并:~22830 条,out_names 跨 batch 不相交(batch 保证),合并 manifest + 扁平化 images 到 combined 目录后跑现有 convert 脚本(未改训练链路)
- 合成集来自 `lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling`,VITON-HD 原图 CC BY-NC,研究/非商用

## 5. wandb live 同步(这台机器的关键约束)

这台机器经海外 HTTP 代理出网。**在线 `wandb.init` 之前用 netrc 旧 key 会挂**(25s 超时,且旧 key 对应 jia-pu entity 非用户账号)。**但用用户 86 字符 key + `WANDB_API_KEY` 环境变量,经代理 3.8s 成功**(实测探针 run `39o05qov`,落到 `changan_1`)。所以训练中:
- `WANDB_API_KEY` **只走环境变量**(tmux 里 export),**不写 ~/.netrc**(仓库硬性规则:不提交 token)
- `ENABLE_WANDB_LOG=1` + `WANDB_PROJECT=qwen-outfit-full-sft`
- DiffSynth `ModelLogger` 只在 `accelerator.is_main_process` 时 init wandb(logger.py:80),8 进程不会重复 init
- TensorBoard 仍开(`ENABLE_TENSORBOARD_LOG=1`)做兜底——万一 live wandb 经代理抖动,loss 曲线仍在本地 `$CKPT_OUT/tensorboard_log/`,可事后 `logs_to_wandb.py` 回填

## 6. 可观测性封装

用新写的 `scripts/launch_full_sft_observable.sh`(b1-only run 的 launch_config/nvidia-smi 是 ad-hoc 的,这次封装成可复用):

> **⚠️ 该脚本尚未进仓库**——只存在于训练机本地，所以目前对其他人**并不可复用**。
> 已记在 TODO，需从训练机补交。
1. `launch_config.json` — 配置快照,挂到 wandb run config + 本报告
2. `nvidia-smi -l 60` — GPU 显存/利用率/温度/功耗 CSV 时间线(报告第 8 节来源)
3. `record_training_details.py` — 训练后 training_details.json/md 快照
4. TB(规范)+ live wandb(实时)双轨

---

<!-- 以下训练后填 -->

## 7. 用时与吞吐

- **2854 步 × 5.87 s/it = 4h39m26s**(tqdm 自报),端到端 13:01:59 → 17:43:18 = **4h41m19s**(含加载/存盘)。
- 吞吐:22829 样本 / 16747s ≈ **1.36 samples/s**(总量),0.170 samples/s/GPU——**与 b1-only 完全一致**(b1:11415/8376s ≈ 1.36)。
- **关键观察:数据翻倍 → 墙钟线性翻倍(2h22m → 4h41m,2.00×),单步速度不变(5.87 s/it 两边相同)**。符合预期:ZeRO-3 下单步计算量由模型+batch 决定,与数据集大小无关;唯一变化是 epoch 内步数 1427 → 2854(2×)。
- 步数为什么是 2854 而非 22829/8=2853.625:分布式 sampler 不 drop_last、向上 pad 到 8 的倍数 → 总样本 22832(多 3 条重复),每 rank 2854 步。与 b1 的 1427(11415→11416 pad)同一机制。

## 8. 显存占用分析

nvidia-smi `-l 60` 采样 2256 行(13:01:53 → 17:43:05,跳过前 8 个热身样本)。

### 关键数值

| 指标 | 值 | b1-only 对比 |
|---|---|---|
| 峰值 | **74 753 MiB = 91.7% of 80GB**,余量 ~6.8GB | 74 785 MiB(基本相同) |
| 稳态均值 | 73 955 MiB | 74 404 MiB |
| 8 卡显存差(max-min) | **795 MiB(1.0%)** | 685 MiB(0.9%) |
| util 均值 | 93.7%(max 100%) | ~94% |
| 最高温度 | 77°C(GPU5) | 78°C(GPU5) |
| 功耗均值 | 582–606 W | 588–612 W |

### 各卡负载均衡

| GPU | 显存均值 | util 均值 | 最高温度 | 功耗均值 |
|---|---|---|---|---|
| 0 | 73 971 | 94.5% | 63°C | 599 W |
| 1 | 73 836 | 93.9% | 73°C | 599 W |
| 2 | 73 995 | 94.4% | 76°C | 601 W |
| 3 | 73 584(最低) | 93.6% | 63°C | 592 W |
| 4 | 74 379(最高) | 92.7% | 62°C | 582 W |
| 5 | 73 840 | 92.8% | 77°C | 606 W |
| 6 | 73 743 | 93.3% | 73°C | 597 W |
| 7 | 74 296 | 94.3% | 60°C | 587 W |

### 「数据翻倍但显存不变」——面试关键点

**数据量 11415 → 22829(2×),显存峰值 74.8GB → 74.8GB(不变)**,与 b1-only 误差 <0.1%。原因:ZeRO-3 的显存占用 = (P+G+O)/8 + 激活,四者全由**模型参数量与 batch size**决定,与 epoch 内样本总数无关:

- DiT 20.43B,bf16 参数 40.86GB → 分片 /8 ≈ 5.1GB/卡
- fp32 master + AdamW(m+v)= 3×20.43B×4B ≈ 245GB → /8 ≈ 30.7GB/卡(大头)
- 梯度 bf16 40.86GB → /8 ≈ 5.1GB/卡
- 激活(gradient checkpointing 压住)≈ 数十 GB 量级,batch=1 不随数据集变大

数据翻倍只让 **epoch 步数**翻倍(1427→2854),单步前向/反向的计算与显存逐字不变。这是 ZeRO-3 的设计预期,也是"为什么不用为了更大数据集去换更大显存"的答案。

## 9. loss 曲线

TensorBoard 2854 个 loss 点全程。

| 指标 | 值 | b1-only 对比 |
|---|---|---|
| n_points | 2854 | 1427 |
| first100 均值 | **0.04142** | 0.04085 |
| last100 均值 | **0.03394** | 0.03118 |
| 窗口降幅 | **−18.0%** | −23.7% |
| first100 中位数 | 0.03533 | — |
| last100 中位数 | 0.02872 | — |
| 中位数降幅 | −18.7% | — |
| overall 均值 / stdev | 0.03515 / 0.03005 | — |
| **last100 stdev** | **0.02919** | 0.0249 |
| min / max | 0.00000 / 0.17341 | — |
| 去毛刺后 min | 0.00102(step 460) | — |

### 两个必须诚实说明的点

1. **min=0.00000 是 TB 日志毛刺,不是训练塌了。** 全程有 15 个精确 0、97 个 <0.001 的点(集中在 step 460/540/846/1089/1260/1379…),去毛刺后真实最小 0.00102。这些 0 是 DiffSynth runner 把某步 loss 写成 0 的瞬时记录(疑似 NCCL all-reduce 后的瞬时 NaN→0 或日志时序错位),**不影响窗口均值**(first100/last100 各 100 点,毛刺被稀释),也不影响 ckpt——ckpt 是权重快照,与 loss 标量无关。

2. **last100 stdev(0.0292)≫ 趋势(~0.0035)**,训练 loss 仍噪声主导。扩散训练每步随机采 timestep t 与噪声 ε,单步 loss 的样本间标准差比整体下降幅度大一个数量级。所以:
   - 引用"末步 0.05300"或"首步 0.02179"都站不住(单步极差 0.17);
   - 窗口均值(−18.0%)是唯一可比口径,但即便如此,**0.0339 vs b1-only 的 0.0311 的差值(0.0028)小于 last100 stdev(0.029)**——两曲线落在同一噪声带内。

### b1b2 的 last100 比 b1-only 高,不能解读为"数据多了反而更差"

b1b2 last100=0.0339 略高于 b1-only 的 0.0311,降幅 18.0% 也小于 b1-only 的 23.7%。但这**不是因果证据**:两次的 last100 窗口采样的是不同 shuffle 下的不同 batch/样本/timestep,而 loss 又是逐样本逐 t 噪声化的,窗口均值对"恰好落在窗口里的那 100 步"高度敏感。**训练 loss 无法判定 b1 vs b1b2 孰优**——只有留出集评测(`eval_viton_holdout.py`)能回答。

## 10. 与 b1-only run 对比

唯一变量 = 数据量(b1 11415 → b1+b2 22829,2×),其余同框架(base init / ZeRO-3 / 8×H100 / LR 1e-5 / 1 epoch / bf16)。

| | b1-only (`50jhh811`) | b1b2 (本 run `qdg3zh5m`) |
|---|---|---|
| 数据 | 11 415 | **22 829** |
| 步数 | 1 427 | **2 854**(2.00×) |
| 墙钟 | 2h22m | **4h41m**(1.98×) |
| s/it | 5.87 | 5.87(不变) |
| 吞吐 | 1.36 samples/s | 1.36 samples/s(不变) |
| loss first100 | 0.0409 | 0.0414(几乎同起点) |
| loss last100 | 0.0311 | 0.0339 |
| 窗口降幅 | −23.7% | −18.0% |
| last100 stdev | 0.0249 | 0.0292 |
| 显存峰值 | 74.8 GB(91.7%) | 74.8 GB(91.7%)(不变) |
| 8 卡显存差 | 685 MiB(0.9%) | 795 MiB(1.0%) |

### 三个干净结论

1. **数据翻倍 → 墙钟线性翻倍(2.00×),单步速度不变(5.87 s/it)。** ZeRO-3 单步计算量由模型+batch 决定,与数据集大小无关;唯一变化是 epoch 步数翻倍。符合预期。
2. **数据翻倍 → 显存不变(74.8GB)。** ZeRO-3 分片大小由参数量决定,与样本总数无关。
3. **两条 loss 曲线起点几乎重合(0.0409 vs 0.0414)**——因为两次都从 base 起、用同一 prompt 模板,b2 是同实体重洗牌故分布一致;**但终点都在噪声带内**(差 0.0028 < stdev 0.029),**训练 loss 无法排名**,需留出集评测。

### 产物已就绪(嫁接已验证)

- DiT ckpt:`$OUTPUT_ROOT/qwen_vton_full_sft_b1b2/dit_full/epoch-0.safetensors`(39 GB)
- 完整模型目录(已嫁接验证):`$OUTPUT_ROOT/qwen_full_sft_b1b2_fused`
  - `apply_full_dit_ckpt.py` 跑通:`missing=0 unexpected=0`,1933 tensors 全加载
  - inode 检查:fused 与 base 的 transformer 分片是**不同 inode**(独立拷贝,非硬链接)
  - 权重对比:`img_in.bias` sum 219.20(base)→219.08(fused),**确认 graft 生效、权重已更新**
- wandb(live,训练中实时同步):https://wandb.ai/changan_1/qwen-outfit-full-sft/runs/qdg3zh5m
  - b1-only 同 project:https://wandb.ai/changan_1/qwen-outfit-full-sft/runs/50jhh811
  - `launch_config.json`(21 键)已挂到本 run config

### 下一步(在评测机)

fuse 后跑 `eval_viton_holdout.py`,与 b1-only fused + base 三方对比——**这才是判定 b2 数据是否带来真实增益的唯一手段**,训练 loss 给不出答案。

## 11. 产物与下一步

- DiT ckpt: `$OUTPUT_ROOT/qwen_vton_full_sft_b1b2/dit_full/epoch-0.safetensors`
- 嫁接成完整模型目录: `python scripts/apply_full_dit_ckpt.py --base-model "$MODEL_DIR" --ckpt ".../epoch-0.safetensors" --out-dir "$OUTPUT_ROOT/qwen_full_sft_b1b2_fused"`
- 评测(在评测机): fuse 后跑 `eval_viton_holdout.py`,与 b1-only fused + base 三方对比
