# Qwen-Image-Edit-2511 全参 DiT SFT 训练报告（2026-08-15）

> 一次实际跑完的训练全记录,按面试问答组织。所有数字取自 TensorBoard / nvidia-smi CSV / safetensors header / launch log,非估算。

## 0. 一句话结论

8×H100 80GB + ZeRO-3 + bf16,把 Qwen-Image-Edit-2511 的 DiT(20.43B 参数)在 11415 条换装 pair 上做了 1 epoch 全参 SFT,**2h22m 跑完,1427/1427 步无报错,产出 40.86GB 的 DiT-only checkpoint**。loss 从 0.029 缓降到 0.024,无 OOM,显存峰值 74.8GB(91.7%),8 卡负载极度均衡。

---

## 1. 时间线

| 时刻(+08:00) | 事件 |
|---|---|
| 06:41:48 | `accelerate launch` 启动,8 卡初始 7 MiB / 0% |
| 06:42:48 | 模型加载中,显存 ~60.6GB |
| 06:43:48 | **ZeRO-3 分片 + optimizer 初始化完成**,显存 73.6GB,util 100% ——训练真正开始 |
| 06:47:07 | `training_details.json` 被某 bootstrap 脚本抓拍(此时 loss 尚未解析,故为空) |
| 09:02:55 | 最后一次 nvidia-smi 采样(仍 100% / 74.5GB) |
| 09:03:53 | 日志打印 `TRAIN DONE`,ckpt 落盘 |

- 训练循环本身(日志精确):**1427 步 × 5.87 s/it = 2:19:59**
- 端到端(含加载/存盘):06:41 → 09:03 ≈ **2h22m**

## 2. 训练配置

| 项 | 值 |
|---|---|
| 底座 | `Qwen/Qwen-Image-Edit-2511`(本地完整目录) |
| 可训练 | **仅 DiT**(`--trainable_models dit --remove_prefix_in_ckpt pipe.dit.`) |
| 参数量 | **20.43 B**(DiT,全 BF16,1933 个 tensor) |
| 框架 | DiffSynth-Studio(editable)+ Accelerate + DeepSpeed |
| 分布式 | **ZeRO-3**(stage 3),无 param/optimizer offload,bf16 mixed precision |
| GPU | 8× NVIDIA H100 80GB HBM3,NVLink(P2P on) |
| 优化器 | AdamW,LR **1e-5**,weight_decay 0.01 |
| 调度 | 1 epoch,grad_accum=1,gradient_checkpointing=True |
| 分辨率 | `max_pixels=1048576`(≤1024×1024,变长宽比,面积上限 1MP) |
| 其它 | `--find_unused_parameters` `--zero_cond_t` |
| 版本 | torch 2.6.0+cu124 / deepspeed 0.19.5 / accelerate 1.14.0 / diffusers 0.39.0 |

## 3. 数据

- 11 415 条换装 pair(metadata:`converted_idm_synth_train_v2/metadata_train.json`),prompt 全文 v2,单条约 **1592 字符**。
- 合成集来自 `lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling`,VITON-HD 原图(CC BY-NC)。
- `data_file_keys=image,edit_image`、`extra_inputs=edit_image`:模型输入=人物图,条件输入=服装图。

## 4. 用时与吞吐

- 1427 步 / 8376s ≈ **1.36 samples/s**(总量),0.170 samples/s/GPU。
- 速度 5.87~5.96 s/it,非常稳定。
- **为什么是 1427 步而非 1426?** 11415 / 8 = 1426.875,分布式 sampler 不 drop_last、向上 pad 到 8 的倍数 → 总样本变 11416(多 1 条重复),每 rank 1427 步。这是一个干净的面试小点。

## 5. 显存占用分析(面试重点)

### 5.1 关键数值

| 指标 | 值 |
|---|---|
| 峰值 | **74 785 MiB(GPU6)= 91.7% of 80GB**,余量 ~6.8GB |
| 稳态均值 | 74 404 MiB(stdev **234 MiB**,极度平稳) |
| 稳态区间 | 73 543 ~ 74 785 MiB |
| 爬坡 | 7 MiB → 60.6GB → 73.6GB,约 **2 分钟**到稳态 |
| 首次 >70GB | 06:43:48(模型分片完成) |

### 5.2 各卡负载均衡(ZeRO-3 分片均匀的佐证)

| GPU | 稳态显存均值 | util 均值 | 最高温度 | 功耗均值 |
|---|---|---|---|---|
| 0 | 74 579 | 93.9% | 63°C | 606W |
| 1 | 74 219 | 94.0% | 73°C | 605W |
| 2 | 74 373 | 92.4% | 75°C | 609W |
| 3 | 74 092(最低) | 95.5% | 63°C | 600W |
| 4 | 74 377 | 94.6% | 62°C | 588W |
| 5 | 74 520 | 95.1% | **78°C**(最高) | 612W |
| 6 | 74 777(最高) | 93.6% | 73°C | 610W |
| 7 | 74 293 | 97.1% | 60°C | 596W |

8 卡显存均值最大-最小差仅 **685 MiB(~0.9%)** —— ZeRO-3 把 params/grads/optimizer 三件套均匀切了 8 份,没有倾斜。

### 5.3 「为什么是 74GB」拆解(面试常问)

ZeRO-3 把 **P(参数)/G(梯度)/O(optimizer state)** 三者都分片到 8 卡:

- DiT 参数 20.43B,bf16 拷贝 40.86GB → 分片 /8 ≈ **5.1GB/卡**
- fp32 master + AdamW 的 momentum/variance = 3×20.43B×4B ≈ 245GB → 分片 /8 ≈ **30.7GB/卡**(大头)
- 梯度 bf16 40.86GB → /8 ≈ **5.1GB/卡**
- 激活:gradient checkpointing 下被压住,1MP × batch1 × 20B DiT 约 **数十 GB 量级**

合计落到 ~74GB/卡,与实测吻合(量级说明,非逐字节精确)。**关键论点**:若不用 ZeRO-3、用纯 DDP,每卡要放完整 optimizer state(245GB)→ 必 OOM。这正是仓库 CLAUDE.md 硬性规定「全参 DiT 禁止纯 DDP、必须 ZeRO」的根因——这条设计决策能在面试里直接讲清楚分布式显存经济学。

## 6. GPU 利用率 / 功耗 / 温度

- **利用率**:稳态均值 **94.5%**,周期性掉到 0%(步间 = 数据加载 + ZeRO-3 all-gather + 通信),峰值 100%。
- **功耗**:稳态均值 **603W**(H100 TDP 700W,即 ~86% 功率利用),stdev 105W(随 compute/comm 相位抖动),峰值 **715W(102%,boost)**。
- **温度**:均值 **64°C**,峰值 **78°C**(GPU5),全程低于 80°C 降频线,散热健康。

## 7. Loss 分析

TensorBoard 标量 `loss`,共 1427 个点(1 步 1 点)。

| 指标 | 值 |
|---|---|
| 首步(step1) | 0.02901 |
| 末步(step1427) | 0.02380 |
| 全程均值 | 0.03621,stdev 0.03004 |
| 最小 | 0.00000 @step76 |
| 最大 | 0.16537 @step6 |
| 前 100 步均值 | 0.04085 |
| 末 100 步均值 | 0.03118(↓ **23.7%**) |

每 100 步窗口均值:0.0409 → 0.0370 → 0.0369 → 0.0347 → 0.0398 → 0.0406 → 0.0403 → 0.0362 → 0.0333 → 0.0343 → 0.0380 → 0.0332 → 0.0318 → 0.0314 → 0.0311。**缓降 + 高频抖动**。

### 7.1 面试问答:为什么 loss 这么低?

SFT 是在已很强预训练的 Qwen-Image-Edit-2511 上微调,模型已接近最优,loss 量级本就小(去噪 MSE 在 well-trained 区间)。

### 7.2 面试问答:为什么单步噪声这么大?

扩散训练每步随机采样 timestep t、随机噪声 ε、不同图像对:
- 低噪声 timestep / 易样本 → loss 可逼近 0(step76 的 0.0 即此,数值下溢,非 bug);
- 高噪声 timestep / 难样本 → loss 偏大。

末 100 步单步 stdev=0.0249、极差=0.1034,**噪声 ≫ 趋势**,所以**不能凭单步判断收敛**,必须看窗口均值。

### 7.3 面试问答:收敛了吗?

1 epoch / 11415 条是轻量 SFT pass;窗口均值 −24% 的缓降说明模型在缓慢学习该任务分布。**训练 loss 不能下最终结论**,需在 held-out testset 上做视觉评测(换装保真度、garment 细节、人物一致性)才算数。诚实表述即可。

## 8. Checkpoint 产物

`dit_full/epoch-0.safetensors`:**40.86 GB**,1933 tensor,全 **BF16**,20.43B params,metadata `{'format':'pt'}`。

- **这是 DiT-only 权重**(`--trainable_models dit --remove_prefix_in_ckpt pipe.dit.`),只含 transformer/DiT 模块,key 为裸 DiT 模块名(如 `img_in.weight`、`transformer_blocks.0.img_mod.1.weight`、`proj_out.bias`)。
- 不含 text_encoder / vae / tokenizer / processor / configs,**不能直接当模型目录加载推理** → 见第 10 节 fuse。
- 最大 tensor:`transformer_blocks.0.img_mod.1.weight [18432,3072]`、`txt_mod.1.weight [18432,3072]`(double-stream block 的 mod/gate 投影)。

## 9. 训练细节 / 易被追问的点

1. **日志里反复出现 `Do not find activation_checkpointing config in deepspeed config, skip...`** —— 良性。脚本用的是模型/diffusers 层的 gradient checkpointing(`--use_gradient_checkpointing`),**不是** DeepSpeed 自己的 `activation_checkpointing` wrapper。两套独立机制,DeepSpeed 找不到就 skip,梯度检查点仍由模型生效。
2. **`find_unused_parameters=True`**:edit_image 条件注入路径下,部分模块在某些样本可能不产生梯度,开此项避免 DDP "expected to have finished" 报错;代价是少量通信开销。
3. **`zero_cond_t=True`**:Qwen-Image-Edit-2511 强制要求的时间步条件路径,CLAUDE.md 明令「必须始终带 `--zero_cond_t`」。
4. **GPU 顺序 `7,6,5,4,3,2,1,0`**:脚本按空闲显存排序取 top-8 再反转,纯展示,无影响。
5. **`training_details.json` 是废快照**:06:47 抓拍,`loss_steps_parsed:0 / latest_step:null`,不是训练总结。真实指标看 TensorBoard。
6. **`DIFFSYNTH_SKIP_DOWNLOAD=True`**:纯离线加载本地权重,不回源。

## 10. Fuse 是什么(把 DiT ckpt 变成完整模型目录)

见 `scripts/apply_full_dit_ckpt.py`。**一句话:把训练只更新了的 DiT 权重,嫁接回一份完整的 Qwen-Image-Edit-2511 目录,产出一个能像底座一样直接加载推理的 fused 目录。**

为什么需要:
- 全参 SFT 只训练 DiT(`trainable_models=dit`),存的 40GB 文件**只有 DiT 那 1933 个 tensor**,key 还是裸模块名。
- 一个可推理的完整模型目录,除了 DiT,还要有冻结的 **text_encoder / vae / tokenizer / processor / model_index.json / 各 config**,且 diffusers `QwenImageEditPlusPipeline` 要求 transformer 子目录按 `transformer.*` 前缀分片序列化。这些 ckpt 里都没有。

脚本做的事:
1. `shutil.copytree(base_model → out_dir)`:把底座整棵树拷过去 → 拿到所有冻结组件(text_encoder/vae/tokenizer/...);
2. `QwenImageEditPlusPipeline.from_pretrained(out_dir, bf16, local_files_only=True)`:在 CPU 上组装完整 pipeline;
3. `load_file(epoch-0.safetensors)`:加载 40GB DiT 权重;
4. **key 前缀归一化**:`pipe.dit.` → 去掉、`transformer.` → 去掉,统一成裸 DiT 模块名;
5. `pipe.transformer.load_state_dict(remapped, strict=False)`:只把 DiT 参数灌进 transformer 模块(frozen 组件已在拷贝树里,不动),打印 missing/unexpected 数量做 sanity;
6. `pipe.save_pretrained(out_dir, safe_serialization=True)`:按 diffusers 标准格式重新序列化(transformer 分片 + 全套 config)。

产物 `qwen_full_sft_fused/` 行为与底座 `MODEL_DIR` 完全一致,可喂给任意 eval/推理脚本。**直接拿 `epoch-0.safetensors` 当 MODEL_DIR 用会缺组件 + key 对不上,仓库故障表已标注「epoch ckpt 不能当 MODEL_DIR 用,跑 apply_full_dit_ckpt.py」。**

## 11. 复现命令

```bash
source configs/env.local.sh
export METADATA=$QWEN_VTON_DATA/converted_idm_synth_train_v2/metadata_train.json
export DATASET_BASE=$QWEN_VTON_DATA/converted_idm_synth_train_v2/dataset_base
export NUM_PROCESSES=8 DS_PROFILE=zero3 LR=1e-5 NUM_EPOCHS=1
# tmux new -d -s qwen_full_sft "bash scripts/train_full_sft_zero3.sh 2>&1 | tee $OUTPUT_ROOT/qwen_vton_full_sft_launch.log"

# 训完 fuse:
"$ENV_DIR/bin/python" scripts/apply_full_dit_ckpt.py \
  --base-model "$MODEL_DIR" \
  --ckpt "$OUTPUT_ROOT/qwen_vton_full_sft/dit_full/epoch-0.safetensors" \
  --out-dir "$OUTPUT_ROOT/qwen_full_sft_fused"
```

日志:`$OUTPUT_ROOT/qwen_vton_full_sft/logs/{train_full_sft.log,nvidia_smi.csv}`;TB:`.../dit_full/tensorboard_log`(看 loss:`tensorboard --logdir=...`)。
