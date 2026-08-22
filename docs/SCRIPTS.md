# 脚本索引

37 个脚本按用途分组。**当前主线**用粗体标出；其余是被取代的早期版本，保留是为了留痕，日常不需要碰。
更早的一次性编排与 VITON 重建任务线已在 2026-08-16 删除（可在 git 历史中找回）。

约定：所有 shell 入口都 `source scripts/lib_env.sh`，路径来自 `configs/env.local.sh`。

---

## 1. 环境

| 脚本 | 用途 |
|---|---|
| `lib_env.sh` | 所有 shell 入口共用的路径解析，加载 `configs/env.local.sh` |
| `setup_env.sh` | 建 Qwen-Image-Edit conda 环境 |
| `setup_idm_vton.sh` | 装 IDM-VTON teacher（独立环境 + 权重） |

## 2. 数据：下载与合成

| 脚本 | 用途 |
|---|---|
| **`prepare_data_from_hf.sh`** | **一键：从公开 HF 拉合成集 + VITON-HD，扁平化分片，重建 metadata** |
| `download_vton_datasets.sh` | 直接下 VITON-HD / DressCode 原始数据 |
| `extract_dresscode_zip.py` | 解 DressCode 分卷（文件名可能是 .tar 实为 zip） |
| **`make_pair_batch.py`** | **生成新批次配对：排除同 id 自配、与既往批次去重（写盘前断言）** |
| **`run_idm_synth_batch.sh`** | **批次化多卡 IDM 合成，输出目录与配对文件均可配** |
| `synthesize_unpaired_idm.py` | 合成核心（被上面的启动脚本调用） |
| `merge_idm_shard_manifests.sh` | 合并各 shard 的 manifest |
| `preview_idm_synth_pairs.py` | 抽样预览合成质量：人物 \| 服装 \| 合成 GT |
| `run_idm_train_multigpu.sh` | 早期版：路径写死的 train split 合成，已被 `run_idm_synth_batch.sh` 取代 |
| `run_idm_batch_synth.sh` | 早期版：test split 续跑 |

## 3. 数据：转 DiffSynth metadata

| 脚本 | 用途 |
|---|---|
| **`convert_idm_synth_to_qwen_edit_v2.py`** | **合成 pair → 全文 v2 prompt metadata（当前训练用）** |
| **`run_convert_idm_v2.sh`** | **上者的封装，附带 prompt 审计输出** |
| `prompts_train_v2.py` | v2 prompt 构造（`prompts/outfit_v2.py` 的训练侧封装） |
| `convert_idm_synth_to_qwen_edit.py` | v1：短英文 prompt，`idm_lora_v1` 用的就是它 |

## 4. 训练

| 脚本 | 用途 |
|---|---|
| **`train_full_sft_zero3.sh`** | **全参 DiT SFT，Accelerate + DeepSpeed ZeRO-3 / ZeRO-2-offload** |
| **`train_lora_v2_multigpu.sh`** | **同数据 LoRA 对照，8 卡使步数与全参一致（1427）** |
| `train_idm_lora_multigpu.sh` | 早期 LoRA（v1 短 prompt、4 卡），产出 `idm_lora_v1` |

## 5. 权重后处理

| 脚本 | 用途 |
|---|---|
| **`apply_full_dit_ckpt.py`** | **全参产出的 DiT-only ckpt → 可加载的完整模型目录** |
| **`fuse_qwen_edit_lora.py`** | **LoRA 适配器融进底座（`W' = W + BA`）** |

## 6. 评测

| 脚本 | 用途 |
|---|---|
| **`eval_viton_holdout.py`** | **域内留出集评测（有 GT）；`--model` 同时接受模型目录与适配器文件** |
| **`run_case02_v2_prompt_eval.sh`** | **业务域评测，live v2 prompt + GPT 参照** |
| **`paired_eval_stats.py`** | **配对检验：判断评测差异是真实还是噪声** |
| **`check_paired_assumptions.py`** | **检验配对差值的正态假设，并用 Wilcoxon 做稳健性交叉验证**（需 scipy） |
| **`visualize_metrics.py`** | **指标可视化：差分热力图 + H-S 直方图 + 差异 CDF** |
| **`compose_case02_matrix.py`** | **把散落各目录的 case02 结果合成一张对照条** |
| `zero_shot_compare.py` | 业务域推理核心（被 case02 脚本调用） |
| `compose_idm_compare.py` | case02 拼图（`--idm-label` 指定第二个模型栏位名） |
| `probe_editplus_truncation.py` | 验证 EditPlus 是否截断长 prompt（结论见 KNOWLEDGE） |
| `run_case02_idm_vs_base.sh` | 短指令对照 |

## 7. 发布与记录

| 脚本 | 用途 |
|---|---|
| **`upload_all_synth_to_hf.py`** | **按批次上传数据集，`--synth NAME=PATH`，带溯源文件** |
| **`finish_and_publish_batch.sh`** | **合成完自动校验并发布：校验不过绝不上传** |
| `upload_lora_v2_to_hf.sh` | 上传 LoRA fused 模型到 HF 子目录 |
| `record_training_details.py` | 抓训练快照（配置/环境/数据/loss 曲线）→ JSON + Markdown |
| `logs_to_wandb.py` | 事后把 TensorBoard 曲线补录进 wandb |

---

## 典型链路

**从零到能开训**（详见 [REPRODUCE.md](REPRODUCE.md)）：

```
setup_env.sh → prepare_data_from_hf.sh → train_full_sft_zero3.sh → apply_full_dit_ckpt.py
```

**加一批合成数据并发布**：

```
make_pair_batch.py → run_idm_synth_batch.sh → finish_and_publish_batch.sh
```

**评一个新模型**：

```
eval_viton_holdout.py → paired_eval_stats.py → visualize_metrics.py   # 域内
run_case02_v2_prompt_eval.sh → compose_case02_matrix.py               # 业务域
```
