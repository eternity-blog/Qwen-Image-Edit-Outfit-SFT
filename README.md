# Qwen-Image-Edit Outfit SFT

仓库：https://github.com/eternity-blog/Qwen-Image-Edit-Outfit-SFT

基于 **Qwen-Image-Edit-2511** 的电商关键帧局部换装 SFT 项目：合成编辑对、全文换装指令对齐、多卡 LoRA 训练，并支持后续多卡全参 SFT（ZeRO-3）。

## 功能

- **数据**：VITON-HD + IDM-VTON teacher 合成 unpaired 换装 pair，导出 DiffSynth metadata（全文 v2 prompt）
- **训练**：DiffSynth DiT-LoRA（`zero_cond_t`）；多卡全参 ZeRO-3（`scripts/train_full_sft_zero3.sh`）
- **评测**：域内留出集（有 GT）+ 业务域（GPT 参照）双轨，含指标可视化
- **发布**：合成数据与 SFT 模型已上传 Hugging Face

数据集：[lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling](https://huggingface.co/datasets/lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling)（IDM synth train/test + v2 metadata；NC 衍生许可）  
模型：[lee31221/Qwen-Image-Edit-Outfit-2511-SFT](https://huggingface.co/lee31221/Qwen-Image-Edit-Outfit-2511-SFT)（全参 DiT SFT，结果见 [EVAL_RESULTS](docs/EVAL_RESULTS_20260815.md)）

**从零复现（含全参）：** [docs/REPRODUCE.md](docs/REPRODUCE.md)。  
**给 Claude Code 在新机器上重建环境：** 根目录 [CLAUDE.md](CLAUDE.md)（按阶段执行即可开训）。  
相关原理见 [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md)；任务背景见 [docs/BACKGROUND.md](docs/BACKGROUND.md)。

## 仓库结构

```text
.
├── CLAUDE.md          # Claude Code：新机器环境重建 runbook
├── README.md
├── LICENSE / NOTICE.md
├── TODO.md
├── requirements.txt
├── configs/           # env 模板、Accelerate ZeRO 配置
├── prompts/           # 全文 garment-only 编辑指令模板
├── scripts/           # 48 个脚本，分组见 docs/SCRIPTS.md
├── docs/
│   ├── SCRIPTS.md            # 脚本索引：按用途分组，标注当前主线
│   ├── REPRODUCE.md          # 从零复现（含全参）
│   ├── BACKGROUND.md         # 任务背景 + 技术选型论证
│   ├── DATA.md               # 数据构建与 HF 发布
│   ├── DATA_SCALING_PLAN.md  # 数据缺口诊断与扩充方案
│   ├── TRAINING.md           # LoRA 与全参 ZeRO-3
│   ├── EVAL.md               # 评测协议、指标定义与读法
│   ├── EVAL_RESULTS_*.md     # 各轮评测结果（含对照有效边界）
│   ├── HP_SWEEP_PLAN.md      # 单变量超参扫描方案
│   ├── FULL_SFT_*_RUN_*.md   # 全参训练全记录（b1 / b1+b2）
│   ├── LORA_V2_RUN_*.md      # 同数据 LoRA 对照训练记录
│   ├── KNOWLEDGE.md          # 扩散编辑 / SFT / 并行与显存
│   ├── images/               # 文档配图（评测对照图）
│   └── archive/              # 早期方案留档
├── dataset_card/      # HF Dataset 卡片模板
└── examples/          # metadata 单条样例
```

`outputs/` 被 gitignore：评测拼图、指标可视化、训练日志都在那里，不随仓库分发。

大权重与原始数据不进 git；路径用 `configs/env.local.sh`（见 `configs/env.example.sh`）。

## 快速开始

```bash
cp configs/env.example.sh configs/env.local.sh   # 填写 MODEL_DIR / DATA 等
pip install -r requirements.txt
# DiffSynth、底座权重：见 docs/REPRODUCE.md

bash scripts/prepare_data_from_hf.sh        # 或本地合成后 run_convert_idm_v2.sh
bash scripts/train_idm_lora_multigpu.sh     # 多卡 LoRA（可选）
bash scripts/train_full_sft_zero3.sh        # 全参 ZeRO-3（建议 8×80GB）
python scripts/apply_full_dit_ckpt.py --help
bash scripts/run_case02_v2_prompt_eval.sh   # 评测
```

## 文档

| 文档 | 内容 |
|---|---|
| [SCRIPTS](docs/SCRIPTS.md) | 脚本索引（48 个，按用途分组） |
| [REPRODUCE](docs/REPRODUCE.md) | 从零复现 |
| [DATA](docs/DATA.md) | 数据构建与 HF 发布 |
| [DATA_SCALING_PLAN](docs/DATA_SCALING_PLAN.md) | 数据缺口诊断与扩充方案 |
| [TRAINING](docs/TRAINING.md) | LoRA 与全参 ZeRO-3 |
| [EVAL](docs/EVAL.md) | 评测协议、指标定义与读法、可视化 |
| [EVAL_RESULTS_20260815](docs/EVAL_RESULTS_20260815.md) | 双轨评测结果：7 模型对照、LR 扫描、失效归因 |
| [FULL_SFT_RUN_20260815](docs/FULL_SFT_RUN_20260815.md) | 全参 SFT 训练全记录（b1，11 415 条） |
| [FULL_SFT_B1B2_RUN_20260816](docs/FULL_SFT_B1B2_RUN_20260816.md) | 全参 SFT 训练全记录（b1+b2，22 829 条） |
| [HP_SWEEP_PLAN](docs/HP_SWEEP_PLAN.md) | 单变量超参扫描方案 |
| [LORA_V2_RUN_20260815](docs/LORA_V2_RUN_20260815.md) | 同数据 LoRA 对照训练记录 |
| [KNOWLEDGE](docs/KNOWLEDGE.md) | 扩散编辑 / SFT / 并行与显存等知识详解 |
| [BACKGROUND](docs/BACKGROUND.md) | 任务背景与技术选型论证 |

## 许可

代码：MIT。VITON-HD / IDM-VTON 及派生合成数据多为非商用，见 [NOTICE.md](NOTICE.md)。
