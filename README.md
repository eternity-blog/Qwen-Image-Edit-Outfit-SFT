# Qwen-Image-Edit Outfit SFT

仓库：https://github.com/eternity-blog/Qwen-Image-Edit-Outfit-SFT

基于 **Qwen-Image-Edit-2511** 的电商关键帧局部换装 SFT 项目：合成编辑对、全文换装指令对齐、多卡 LoRA 训练，并支持后续多卡全参 SFT（ZeRO-3）。

## 功能

- **数据**：VITON-HD + IDM-VTON teacher 合成 unpaired 换装 pair，导出 DiffSynth metadata（全文 v2 prompt）
- **训练**：DiffSynth DiT-LoRA（`zero_cond_t`）；全参 ZeRO-3 配置草案见 `docs/TRAINING.md`
- **评测**：与 GPT / base 对照的关键帧拼图评测
- **发布**：合成数据可上传至 Hugging Face Dataset

数据集：[lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling](https://huggingface.co/datasets/lee31221/Outfit_Qwen-Image-Edit-2511_in_Kling)（IDM synth train/test + v2 metadata；NC 衍生许可）

相关原理与模型知识见 [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md)。任务背景见 [docs/BACKGROUND.md](docs/BACKGROUND.md)。

## 仓库结构

```text
.
├── README.md
├── LICENSE / NOTICE.md
├── TODO.md
├── requirements.txt
├── configs/           # env、DeepSpeed ZeRO-3 示例
├── prompts/           # 全文 garment-only 编辑指令
├── scripts/           # 数据 / 训练 / 评测 / HF 上传
├── docs/
│   ├── KNOWLEDGE.md   # 模型与训练相关知识详解
│   ├── BACKGROUND.md
│   ├── DATA.md
│   ├── TRAINING.md
│   ├── EVAL.md
│   └── REPRODUCE.md
├── dataset_card/      # HF Dataset 卡片模板
└── examples/
```

大权重与原始数据不进 git；路径用 `configs/env.local.sh`（见 `configs/env.example.sh`）。

## 快速开始

```bash
cp configs/env.example.sh configs/env.local.sh   # 填写 MODEL_DIR / DATA 等
pip install -r requirements.txt
# DiffSynth、底座权重：见 docs/REPRODUCE.md

bash scripts/run_convert_idm_v2.sh          # 全文 v2 metadata
bash scripts/train_idm_lora_multigpu.sh     # 多卡 LoRA
bash scripts/run_case02_v2_prompt_eval.sh   # 评测

# 可选：上传合成数据
python scripts/upload_all_synth_to_hf.py --help
```

## 文档

| 文档 | 内容 |
|---|---|
| [REPRODUCE](docs/REPRODUCE.md) | 从零复现 |
| [DATA](docs/DATA.md) | 数据构建与 HF 发布 |
| [TRAINING](docs/TRAINING.md) | LoRA 与全参 ZeRO-3 |
| [EVAL](docs/EVAL.md) | 评测 |
| [KNOWLEDGE](docs/KNOWLEDGE.md) | 扩散编辑 / SFT / 并行与显存等知识详解 |
| [BACKGROUND](docs/BACKGROUND.md) | 任务背景 |

## 许可

代码：MIT。VITON-HD / IDM-VTON 及派生合成数据多为非商用，见 [NOTICE.md](NOTICE.md)。
