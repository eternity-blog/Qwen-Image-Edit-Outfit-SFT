# 数据构建与 Hugging Face 发布

目标：任何人按文档能**复现 metadata**；合成图体积大，建议放到 **个人 Hugging Face Dataset**，仓库只保留脚本与卡片。

---

## 1. 目录约定（本地 / 集群）

通过 `configs/env.local.sh` 设置 `QWEN_VTON_DATA`（或 `DATA_ROOT`）：

```text
$QWEN_VTON_DATA/
  raw/viton_hd/                     # 原始 VITON（自行下载）
  synth/idm_unpaired_train/         # IDM 输出: images/ + manifest.jsonl
  converted_idm_synth_train_v2/     # 全文 v2 metadata（本仓脚本生成）
    dataset_base/                   # symlink → viton + synth
    metadata_train.jsonl
    metadata_val.jsonl
    stats.json
```

DiffSynth 训练读：`dataset_base` + `metadata_*.json`。

---

## 2. 构建流水线（脚本均在 `scripts/`）

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 下 VITON | `download_vton_datasets.sh` | 按需改镜像/路径 |
| 装 IDM | `setup_idm_vton.sh` | teacher 代码+权重 |
| 合成 | `synthesize_unpaired_idm.py` / `run_idm_*.sh` | unpaired try-on → manifest |
| 短指令 metadata（旧） | `convert_idm_synth_to_qwen_edit.py` | 对比实验用 |
| **全文 v2 metadata（主）** | `run_convert_idm_v2.sh` | 不重出图，只改 prompt |

每条样本字段见 `examples/metadata_row.json`。

**许可：** VITON BY-NC、IDM BY-NC-SA → HF 上务必选非商用许可并写 NOTICE。

---

## 3. 上传到个人 Hugging Face Dataset

### 3.1 建议上传内容

为方便复现，推荐 **一个 dataset repo** 包含：

```text
idm_unpaired_train/
  images/                 # 或 tar 分卷
  manifest.jsonl
converted_idm_synth_train_v2/
  metadata_train.jsonl
  metadata_val.jsonl
  stats.json
  prompt_example_full.txt
README.md                 # dataset card（本仓 dataset_card/ 可作模板）
```

VITON 原图很大且有独立许可：可要求用户自行下载 VITON，你的 HF 只放 **synth + metadata**（manifest 里 person/cloth 相对路径说明如何拼回 VITON）。

### 3.2 一键脚本

```bash
huggingface-cli login
pip install datasets huggingface_hub

python scripts/upload_dataset_to_hf.py \
  --repo-id <your-username>/qwen-outfit-idm-synth-v2 \
  --synth-dir $QWEN_VTON_DATA/synth/idm_unpaired_train \
  --converted-dir $QWEN_VTON_DATA/converted_idm_synth_train_v2 \
  --private   # 或去掉该 flag 公开（仍需 NC 声明）
```

### 3.3 别人怎么用你的数据

```text
1. huggingface-cli download <repo-id> --local-dir ./hf_data
2. 自行准备 viton_hd 到 raw/
3. 按 converted 的 dataset_base 建 symlink
4. bash scripts/train_idm_lora_multigpu.sh
```

在本仓 README 填上你的 `repo-id` 链接即可闭环。

---

## 4. 多参考

当前阶段默认 2 图输入。无同款多视图数据前不做多参考增强（见 TODO）。
