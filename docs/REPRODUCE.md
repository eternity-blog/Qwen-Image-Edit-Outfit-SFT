# 从零复现

## A. 环境

1. Clone 本仓库  
2. `cp configs/env.example.sh configs/env.local.sh`，填写 `MODEL_DIR`、`DIFFSYNTH_DIR`、`QWEN_VTON_DATA`、`OUTPUT_ROOT`、`ENV_DIR`  
3. `pip install -r requirements.txt`  
4. `pip install -e $DIFFSYNTH_DIR`  
5. 全参训练另装：`pip install deepspeed`  

## B. 数据

**使用已发布的 HF Dataset：**

```bash
huggingface-cli download <hf-dataset-repo> --local-dir $QWEN_VTON_DATA/from_hf
# 按 docs/DATA.md 接好 viton_hd 与 symlink
```

**或自行合成：**

```bash
bash scripts/setup_idm_vton.sh
# synthesize / run_idm_* …
bash scripts/run_convert_idm_v2.sh
```

## C. LoRA 训练

```bash
bash scripts/train_idm_lora_multigpu.sh
python scripts/fuse_qwen_edit_lora.py \
  --base-model "$MODEL_DIR" \
  --lora-path <lora_out> \
  --out-dir <fused_out>
```

## D. 评测

```bash
bash scripts/run_case02_v2_prompt_eval.sh
```

详见 [EVAL.md](EVAL.md)。

## E. 全参 ZeRO-3

见 [TRAINING.md](TRAINING.md)；启动脚本待补（TODO）。

## F. 检查清单

- [ ] metadata 指向全文 v2  
- [ ] LoRA 训练完成并可 fuse  
- [ ] live v2 评测出图  
- [ ] （可选）HF 数据可下载复现  
