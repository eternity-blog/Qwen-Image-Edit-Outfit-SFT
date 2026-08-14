# 评测

## 入口

- 主评测（全文 v2）：`scripts/run_case02_v2_prompt_eval.sh`  
- 短指令对照：`scripts/run_case02_idm_vs_base.sh`  
- 核心逻辑：`scripts/zero_shot_compare.py`  
- 拼图：`scripts/compose_idm_compare.py`  
- 截断探测：`scripts/probe_editplus_truncation.py`  

## Prompt 模式

| `--prompt-mode` | 说明 |
|---|---|
| `v2` | 全文 garment-only 模板（主评测） |
| `short` | 短换衣指令 |
| `production` | outfit_spec 内存贮 prompt（可能过时） |

## 建议关注点

1. 长指令下是否整幕重画  
2. 朝向 / 背景是否与源帧一致  
3. 颜色是否跟随主商品图  
4. 短指令回归是否明显退化  

原理说明见 [KNOWLEDGE.md](KNOWLEDGE.md)。
