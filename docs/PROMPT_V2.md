# 编辑指令（全文 v2）

训练与评测使用同一长模板：`prompts/outfit_v2.py` 中的 `outfit_garment_only_keyframe_prompt`。

- 与线上 garment-only 风格对齐（不压缩）  
- 合成数据无花字 GT → `overlay_placement=none`  
- 当前 2 图：`[源帧, 商品]`  

构建：`scripts/prompts_train_v2.py`、`convert_idm_synth_to_qwen_edit_v2.py`。
