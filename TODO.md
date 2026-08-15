# TODO

## Done

- [x] 全文 v2 metadata 转换  
- [x] IDM 合成 / LoRA 训练 / fuse / 评测脚本  
- [x] EditPlus 长 prompt 行为结论（见 KNOWLEDGE）  
- [x] HF 数据集发布与 README 链接  
- [x] 全参 ZeRO-3 启动脚本 + 完整复现指南（REPRODUCE.md）  
- [x] 8 卡全参 SFT 跑通并上传模型（`lee31221/Qwen-Image-Edit-Outfit-2511-SFT`）  
- [x] 双轨评测：VITON 留出集 + case02 业务域（见 EVAL_RESULTS_20260815.md）  
- [x] 数据缺口诊断与扩充方案（DATA_SCALING_PLAN.md）  

## Next

- [ ] **prompt 表层增广**后重训（方案 1，零新增图片，验证「指令单一」是否为主因）  
- [ ] **真实帧 + GPT 作第二 teacher**（方案 4，补目标域 / 字幕 / 颜色保真）  
- [ ] 扩 pair 到 k=3~5（方案 2）  
- [ ] 修 `record_training_details.py`：硬编码 repo_root、`$ENV_DIR` 字面量、无效 loss 正则  
- [ ] 模型仓库当前为 private，决定是否公开  
- [ ] 删掉未被引用的 `configs/ds_zero3_bf16_example.json`（实际用 accelerate yaml）  
- [ ] 把 nvidia-smi 采样并入训练脚本（当前显存数据依赖手动采样，不可复现）  

## Deferred

- [ ] 多参考增强（VITON 无同款多视图；DressCode / 电商图可解）  
- [ ] 花字 / 水印专用监督  
