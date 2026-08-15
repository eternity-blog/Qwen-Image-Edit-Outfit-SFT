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
- [x] 指标定义/读法文档化 + 可视化脚本（EVAL.md、`visualize_metrics.py`）  
- [x] 仓库清理：合并可视化脚本、删除死配置、修 `record_training_details.py`  

## Next

- [ ] **用 v2 数据重训 LoRA**（缺失的对照实验：现有 LoRA 训的是 v1 短英文 prompt，
      与全参不同数据，故「全参 vs LoRA」目前无法成立。约 4h/4×H100，见 EVAL_RESULTS 第 3 节）  
- [ ] **prompt 表层增广**后重训（方案 1，零新增图片，验证「指令单一」是否为主因）  
- [ ] **真实帧 + GPT 作第二 teacher**（方案 4，补目标域 / 字幕 / 颜色保真）  
- [ ] 扩 pair 到 k=3~5（方案 2）  
- [ ] 模型仓库当前为 private，决定是否公开  
- [ ] 把 nvidia-smi 采样并入训练脚本（当前显存数据依赖手动采样，不可复现）  

## Deferred

- [ ] 多参考增强（VITON 无同款多视图；DressCode / 电商图可解）  
- [ ] 花字 / 水印专用监督  
