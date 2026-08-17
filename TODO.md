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
- [x] 同数据 LoRA 对照 + LR 扫描（4 个 LR 点）  
- [x] **200 样本复现**：推翻 6 样本的「LoRA 与全参打平」，全参在 MAD 上显著更优但 LoRA 拿到 98.9% 收益  
- [x] case02 业务域补齐 LoRA 列 + 参考图数量对照（定位崩溃主因是多参考输入）  
- [x] batch2 合成 11 647 条并发布 HF（与 batch1 零重复）  
- [x] b1+b2 合并数据全参 SFT（22 829 条 / 2854 步 / 4h41m，见 FULL_SFT_B1B2_RUN）  
- [x] b1b2 模型评测：配对检验三项指标**均无显著变化**，证伪「堆配对数量」路线  
- [x] 指标定义/读法文档化 + 可视化脚本（EVAL.md、`visualize_metrics.py`）  
- [x] 仓库清理：合并可视化脚本、删除死配置、修 `record_training_details.py`  

## Next

- [ ] 用 IDM-VTON 在真实业务帧上跑换装，检查其前置产物（parsing / densepose / mask）是否可用——
      验证「真实帧的 teacher 必须换成 GPT」这一判断  
- [ ] 业务域样本量仅 2 shot，结论强度远弱于域内；需扩充 case 或明确标注其证据等级  

- [ ] 补交 `scripts/launch_full_sft_observable.sh`：FULL_SFT_B1B2_RUN 第 6 节称其"封装成可复用"，但该文件只在训练机上、未进仓库  
- [ ] **多参考训练数据**：随机化 `n_product_refs ∈ {1,2,3}`（P0，已证实是业务域最大单点故障）  
- [ ] **prompt 表层增广**后重训（方案 1，零新增图片；已知不是 case02 主因，降为 P2）  
- [ ] **真实帧 + GPT 作第二 teacher**（方案 4，补目标域 / 字幕 / 颜色保真）  
- [ ] ~~扩 pair 到 k=3~~（k=2 已证伪，先验变差；如做需先说明为何期待拐点）  
- [ ] 把 nvidia-smi 采样并入训练脚本（当前显存数据依赖手动采样，不可复现）  

## Deferred

- [ ] 花字 / 水印专用监督  
