# 任务背景

电商营销视频关键帧需要**局部换装**：保留源帧人物姿势、构图与背景，将服装替换为目标商品，再交给视频模型做段间运动。

现网常用 GPT Image 2 + 长编辑指令。本仓库用 **Qwen-Image-Edit-2511** 在同一设定下做数据构建、SFT（LoRA / 规划中的全参 ZeRO-3）与评测。

### 数据字段

```text
edit_image[0] = 源关键帧
edit_image[1] = 商品参考
prompt        = 长编辑指令（全文 v2）
image         = 监督目标（合成 GT）
```

模型与训练原理见 [KNOWLEDGE.md](KNOWLEDGE.md)。
