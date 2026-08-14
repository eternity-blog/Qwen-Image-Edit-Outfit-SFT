# 基于 Qwen-Image-Edit-2511 的电商商品图编辑 SFT 方案

## 1. 项目目标

目标是构建一个面向电商营销图片的 **Reference-Guided E-commerce Image Editing** 模型。

给定：

- 原始商品/营销图 `Original Image`
- 目标商品图 `Target Product Image`
- 可选目标模特图 `Target Person Image`
- 目标商品描述或卖点 `Product Description / Selling Points.md`

输出：

- 编辑后的目标营销图 `Target Image`

核心要求：

1. 将原图中的商品/服装替换为目标商品。
2. 如果提供目标模特图，则替换原模特。
3. 删除原图中的平台 Logo、水印、原商品品牌信息等。
4. 删除或替换原商品卖点、标题、价格等营销文字。
5. 根据目标商品 Markdown 中的信息生成新的商品卖点/营销文案。
6. **尽可能只修改需要修改的区域，保持原图其他内容不变**，包括背景、构图、摄影角度、非商品相关元素、光照等。

因此，该任务不应简单定义为 Virtual Try-On（VTON），而应定义为：

> **Reference-Guided E-commerce Image Editing**

其中 VTON 只是其中一个子能力。

---

# 2. 总体系统设计

最终系统可以拆成四个功能部分：

```text
                    User Inputs
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ↓               ↓                ↓
   Original Image   Target Product   Target Person
                                      (optional)
        │               │                │
        └───────────────┼────────────────┘
                        │
                Product Description
                     / Selling
                     Points.md
                        │
                        ↓
             ┌─────────────────────┐
             │ ① 内容理解与编辑规划 │
             │                     │
             │ Person / Product    │
             │ Text / Logo / Layout│
             └──────────┬──────────┘
                        │
                        ↓
             ┌─────────────────────┐
             │ ② 商品/模特编辑     │
             │                     │
             │ Garment Transfer    │
             │ Person Replacement  │
             └──────────┬──────────┘
                        │
                        ↓
             ┌─────────────────────┐
             │ ③ 文本/水印编辑     │
             │                     │
             │ Remove / Replace    │
             │ Slogan / Logo       │
             └──────────┬──────────┘
                        │
                        ↓
             ┌─────────────────────┐
             │ ④ 构图保持与质量控制 │
             │                     │
             │ Preserve Background │
             │ Preserve Layout     │
             │ Check Identity      │
             │ Check Product       │
             └──────────┬──────────┘
                        │
                        ↓
                   Final Image
```

但是在**第一阶段模型训练**时，不建议立即实现四个独立模型。

第一阶段采用：

> **Qwen-Image-Edit-2511 + LoRA/SFT，将四类能力统一为多图条件指令式图像编辑任务。**

---

# 3. Base Model：Qwen-Image-Edit-2511

## 3.1 模型定位

Qwen-Image-Edit-2511 作为本项目第一阶段的 Base Model。

它属于 **Diffusion / Flow-based Image Generation & Editing** 路线，并采用 Transformer/DiT 类架构，而不是传统 Stable Diffusion 的 U-Net 架构。

模型主要用于：

- 图像编辑
- 多图参考编辑
- 视觉内容保持
- 人物一致性
- 多人物一致性
- 商品/物体替换
- 局部编辑
- 文本相关编辑

其多图输入能力适合本项目的：

```text
Original Image
+
Target Product Image
+
Target Person Image（可选）
+
Text Instruction
```

---

# 4. 为什么不直接使用 VTON 作为 Base Model

CatVTON、IDM-VTON 等专用 VTON 模型非常适合：

```text
Person Image
+
Garment Image
→
Person Wearing Garment
```

但本项目除了换装之外，还需要：

- 换模特
- 删除平台水印
- 删除品牌 Logo
- 删除原商品卖点
- 生成新商品文案
- 保持原图构图
- 保持背景
- 保持非编辑区域

因此任务范围已经超过传统 VTON。

更合理的设计是：

```text
Qwen-Image-Edit-2511
        │
        ├── Garment Transfer
        ├── Person Replacement
        ├── Watermark Removal
        ├── Text Replacement
        └── Image Preservation
```

VTON 数据则作为其中的专项训练数据。

---

# 5. 统一任务建模

所有训练数据统一表示为：

```text
Input Images
+
Edit Instruction
→
Target Image
```

而不是简单：

```text
Source Image
→
Target Image
```

例如完整任务：

```text
Input:
    Original Image
    Target Product Image
    Target Person Image

Instruction:
    将原图中的商品替换为参考图中的目标商品。
    将原图中的模特替换为参考图中的模特。
    删除原图中的平台水印、原商品品牌信息和商品卖点。
    根据提供的商品描述更新商品相关文案。
    保持原图背景、构图、摄影角度和非商品相关区域不变。

Target:
    Edited Image
```

核心思想：

> 同时告诉模型 **What to Change** 和 **What to Preserve**。

---

# 6. 四类核心编辑任务

## 6.1 商品/服装替换

输入：

```text
Original Person Image
+
Target Garment/Product Image
```

输出：

```text
Person wearing Target Garment
```

Instruction 示例：

> 将人物身上的服装替换为参考图中的目标服装，保持人物身份、姿态、背景、构图和光照基本不变。

主要解决：

- Garment Transfer
- Product Transfer
- Clothing Texture Preservation
- Logo/Pattern Preservation

---

## 6.2 模特替换

输入：

```text
Original Image
+
Target Person Reference
```

输出：

```text
Target Person
+
Original Product/Scene
```

Instruction 示例：

> 将原图中的模特替换为参考图中的模特，保持原图商品、姿态、背景和构图尽可能不变。

需要训练模型学习：

- 人物身份保持
- 人脸保持
- 人体结构
- 姿态适配
- 服装适配

---

## 6.3 水印/Logo/商品信息擦除

输入：

```text
Original Image
```

Instruction：

> 删除原图中的平台 Logo 和水印，保持其他内容不变。

或：

> 删除原商品品牌 Logo 和商品相关文字，保持人物、商品主体和背景不变。

输出：

```text
Clean Image
```

该能力主要对应：

- Region Editing
- Object Removal
- Inpainting
- Text Removal

注意区分：

### 平台水印

例如：

- 平台 Logo
- 平台账号
- 平台标识

### 商品信息

例如：

- 商品品牌
- 商品名称
- 商品卖点
- 价格
- 原营销文案

二者在数据中应分别标注。

---

## 6.4 商品卖点/文字替换

输入：

```text
Original Image
+
Target Product Image
+
Product Description.md
```

例如：

```yaml
product:
  name: "轻量羽绒服"
  category: "羽绒服"

selling_points:
  - "800蓬高蓬松"
  - "轻量保暖"
  - "防泼水"

marketing_text:
  title: "轻盈保暖，一件过冬"
  subtitle: "800蓬高蓬松羽绒"

price:
  current: "¥699"
```

输出：

- 删除原商品相关文案
- 根据 Markdown 生成目标商品文案
- 尽可能保持原图文字布局

---

# 7. 数据体系

不建议寻找单一数据集解决全部问题。

建议构建四级数据体系。

```text
                    Training Data
                         │
       ┌─────────────────┼─────────────────┐
       ↓                 ↓                 ↓
 General Editing       VTON            Person Editing
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ↓
                  E-commerce Data
                         ↓
                    Final SFT
```

---

# 8. Level 1：通用 Image Editing 数据

主要目标：

> 让 Base Model 学习稳定的 instruction-based image editing。

推荐：

- UltraEdit
- ImgEdit
- MagicBrush
- InstructPix2Pix

其中重点关注：

## UltraEdit

UltraEdit 是大规模 image editing 数据集，包含约 400 万 editing samples，并包含 region-based editing 数据。

适合学习：

- instruction following
- image editing
- object editing
- region editing
- object removal
- object replacement

## ImgEdit

ImgEdit 的数据组织方式包含：

- single-turn editing
- multi-turn editing
- region-based editing

其数据构建 pipeline 也具有参考价值：

```text
Image
↓
VLM Caption
↓
Object Detection
↓
Segmentation
↓
Edit Instruction
↓
Image Editing
↓
Quality Filtering
```

可以直接参考其自动化数据构建思想。

---

# 9. Level 2：VTON / Clothing Editing 数据

推荐：

- VITON-HD
- DressCode
- DeepFashion2

主要目标：

> 学习 Garment Transfer。

统一转换成：

```text
Input:
    Person Image
    Garment Image

Instruction:
    将人物身上的服装替换为参考服装，
    保持人物身份、姿态、背景和构图不变。

Target:
    Try-on Image
```

不要直接使用原始 VTON 数据格式，而是转换成统一的：

```text
Images
+
Instruction
+
Target
```

---

# 10. Level 3：Person Replacement 数据

VTON 数据无法解决：

```text
Person A
+
Person B Reference
→
Person B
```

因此需要额外构建人物替换数据。

推荐形式：

```text
Input:
    Source Person
    Target Person Reference

Instruction:
    将原图中的模特替换为参考图中的模特，
    保持原商品、背景和构图尽可能不变。

Target:
    Edited Image
```

如果存在真实的同一场景/同一商品的不同模特图片，可以直接利用真实图片构造训练 pair。

---

# 11. Level 4：E-commerce Image Editing 数据

这是最终最重要的数据。

建议数据格式：

```json
{
  "source_image": "original.jpg",
  "product_image": "target_product.jpg",
  "person_image": "target_person.jpg",
  "product_description": "product.md",
  "instruction": "...",
  "target_image": "target.jpg",
  "edit_type": [
    "product_replace",
    "person_replace",
    "watermark_remove",
    "text_replace"
  ]
}
```

其中：

```text
person_image = null
```

表示不需要换模特。

---

# 12. 电商数据获取方案

不建议只依赖现成公开数据集。

更现实的方式是：

> **公开数据集打底 + 真实商品图片构造 + 自动生成编辑 pair。**

例如一个商品可能存在：

```text
Product A
├── Model A wearing A
├── Model B wearing A
├── Product-only Image
├── Detail Image
└── Marketing Image
```

可以利用这些图片构造：

```text
Source:
    Model A + Product A

Reference:
    Model B
    Product B

Target:
    Model B + Product B
```

如果目标图本身是真实商品营销图，则可以避免依赖生成模型生成 target。

核心原则：

> **尽可能使用真实 target image，而不是让另一个生成模型生成 target。**

---

# 13. Multi-reference 数据

Qwen-Image-Edit-2511 的多图输入非常适合本项目。

统一设计三种主要输入形式。

## Case A：只换商品

```text
Image 1 = Original
Image 2 = Target Product
```

```text
Original
+
Product
→
Target
```

---

## Case B：换商品 + 换模特

```text
Image 1 = Original
Image 2 = Target Product
Image 3 = Target Person
```

```text
Original
+
Product
+
Person
→
Target
```

---

## Case C：只编辑原图

```text
Image 1 = Original
```

例如：

```text
Remove watermark
```

或：

```text
Replace product slogan
```

---

# 14. Markdown 商品信息

不建议直接把 Markdown 原文作为任意长文本输入。

建议先标准化为结构化数据：

```yaml
product:
  name: "轻量羽绒服"
  brand: "XXX"
  category: "羽绒服"

selling_points:
  - "800蓬高蓬松"
  - "轻量保暖"
  - "防泼水"

marketing_text:
  title: "轻盈保暖，一件过冬"
  subtitle: "800蓬高蓬松羽绒"

price:
  current: "¥699"
```

再序列化成模型 Instruction。

这样可以控制：

- 必须出现的信息
- 可选信息
- 必须删除的信息
- 必须替换的信息

---

# 15. Instruction 数据设计

建议同时训练四种 Instruction。

## 简单指令

> 将原图中的商品替换为参考图中的商品。

## 约束指令

> 仅替换商品，保持人物身份、姿态、背景、构图和光照不变。

## 完整任务指令

> 将原图中的服装替换为参考商品，将原模特替换为参考模特，删除原有平台水印和商品卖点，并根据提供的商品信息更新文案。保持原图构图和背景不变。

## 强约束指令

> 除人物、商品及商品相关文字外，不修改原图其他区域。

核心原则：

> **Instruction 同时描述 What to Change 和 What to Preserve。**

---

# 16. Mask 数据

每条高质量电商数据建议额外保存：

```text
source.jpg
target.jpg

person_mask.png
product_mask.png
text_mask.png
watermark_mask.png
logo_mask.png
background_mask.png
```

以及：

```json
{
  "edit_regions": {
    "person": true,
    "product": true,
    "watermark": true,
    "product_text": true,
    "background": false
  }
}
```

第一阶段即使不把 mask 作为模型输入，也可以用于：

- 数据质量检查
- preservation loss
- 区域指标计算
- 局部编辑
- 数据过滤
- 后续 Mask-guided Editing

---

# 17. SFT / LoRA 训练策略

Qwen-Image-Edit-2511 是图像生成/编辑模型，因此这里的 SFT 不是 LLM 的 next-token prediction。

更准确地说：

> 对 Diffusion / Flow-based Transformer 图像生成模型进行 supervised fine-tuning，第一阶段采用 LoRA/PEFT。

不建议一开始 Full Fine-tuning 20B 模型。

推荐：

```text
Qwen-Image-Edit-2511
        ↓
      LoRA
        ↓
General Editing
+
VTON
+
Person Editing
+
E-commerce Editing
```

---

# 18. 两阶段训练

## Stage 0：Base Baseline

直接使用：

```text
Qwen-Image-Edit-2511
```

不微调。

测试：

- 换商品
- 换模特
- 擦水印
- 改文案
- 保持背景

得到 Base Model baseline。

---

## Stage 1：General Editing Adaptation

数据：

```text
UltraEdit
ImgEdit
MagicBrush
VITON-HD
DressCode
```

目标：

- instruction following
- multi-image editing
- garment transfer
- object replacement
- object removal

---

## Stage 2：E-commerce Domain SFT

使用高质量自构造电商数据：

```text
Stage 1 LoRA
      ↓
E-commerce Dataset
      ↓
High-quality SFT
      ↓
Final LoRA
```

重点学习：

- 商品替换
- 模特替换
- 水印删除
- 商品文字替换
- 构图保持
- 非编辑区域保持

---

# 19. 初始数据配比

第一版可以尝试：

| 数据类型 | 比例 | 目标 |
|---|---:|---|
| General Image Editing | 30% | 通用编辑能力 |
| VTON | 25% | 换装 |
| Person Replacement | 15% | 换模特 |
| Text / Watermark | 10% | 水印/文字 |
| E-commerce | 20% | 目标领域 |

随着自有电商数据质量提高，可以逐渐提高电商数据比例：

```text
General Editing     20%
VTON                20%
Person Editing      15%
Text/Watermark      10%
E-commerce          35%
```

---

# 20. 数据质量控制

训练数据质量非常重要。

建议：

```text
Raw Data
   ↓
Image Quality Filter
   ↓
Resolution Filter
   ↓
OCR / Object Detection
   ↓
Pair Consistency Check
   ↓
Instruction Validation
   ↓
Target Quality Check
   ↓
Duplicate Removal
   ↓
Human Sampling
   ↓
SFT Dataset
```

对于自动生成的数据，尤其需要过滤：

- 人物异常
- 手部异常
- 商品变形
- Logo 变形
- 文本乱码
- 背景发生非预期变化
- 目标商品不一致
- 模特身份错误

---

# 21. 评价指标

不能只使用 CLIP Score。

## 21.1 Product Fidelity

衡量目标商品与输出商品的一致性：

- CLIP similarity
- DINO similarity
- garment/product embedding similarity
- 商品局部区域相似度

---

## 21.2 Person Identity

如果不换模特：

```text
Source Person ↔ Output Person
```

如果换模特：

```text
Reference Person ↔ Output Person
```

可以使用：

- Face embedding similarity
- Person embedding similarity

---

## 21.3 Background Preservation

重点评估非编辑区域：

```text
Source Background
↔
Output Background
```

可以使用：

- LPIPS
- SSIM
- DINO similarity

最好排除编辑区域后计算。

---

## 21.4 Text Accuracy

使用 OCR：

```text
Output Image
    ↓
OCR
    ↓
Extracted Text
    ↓
Compare with Target Markdown
```

指标：

- Character Accuracy
- CER
- WER

例如目标：

```text
轻盈保暖
800蓬高蓬松
¥699
```

输出 OCR 应尽可能完全一致。

---

## 21.5 Edit Locality

这是本项目最重要的指标之一。

定义：

> 非编辑区域发生变化的程度。

```text
Change(Source, Output, Non-edit Region)
```

越小越好。

它直接衡量：

> **“尽可能只换商品，其他部分保持原图一样。”**

---

# 22. 最终训练路线

```text
                 Qwen-Image-Edit-2511
                          │
                          ↓
                ┌───────────────────┐
                │ Stage 0 Baseline  │
                │     No Training   │
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │ Stage 1 General   │
                │ Image Editing SFT │
                │                   │
                │ UltraEdit         │
                │ ImgEdit           │
                │ MagicBrush        │
                │ VITON/DressCode   │
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │ Stage 2 E-commerce│
                │      SFT/LoRA     │
                │                   │
                │ Product Replace   │
                │ Person Replace    │
                │ Watermark Remove  │
                │ Text Replace      │
                │ Layout Preserve   │
                └─────────┬─────────┘
                          ↓
                 Final E-commerce
                   Image Editor
```

---

# 23. 第一阶段建议的最小可行实验

不要一开始就构建完整四模块系统。

建议先完成一个 MVP：

### Input

```text
Original Image
+
Target Product Image
+
Target Person Image(optional)
+
Product.md
```

### Model

```text
Qwen-Image-Edit-2511
+
LoRA
```

### Training Data

```text
VITON-HD
+
DressCode
+
UltraEdit/ImgEdit
+
少量自构造电商数据
```

### Target

```text
Target Marketing Image
```

### 对比

```text
Qwen-Image-Edit-2511
vs
Qwen-Image-Edit-2511 + LoRA
```

重点验证：

1. 商品 fidelity 是否提升
2. 模特替换是否成功
3. 原图背景是否保持
4. 非编辑区域是否发生 drift
5. 商品文字是否正确
6. 平台水印是否能够删除

如果这个 MVP 成功，再进入：

```text
Mask-guided Editing
+
四阶段 Pipeline
+
Agent Orchestration
+
QA
```

---

# 24. 最终定位

本项目最终不是：

> “训练一个 VTON 模型”。

而是：

> **基于 Qwen-Image-Edit-2511 的 Reference-Guided E-commerce Image Editing 模型。**

能力拆分为：

```text
Product Replacement
        ↓
VTON / Product Transfer

Person Replacement
        ↓
Reference-based Person Editing

Watermark Removal
        ↓
Region-based Editing / Inpainting

Product Text Replacement
        ↓
Text-aware Image Editing

Image Preservation
        ↓
Minimal-edit / Locality Optimization
```

统一接口：

```text
Original Image
+
Target Product
+
Target Person (optional)
+
Product Description / Selling Points
+
Instruction
        ↓
Qwen-Image-Edit-2511 + LoRA
        ↓
Target E-commerce Image
```

后续再将上述能力拆成独立的四阶段 Pipeline，是比较自然的工程演进路线。
