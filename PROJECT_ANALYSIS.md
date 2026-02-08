# SparseVLM 项目分析报告

## 项目概述

**SparseVLM** 是一个用于视觉语言模型（VLM）高效推理的项目。它通过**视觉token稀疏化**技术，在保持模型性能的同时显著减少计算开销。

### 核心特点
- **自适应稀疏化**：根据问题提示（question prompt）动态选择相关的视觉patches
- **支持多种token保留数量**：192, 128, 96, 64 tokens
- **兼容LLaVA架构**：基于LLaVA-1.5和LLaVA-1.6
- **两种模式**：V1.0（基础版）和V2.0（SparseVLM+，性能更强）

---

## 1. 模型需求

### 1.1 基础模型
项目需要下载**LLaVA预训练模型**，常用选项：

| 模型名称 | HuggingFace路径 | 参数量 | 显存需求（推理） |
|---------|----------------|--------|-----------------|
| LLaVA-1.5-7B | `liuhaotian/llava-v1.5-7b` | 7B | ~14GB |
| LLaVA-1.5-13B | `liuhaotian/llava-v1.5-13b` | 13B | ~26GB |
| LLaVA-1.6-7B | `liuhaotian/llava-v1.6-vicuna-7b` | 7B | ~14GB |
| LLaVA-1.6-13B | `liuhaotian/llava-v1.6-vicuna-13b` | 13B | ~26GB |

### 1.2 视觉编码器
- **CLIP ViT-L/14** (336px)：`openai/clip-vit-large-patch14-336`
- 模型会自动下载，无需手动配置

### 1.3 模型下载方式
```bash
# 方式1：使用HuggingFace CLI
huggingface-cli download liuhaotian/llava-v1.5-7b

# 方式2：代码中自动下载（首次运行时）
# 模型会缓存在 ~/.cache/huggingface/ 目录
```

---

## 2. 算力资源需求

### 2.1 推理（Inference）

#### 最低配置
- **GPU**：1张 NVIDIA GPU
- **显存**：
  - 7B模型：至少 **14GB**（FP16）
  - 13B模型：至少 **26GB**（FP16）
- **内存**：16GB+ RAM
- **存储**：每个模型约 **13-26GB** 磁盘空间

#### 推荐配置
- **GPU**：1-2张 NVIDIA A100/V100/RTX 3090/4090
- **显存**：24GB+（7B模型）或 40GB+（13B模型）
- **内存**：32GB+ RAM

#### 量化选项（降低显存需求）
- **8-bit量化**：显存需求减半
- **4-bit量化**：显存需求降至1/4
- 使用方式：在代码中设置 `load_8bit=True` 或 `load_4bit=True`

### 2.2 训练（Training）

#### 硬件需求
- **GPU**：多张GPU（通常4-8张）
- **显存**：每张GPU至少24GB（使用DeepSpeed ZeRO-2）
- **内存**：64GB+ RAM
- **存储**：100GB+（数据集+模型checkpoints）

#### 训练配置
- **框架**：DeepSpeed（分布式训练）
- **精度**：BF16/FP16混合精度
- **优化器**：AdamW
- **学习率**：2e-5（微调阶段）

#### 训练参数规模
- **Teacher模型**：完整的LLaVA模型（7B或13B）
- **Student模型**：需要训练的稀疏化模型
- **可训练参数**：主要训练视觉token选择机制，参数量相对较小

---

## 3. 训练参数配置

### 3.1 关键训练参数

从 `llava/train/sparse_train.py` 和训练脚本分析：

```python
# 模型参数
model_name_or_path: "llava-v1.5-7b"  # 基础模型路径
teacher_model_name_or_path: "llava-v1.5-7b"  # Teacher模型（用于知识蒸馏）

# 训练参数
num_train_epochs: 1  # 训练轮数
per_device_train_batch_size: 16  # 每设备batch size
gradient_accumulation_steps: 1  # 梯度累积
learning_rate: 2e-5  # 学习率
weight_decay: 0.0
warmup_ratio: 0.03
lr_scheduler_type: "cosine"

# 优化设置
bf16: True  # 混合精度
gradient_checkpointing: True  # 梯度检查点（节省显存）
model_max_length: 2048  # 最大序列长度
```

### 3.2 训练数据
- **预训练数据**：LCS-558K（558K图像-文本对）
- **微调数据**：LLaVA-Instruct-80K（80K指令数据）
- **数据格式**：JSON格式，包含图像路径和对话数据

### 3.3 训练流程
1. **预训练阶段**：训练视觉-语言投影层（mm_projector）
2. **微调阶段**：在指令数据上微调
3. **稀疏化训练**：使用Teacher-Student框架训练token选择机制

---

## 4. 环境配置

### 4.1 Python环境
```bash
conda create -n SparseVLMs python=3.10 -y
conda activate SparseVLMs
```

### 4.2 依赖安装
```bash
# 基础依赖
pip install -e .
pip install transformers==4.37.0
pip install flash_attn==2.3.3  # FlashAttention加速

# 训练相关（可选）
pip install deepspeed==0.12.6
pip install wandb  # 训练监控
```

### 4.3 系统要求
- **CUDA**：11.8+（支持FlashAttention 2）
- **PyTorch**：2.1.2
- **操作系统**：Linux（推荐Ubuntu 20.04+）

---

## 5. 快速开始

### 5.1 推理Demo
参考 `demo_inference.py` 脚本

### 5.2 评估Benchmark
```bash
# MME评估（保留192 tokens）
RETAIN_TOKN=192 bash scripts/v1_5/eval/mme.sh

# 启用V2.0模式
USE_VERSION=2_0 RETAIN_TOKN=192 bash scripts/v1_5/eval/mme.sh
```

---

## 6. 性能对比

### 6.1 Token保留数量影响
- **192 tokens**：性能最高，速度较慢
- **128 tokens**：性能与速度平衡
- **96 tokens**：速度较快，性能略有下降
- **64 tokens**：速度最快，性能下降明显

### 6.2 加速效果
- **计算量减少**：约50-80%（取决于保留token数量）
- **显存占用**：显著降低
- **推理速度**：提升2-4倍

---

## 7. 注意事项

1. **模型下载**：首次运行需要下载模型，可能需要较长时间
2. **显存管理**：如果显存不足，使用量化或减小batch size
3. **FlashAttention**：需要CUDA和兼容的GPU（A100/V100/RTX 3090+）
4. **数据准备**：评估需要下载对应的benchmark数据集

---

## 8. 常见问题

### Q: 如何选择模型大小？
A: 7B模型适合大多数场景，13B模型性能更好但需要更多资源。

### Q: 训练需要多长时间？
A: 取决于数据量和GPU数量，通常需要数天到数周。

### Q: 是否支持CPU推理？
A: 理论上支持，但速度极慢，不推荐。

### Q: 如何自定义token保留数量？
A: 修改 `llava/model/language_model/score.py` 中的配置。

