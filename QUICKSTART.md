# SparseVLM 快速开始指南

## 📋 目录
1. [环境准备](#环境准备)
2. [模型下载](#模型下载)
3. [单张图像推理Demo](#单张图像推理demo)
4. [批量推理Demo](#批量推理demo)
5. [评估Benchmark](#评估benchmark)
6. [常见问题](#常见问题)

---

## 环境准备

### 1. 创建Conda环境
```bash
conda create -n SparseVLMs python=3.10 -y
conda activate SparseVLMs
```

### 2. 安装依赖
```bash
# 进入项目目录
cd /home/liying/Desktop/ECCV_2026/SparseVLMs

# 安装基础包
pip install -e .
pip install transformers==4.37.0

# 安装FlashAttention（可选，但推荐）
pip install flash_attn==2.3.3
```

**注意**: FlashAttention需要CUDA 11.8+和兼容的GPU（A100/V100/RTX 3090+）

---

## 模型下载

### 方式1: 使用HuggingFace CLI（推荐）
```bash
# 安装huggingface-cli
pip install huggingface_hub

# 下载7B模型（约13GB）
huggingface-cli download liuhaotian/llava-v1.5-7b

# 或下载13B模型（约26GB）
huggingface-cli download liuhaotian/llava-v1.5-13b
```

### 方式2: 代码自动下载
首次运行时，模型会自动从HuggingFace下载并缓存到 `~/.cache/huggingface/`

### 方式3: 使用Git LFS
```bash
git lfs install
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
```

---

## 单张图像推理Demo

### 基本用法
```bash
python demo_inference.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file llava/serve/examples/extreme_ironing.jpg \
    --query "What is unusual about this image?" \
    --retain-tokens 128
```

### 使用SparseVLM+ (V2.0)
```bash
python demo_inference.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file llava/serve/examples/extreme_ironing.jpg \
    --query "What is unusual about this image?" \
    --retain-tokens 128 \
    --use-version 2_0
```

### 使用量化（降低显存需求）
```bash
# 8-bit量化（显存需求减半）
python demo_inference.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file llava/serve/examples/extreme_ironing.jpg \
    --query "Describe this image." \
    --retain-tokens 128 \
    --load-8bit

# 4-bit量化（显存需求降至1/4）
python demo_inference.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file llava/serve/examples/extreme_ironing.jpg \
    --query "Describe this image." \
    --retain-tokens 128 \
    --load-4bit
```

### 不同Token保留数量
```bash
# 192 tokens（性能最高）
python demo_inference.py --retain-tokens 192 ...

# 128 tokens（推荐，性能与速度平衡）
python demo_inference.py --retain-tokens 128 ...

# 96 tokens（速度较快）
python demo_inference.py --retain-tokens 96 ...

# 64 tokens（速度最快）
python demo_inference.py --retain-tokens 64 ...
```

---

## 批量推理Demo

### 准备问题文件
创建 `questions.jsonl` 文件，格式如下：
```json
{"question_id": 1, "image": "image1.jpg", "text": "What is in this image?"}
{"question_id": 2, "image": "image2.jpg", "text": "Describe the scene."}
{"question_id": 3, "image": "image3.jpg", "text": "What color is the sky?"}
```

### 运行批量推理
```bash
python demo_batch_inference.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder /path/to/images \
    --questions questions.jsonl \
    --output results.jsonl \
    --retain-tokens 128
```

---

## 评估Benchmark

### MME评估
```bash
# 基础版本（保留192 tokens）
RETAIN_TOKN=192 bash scripts/v1_5/eval/mme.sh

# SparseVLM+版本
USE_VERSION=2_0 RETAIN_TOKN=192 bash scripts/v1_5/eval/mme.sh
```

### TextVQA评估
```bash
RETAIN_TOKN=128 bash scripts/v1_5/eval/textvqa.sh
```

### ScienceQA评估
```bash
RETAIN_TOKN=96 bash scripts/v1_5/eval/sqa.sh
```

### MMBench评估
```bash
RETAIN_TOKN=64 bash scripts/v1_5/eval/mmbench.sh
```

**注意**: 评估前需要先下载对应的benchmark数据集，参考 [LLaVA-Evaluation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)

---

## 常见问题

### Q1: 显存不足怎么办？
**A**: 使用量化选项：
- `--load-8bit`: 8-bit量化，显存减半
- `--load-4bit`: 4-bit量化，显存降至1/4

或使用更小的模型（7B而不是13B）

### Q2: 模型下载很慢怎么办？
**A**: 
1. 使用HuggingFace镜像站点
2. 使用Git LFS直接克隆
3. 手动下载模型文件

### Q3: FlashAttention安装失败？
**A**: 
- 确保CUDA版本 >= 11.8
- 确保GPU支持（A100/V100/RTX 3090+）
- 可以跳过FlashAttention，使用标准attention（性能略低）

### Q4: 如何选择Token保留数量？
**A**: 
- **192**: 追求最高性能
- **128**: 推荐，性能与速度平衡
- **96**: 速度优先
- **64**: 极致速度

### Q5: V1.0和V2.0有什么区别？
**A**: 
- **V1.0**: 基础版本
- **V2.0 (SparseVLM+)**: 改进的文本-视觉注意力模式，性能更强

### Q6: 支持哪些模型？
**A**: 
- LLaVA-1.5-7B/13B
- LLaVA-1.6-7B/13B
- 其他基于LLaVA架构的模型

### Q7: 如何自定义Token保留数量？
**A**: 修改 `llava/model/language_model/score.py` 中的配置

---

## 性能参考

### 推理速度（7B模型，A100 GPU）
- **192 tokens**: ~2-3 tokens/秒
- **128 tokens**: ~3-4 tokens/秒
- **96 tokens**: ~4-5 tokens/秒
- **64 tokens**: ~5-6 tokens/秒

### 显存占用（7B模型，FP16）
- **标准推理**: ~14GB
- **8-bit量化**: ~7GB
- **4-bit量化**: ~4GB

---

## 下一步

1. 查看 [PROJECT_ANALYSIS.md](./PROJECT_ANALYSIS.md) 了解详细的项目分析
2. 查看 [README.md](./README.md) 了解项目概述
3. 查看 `docs/` 目录了解更多文档

