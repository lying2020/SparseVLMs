# SparseVLM 训练细节：参数数量与输入输出

## 📋 目录
1. [训练参数数量](#训练参数数量)
2. [模型输入输出格式](#模型输入输出格式)
3. [训练流程详解](#训练流程详解)
4. [代码示例](#代码示例)

---

## 1. 训练参数数量

### 1.1 参数冻结策略

SparseVLM支持多种训练模式，根据配置不同，可训练参数数量差异很大：

#### **模式1: 全参数微调（Full Fine-tuning）**
- **可训练参数**: 整个模型的所有参数（约7B或13B）
- **配置**: `freeze_backbone=False`（默认）
- **参数量**:
  - LLaVA-1.5-7B: **~7B参数**
  - LLaVA-1.5-13B: **~13B参数**
- **适用场景**: 完整训练，效果最好但资源消耗最大

#### **模式2: 仅训练投影层（Projector-only）**
- **可训练参数**: 仅`mm_projector`（视觉-语言投影层）
- **配置**: `tune_mm_mlp_adapter=True`
- **参数量**: 约**2-4M参数**（取决于投影层配置）
- **适用场景**: 预训练阶段，资源消耗最小

#### **模式3: 冻结主干网络（Freeze Backbone）**
- **可训练参数**: 除LLM主干外的所有参数
- **配置**: `freeze_backbone=True`
- **参数量**: 约**10-50M参数**（主要是投影层和注意力机制相关）
- **适用场景**: 快速微调，保持LLM权重不变

#### **模式4: LoRA微调**
- **可训练参数**: LoRA适配器参数
- **配置**: `lora_enable=True`, `lora_r=64`, `lora_alpha=16`
- **参数量**: 约**10-100M参数**（取决于LoRA rank）
- **适用场景**: 参数高效微调

### 1.2 Teacher模型

- **Teacher模型**: 完全冻结（`requires_grad=False`）
- **用途**: 提供知识蒸馏的监督信号
- **参数量**: 与Student模型相同（7B或13B），但不参与梯度更新

### 1.3 实际训练参数统计

```python
# 代码位置: llava/train/sparse_train.py

# Teacher模型冻结
teacher_model.model.requires_grad_(False)  # 第863行

# 可选：冻结主干网络
if model_args.freeze_backbone:
    model.model.requires_grad_(False)  # 第866行

# 可选：仅训练投影层
if model_args.tune_mm_mlp_adapter:
    model.requires_grad_(False)
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True  # 第954-955行
```

### 1.4 参数数量估算

| 训练模式 | 7B模型可训练参数 | 13B模型可训练参数 | 显存需求（单卡） |
|---------|-----------------|------------------|-----------------|
| 全参数微调 | ~7B | ~13B | 40GB+ |
| 冻结主干 | ~50M | ~100M | 20GB+ |
| 仅投影层 | ~4M | ~8M | 15GB+ |
| LoRA (r=64) | ~50M | ~100M | 18GB+ |

**注意**: 实际显存需求还取决于batch size、序列长度等因素。

---

## 2. 模型输入输出格式

### 2.1 训练输入（Training Input）

#### **数据格式**
```python
# 输入字典结构
inputs = {
    'input_ids': torch.LongTensor,      # [B, L] Token IDs
    'images': torch.FloatTensor,         # [B, 3, H, W] 或 List[Tensor]
    'attention_mask': torch.BoolTensor,  # [B, L] 注意力掩码
    'labels': torch.LongTensor,          # [B, L] 标签（用于计算loss）
    'image_sizes': List[List[int]],      # 图像尺寸列表
}
```

#### **具体维度示例**
```python
# 示例：单个样本
input_ids: torch.Size([1, 668])           # 668个token
images: torch.Size([1, 3, 336, 336])       # 单张图像
attention_mask: torch.Size([1, 668])      # 注意力掩码
labels: torch.Size([1, 668])              # 标签（IGNORE_INDEX用于mask）

# 经过prepare_sparse_inputs_labels_for_multimodal处理后
inputs_embeds: torch.Size([1, 668, 4096]) # 嵌入向量（包含视觉特征）
```

#### **数据预处理流程**
1. **图像编码**: 图像 → CLIP视觉编码器 → 视觉特征 `[B, 576, 4096]`（576个视觉token）
2. **文本Tokenization**: 文本 → Tokenizer → `input_ids`
3. **特征融合**: 将视觉特征插入到文本token序列中
4. **最终输入**: `inputs_embeds` = 文本嵌入 + 视觉嵌入

### 2.2 模型前向传播（Forward Pass）

#### **输入处理**
```python
# 代码位置: llava/model/llava_arch.py:325-509

def prepare_sparse_inputs_labels_for_multimodal(
    self, input_ids, position_ids, attention_mask, past_key_values, labels,
    images, image_sizes=None
):
    # 1. 编码图像
    image_features = self.encode_images(images)  # [B, 576, 4096]
    
    # 2. 融合文本和视觉特征
    new_input_embeds = []  # 包含文本嵌入和视觉嵌入
    
    # 3. 返回处理后的输入
    return (
        None,                    # input_ids (设为None，使用inputs_embeds)
        position_ids,           # [B, L]
        attention_mask,         # [B, L]
        past_key_values,        # KV cache
        new_input_embeds,        # [B, L, 4096] 融合后的嵌入
        new_labels,              # [B, L]
        image_shape,             # 576 (视觉token数量)
        token_length_list,       # 每个样本的实际长度
        pre_prompt_length_list,  # prompt长度列表
    )
```

#### **模型输出**
```python
# 代码位置: llava/model/language_model/modelling_sparse_llama.py:384-391

# 训练时输出
outputs = (
    prev_decision,              # torch.Tensor [B, L] Token选择决策（0/1）
    out_pred_prob,                # 预测概率（如果使用）
    BaseModelOutputWithPast(
        last_hidden_state=hidden_states,  # [B, L_new, 4096] 稀疏化后的隐藏状态
        past_key_values=next_cache,       # KV cache
        hidden_states=all_hidden_states,  # 所有层的隐藏状态
        attentions=all_self_attns,        # 注意力权重
    )
)
```

### 2.3 损失计算（Loss Computation）

#### **损失函数组成**
```python
# 代码位置: llava/train/sparse_llava_trainer.py:430-492

def compute_loss(self, model, inputs, return_outputs=False):
    # 1. Student模型前向传播
    sparse_outputs = model(**inputs)
    prev_decision, out_pred_prob, outputs = sparse_outputs
    hidden_states = outputs.hidden_states  # Student的隐藏状态
    
    # 2. Teacher模型前向传播（无梯度）
    with torch.no_grad():
        teacher_outputs = self.teacher_model(**inputs)
        teacher_hidden_states = teacher_outputs.hidden_states[-1]
    
    # 3. 特征对齐损失（Feature Alignment Loss）
    B, L, C = hidden_states.shape
    bool_mask = prev_decision.reshape(B*L) > 0.5  # 保留的token
    hidden_states = hidden_states.reshape(B*L, C)
    teacher_hidden_states = teacher_hidden_states.reshape(B*L, C)
    
    # 只对齐保留的token
    hidden_states = hidden_states[bool_mask]
    teacher_hidden_states = teacher_hidden_states[bool_mask]
    align_loss = torch.pow(hidden_states - teacher_hidden_states, 2).mean()
    
    # 4. 语言建模损失（Language Modeling Loss）
    if labels is not None:
        loss = self.label_smoother(outputs, labels, shift_labels=True)
    
    # 5. 总损失（可选：添加align_loss）
    # total_loss = loss + alpha * align_loss
    
    return (total_loss, outputs) if return_outputs else total_loss
```

#### **损失函数类型**
1. **语言建模损失（LM Loss）**: 标准的交叉熵损失，用于预测下一个token
2. **特征对齐损失（Alignment Loss）**: L2距离，对齐Student和Teacher的隐藏状态
3. **可选：预测器损失（Predictor Loss）**: 控制保留token的比例

---

## 3. 训练流程详解

### 3.1 训练步骤

```
1. 数据加载
   ↓
2. 图像编码（CLIP Vision Encoder）
   ↓
3. 文本Tokenization
   ↓
4. 特征融合（文本嵌入 + 视觉嵌入）
   ↓
5. Student模型前向传播（带稀疏化）
   ↓
6. Teacher模型前向传播（无梯度）
   ↓
7. 计算损失（LM Loss + Alignment Loss）
   ↓
8. 反向传播（仅更新Student参数）
   ↓
9. 参数更新
```

### 3.2 稀疏化机制

#### **Token选择过程**
```python
# 代码位置: llava/model/language_model/modelling_sparse_llama.py:221-320

# 在特定层（pruning_loc）进行稀疏化
if layer_idx in self.pruning_loc:  # 通常是第2, 6, 15层
    # 1. 计算文本-视觉注意力
    attn_logits = layer_outputs[2]  # [B, H, L, L]
    
    # 2. 提取文本到视觉的注意力权重
    relation_vis_text = attn_logits[:, text_token_idx, v_token_start:v_token_end]
    
    # 3. Top-K选择
    _, indices = torch.topk(relation_vis_text, k=retain_tokens, dim=1)
    
    # 4. 创建保留mask
    policy = torch.zeros(B, L)
    policy[:, indices] = 1  # 保留的token标记为1
    
    # 5. 应用mask，只保留选中的token
    selected_hidden_states = hidden_states[policy == 1]
```

#### **稀疏化位置**
- **层位置**: 第2, 6, 15层（可配置）
- **Token保留数量**: 192, 128, 96, 64（可配置）
- **保留策略**: 基于文本-视觉注意力权重进行Top-K选择

---

## 4. 代码示例

### 4.1 训练脚本示例

```python
# 训练配置
python llava/train/sparse_train.py \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --teacher_model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path ./data/llava_instruct_80k.json \
    --image_folder ./data/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --freeze_backbone False \  # 全参数微调
    --tune_mm_mlp_adapter False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token True \
    --bf16 True \
    --output_dir ./checkpoints/sparsevlm-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing True \
    --model_max_length 2048 \
    --deepspeed ./scripts/zero2.json
```

### 4.2 数据格式示例

```json
// llava_instruct_80k.json
[
  {
    "id": "1",
    "image": "coco/train2017/000000123456.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat is in this image?"
      },
      {
        "from": "gpt",
        "value": "This image shows a cat sitting on a windowsill."
      }
    ]
  }
]
```

### 4.3 检查可训练参数

```python
# 统计可训练参数数量
def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

trainable, total = count_trainable_parameters(model)
print(f"可训练参数: {trainable/1e9:.2f}B / 总参数: {total/1e9:.2f}B")
```

---

## 5. 关键要点总结

### 5.1 训练参数
- **全参数微调**: ~7B或~13B参数
- **冻结主干**: ~50-100M参数
- **仅投影层**: ~4-8M参数
- **LoRA**: ~50-100M参数

### 5.2 输入格式
- **input_ids**: `[B, L]` Token序列
- **images**: `[B, 3, 336, 336]` 图像
- **inputs_embeds**: `[B, L, 4096]` 融合后的嵌入（文本+视觉）

### 5.3 输出格式
- **prev_decision**: `[B, L]` Token选择决策
- **hidden_states**: `[B, L_new, 4096]` 稀疏化后的隐藏状态（L_new < L）
- **loss**: 语言建模损失 + 特征对齐损失

### 5.4 训练特点
- **Teacher-Student框架**: 使用知识蒸馏
- **动态稀疏化**: 根据问题自适应选择视觉token
- **多层稀疏化**: 在第2, 6, 15层进行token选择
- **特征对齐**: 对齐Student和Teacher的隐藏状态

---

## 6. 参考代码位置

- **训练脚本**: `llava/train/sparse_train.py`
- **Trainer类**: `llava/train/sparse_llava_trainer.py`
- **模型架构**: `llava/model/language_model/sparse_llava_llama.py`
- **输入处理**: `llava/model/llava_arch.py:325-509`
- **稀疏化逻辑**: `llava/model/language_model/modelling_sparse_llama.py:221-320`

