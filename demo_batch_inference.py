#!/usr/bin/env python3
"""
SparseVLM 批量推理Demo脚本

这个脚本演示了如何对多张图像进行批量推理。

使用方法:
    python demo_batch_inference.py \
        --model-path liuhaotian/llava-v1.5-7b \
        --image-folder path/to/images \
        --questions questions.jsonl \
        --output results.jsonl \
        --retain-tokens 128
"""

import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


def load_image(image_file):
    """加载图像文件"""
    if image_file.startswith("http") or image_file.startswith("https"):
        import requests
        from io import BytesIO
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def main():
    parser = argparse.ArgumentParser(description="SparseVLM批量推理Demo")
    
    # 模型参数
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    
    # 输入参数
    parser.add_argument("--image-folder", type=str, required=True, help="图像文件夹路径")
    parser.add_argument("--questions", type=str, required=True, help="问题JSONL文件路径")
    parser.add_argument("--output", type=str, default="results.jsonl", help="输出结果文件")
    
    # SparseVLM参数
    parser.add_argument("--retain-tokens", type=int, default=128, choices=[192, 128, 96, 64])
    parser.add_argument("--use-version", type=str, default="1_0", choices=["1_0", "2_0"])
    
    # 推理参数
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1, help="批处理大小（注意：SparseVLM通常单张处理）")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["RETAIN_TOKN"] = str(args.retain_tokens)
    os.environ["USE_VERSION"] = args.use_version
    
    print("=" * 60)
    print("SparseVLM 批量推理")
    print("=" * 60)
    print(f"模型: {args.model_path}")
    print(f"图像文件夹: {args.image_folder}")
    print(f"问题文件: {args.questions}")
    print(f"输出文件: {args.output}")
    print(f"保留Token数: {args.retain_tokens}")
    print("=" * 60)
    
    # 加载模型
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    
    # 推断对话模式
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    
    # 读取问题
    questions = []
    with open(args.questions, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    print(f"\n加载了 {len(questions)} 个问题")
    
    # 处理每个问题
    results = []
    with open(args.output, "w", encoding="utf-8") as f_out:
        for item in tqdm(questions, desc="处理中"):
            image_file = item.get("image", item.get("image_file", ""))
            question = item.get("text", item.get("question", item.get("query", "")))
            question_id = item.get("question_id", item.get("id", len(results)))
            
            # 加载图像
            image_path = os.path.join(args.image_folder, image_file)
            if not os.path.exists(image_path):
                print(f"警告: 图像不存在 {image_path}")
                continue
            
            image = load_image(image_path)
            image_size = image.size
            
            # 准备对话
            qs = question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # 处理图像
            image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            
            # Tokenize
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .to(model.device)
            )
            
            # 生成回答
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # 提取回答
            if conv.sep_style == SeparatorStyle.TWO:
                answer = outputs.split(conv.sep2)[-1].strip()
            else:
                answer = outputs.split(conv.sep)[-1].strip()
            
            # 保存结果
            result = {
                "question_id": question_id,
                "image": image_file,
                "question": question,
                "answer": answer,
                "retain_tokens": args.retain_tokens,
                "version": args.use_version,
            }
            results.append(result)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()
    
    print(f"\n完成! 结果已保存到 {args.output}")
    print(f"处理了 {len(results)} 个问题")


if __name__ == "__main__":
    main()

