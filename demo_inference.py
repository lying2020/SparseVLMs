#!/usr/bin/env python3
"""
SparseVLM 推理Demo脚本

这个脚本演示了如何使用SparseVLM进行图像-文本对话推理。

使用方法:
    python demo_inference.py \
        --model-path liuhaotian/llava-v1.5-7b \
        --image-file path/to/image.jpg \
        --query "What is in this image?" \
        --retain-tokens 128

参数说明:
    --model-path: LLaVA模型路径（HuggingFace路径或本地路径）
    --image-file: 输入图像路径
    --query: 问题文本
    --retain-tokens: 保留的视觉token数量（192/128/96/64，默认128）
    --use-version: 使用版本（1_0或2_0，默认1_0）
    --load-8bit: 使用8-bit量化（降低显存需求）
    --load-4bit: 使用4-bit量化（进一步降低显存需求）
"""

import argparse
import torch
import os
from PIL import Image

import project

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
    parser = argparse.ArgumentParser(
        description="SparseVLM推理Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 模型参数
    parser.add_argument(
        "--model-path",
        type=str,
        default=project.llava_v15_7b_path,
        help="LLaVA模型路径（HuggingFace路径或本地路径）"
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
        help="基础语言模型路径（如果使用LoRA或projector-only模型）"
    )

    # 输入参数
    parser.add_argument(
        "--image-file",
        type=str,
        default="./source_images/1.png",
        help="输入图像路径或URL"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Describe this image in detail.",
        help="问题或指令文本"
    )

    # SparseVLM参数
    parser.add_argument(
        "--retain-tokens",
        type=int,
        default=128,
        choices=[192, 128, 96, 64],
        help="保留的视觉token数量（192/128/96/64）"
    )
    parser.add_argument(
        "--use-version",
        type=str,
        default="1_0",
        choices=["1_0", "2_0"],
        help="SparseVLM版本（1_0=基础版，2_0=SparseVLM+）"
    )

    # 推理参数
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="生成温度（0=确定性，>0=随机性）"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p采样参数"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="最大生成token数量"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam search的beam数量"
    )

    # 设备参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备（cuda/cpu）"
    )
    parser.add_argument(
        "--load-bit",
        default=8,
        choices=[32, 16, 8, 4],
        help="量化位数（32/16/8/4）"
    )
    # 其他参数
    parser.add_argument(
        "--conv-mode",
        type=str,
        default=None,
        help="对话模式（自动推断，通常不需要指定）"
    )

    args = parser.parse_args()

    # 设置环境变量（SparseVLM关键参数）
    os.environ["RETAIN_TOKN"] = str(args.retain_tokens)
    os.environ["USE_VERSION"] = args.use_version

    print("=" * 60)
    print("SparseVLM 推理Demo")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"图像文件: {args.image_file}")
    print(f"问题: {args.query}")
    print(f"保留Token数: {args.retain_tokens}")
    print(f"版本: {args.use_version}")
    print(f"设备: {args.device}")
    load_8bit, load_4bit = False, False
    if args.load_bit == 8:
        print("量化: 8-bit")
        load_8bit = True
    elif args.load_bit == 4:
        print("量化: 4-bit")
        load_4bit = True
    print("=" * 60)

    # 禁用torch初始化优化
    disable_torch_init()

    # 加载模型
    print("\n[1/4] 加载模型...")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device=args.device,
    )
    print(f"✓ 模型加载完成 (context_len={context_len})")

    # 加载图像
    print("\n[2/4] 加载图像...")
    image = load_image(args.image_file)
    image_size = image.size
    print(f"✓ 图像加载完成 (尺寸: {image_size})")

    # 准备对话
    print("\n[3/4] 准备对话...")
    qs = args.query

    # 自动推断对话模式
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

    if args.conv_mode is not None:
        conv_mode = args.conv_mode

    # 添加图像token
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(f"✓ 对话准备完成 (模式: {conv_mode})")

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
    print("\n[4/4] 生成回答...")
    print("-" * 60)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # 提取回答（移除prompt部分）
    if conv.sep_style == SeparatorStyle.TWO:
        outputs = outputs.split(conv.sep2)[-1].strip()
    else:
        outputs = outputs.split(conv.sep)[-1].strip()

    print("\n" + "=" * 60)
    print("回答:")
    print("=" * 60)
    print(outputs)
    print("=" * 60)

    return outputs


if __name__ == "__main__":
    main()
