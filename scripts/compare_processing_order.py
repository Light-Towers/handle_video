#!/usr/bin/env python3
"""
对比不同处理顺序的效果
"""

import cv2
import numpy as np
from pathlib import Path


def gaussian_denoise(image, strength=0.5):
    """高斯去噪"""
    kernel_size = int(5 * strength) * 2 + 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def sharpen_image(image, strength=0.3):
    """锐化图像"""
    kernel = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]]) * strength
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def unsharp_mask(image, sigma=1.0, strength=0.5):
    """Unsharp Mask 锐化（更好的锐化方法）"""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def compare_processing_orders(image_path, output_dir):
    """对比不同处理顺序"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 方法 1: 去噪 → 锐化
    print("方法 1: 去噪 → 锐化")
    denoised = gaussian_denoise(image, strength=0.5)
    sharpened = unsharp_mask(denoised, sigma=1.0, strength=0.5)
    cv2.imwrite(str(output_dir / "1_denoise_then_sharpen.jpg"), sharpened)

    # 方法 2: 锐化 → 去噪
    print("方法 2: 锐化 → 去噪")
    sharpened2 = unsharp_mask(image, sigma=1.0, strength=0.5)
    denoised2 = gaussian_denoise(sharpened2, strength=0.5)
    cv2.imwrite(str(output_dir / "2_sharpen_then_denoise.jpg"), denoised2)

    # 方法 3: 仅去噪
    print("方法 3: 仅去噪")
    denoised3 = gaussian_denoise(image, strength=0.5)
    cv2.imwrite(str(output_dir / "3_denoise_only.jpg"), denoised3)

    # 方法 4: 仅锐化
    print("方法 4: 仅锐化")
    sharpened4 = unsharp_mask(image, sigma=1.0, strength=0.5)
    cv2.imwrite(str(output_dir / "4_sharpen_only.jpg"), sharpened4)

    # 方法 5: NLMeans 去噪 → 锐化
    print("方法 5: NLMeans 去噪 → 锐化")
    denoised5 = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
    sharpened5 = unsharp_mask(denoised5, sigma=1.0, strength=0.5)
    cv2.imwrite(str(output_dir / "5_nlmeans_then_sharpen.jpg"), sharpened5)

    # 原图
    cv2.imwrite(str(output_dir / "0_original.jpg"), image)

    print(f"\n对比结果已保存到: {output_dir}")
    print("\n文件说明:")
    print("  0_original.jpg - 原图")
    print("  1_denoise_then_sharpen.jpg - 高斯去噪 → 锐化")
    print("  2_sharpen_then_denoise.jpg - 锐化 → 高斯去噪")
    print("  3_denoise_only.jpg - 仅高斯去噪")
    print("  4_sharpen_only.jpg - 仅锐化")
    print("  5_nlmeans_then_sharpen.jpg - NLMeans 去噪 → 锐化")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='对比不同处理顺序')
    parser.add_argument('input', type=str, help='输入图像路径')
    parser.add_argument('-o', '--output', type=str, default='output_comparison',
                       help='输出目录')

    args = parser.parse_args()

    compare_processing_orders(args.input, args.output)
