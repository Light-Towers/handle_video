#!/usr/bin/env python3
"""
测试 RealESRGAN anime_6B 模型单帧处理
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import sys

# 添加 realesrgan 路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def test_anime_model(image_path, output_path, model_path):
    """测试 anime_6B 模型"""
    
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print(f"输入图片: {img.shape}, dtype: {img.dtype}")
    print(f"像素范围: [{img.min()}, {img.max()}]")
    
    # 创建模型
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                    num_block=6, num_grow_ch=32, scale=4)
    
    # 创建 upsampler (禁用 FP16)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False  # 禁用 FP16
    )
    
    # 方法1: 直接处理 (BGR -> BGR)
    print("\n方法1: 直接处理 (BGR)")
    output1, _ = upsampler.enhance(img, outscale=4)
    print(f"输出: {output1.shape}, dtype: {output1.dtype}")
    print(f"像素范围: [{output1.min()}, {output1.max()}]")
    cv2.imwrite(output_path.replace('.jpg', '_method1_bgr.jpg'), output1)
    
    # 方法2: RGB 转换处理
    print("\n方法2: RGB 转换")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output2, _ = upsampler.enhance(img_rgb, outscale=4)
    output2_bgr = cv2.cvtColor(output2, cv2.COLOR_RGB2BGR)
    print(f"输出: {output2_bgr.shape}, dtype: {output2_bgr.dtype}")
    print(f"像素范围: [{output2_bgr.min()}, {output2_bgr.max()}]")
    cv2.imwrite(output_path.replace('.jpg', '_method2_rgb.jpg'), output2_bgr)
    
    # 方法3: 归一化处理
    print("\n方法3: 归一化到 [0,1]")
    img_norm = img.astype(np.float32) / 255.0
    output3, _ = upsampler.enhance((img_norm * 255).astype(np.uint8), outscale=4)
    print(f"输出: {output3.shape}, dtype: {output3.dtype}")
    print(f"像素范围: [{output3.min()}, {output3.max()}]")
    cv2.imwrite(output_path.replace('.jpg', '_method3_norm.jpg'), output3)
    
    print(f"\n所有结果已保存到: {Path(output_path).parent}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/workspace/test_frame.jpg')
    parser.add_argument('--output', type=str, default='/workspace/test_output.jpg')
    parser.add_argument('--model', type=str, 
                       default='/workspace/handle_video/models/RealESRGAN_x4plus_anime_6B.pth')
    args = parser.parse_args()
    
    test_anime_model(args.input, args.output, args.model)
