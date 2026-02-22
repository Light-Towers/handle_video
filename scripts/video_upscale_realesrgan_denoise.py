#!/usr/bin/env python3
"""
Real-ESRGAN 视频超分辨率 + 去噪处理
支持降噪强度调整
"""

import cv2
import torch
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import argparse
import time
import os
from pathlib import Path


class RealESRGANDenoiser:
    def __init__(self, scale=4, model_name='RealESRGAN_x4plus', denoise_strength=0.5,
                 use_cuda=True, tile_size=512):
        """
        初始化 Real-ESRGAN 视频处理器（支持去噪）

        Args:
            scale: 放大倍数 (2, 4)
            model_name: 模型名称
            denoise_strength: 去噪强度 (0-1)，越高去噪越强
            tile_size: 分块大小
        """
        self.scale = scale
        self.denoise_strength = denoise_strength
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.tile_size = tile_size

        # 选择模型路径
        script_dir = Path(__file__).parent.parent
        if model_name == 'RealESRGAN_x4plus':
            model_path = str(script_dir / 'models' / 'RealESRGAN_x4plus.pth')
            netscale = 4
            num_in_ch = 3
            num_out_ch = 3
            num_feat = 64
            num_block = 23
            num_grow_ch = 32
        elif model_name == 'RealESRGAN_x2plus':
            model_path = str(script_dir / 'models' / 'RealESRGAN_x2plus.pth')
            netscale = 2
            num_in_ch = 3
            num_out_ch = 3
            num_feat = 64
            num_block = 23
            num_grow_ch = 32
        elif model_name == 'RealESRGAN_x4plus_anime_6B':
            model_path = str(script_dir / 'models' / 'RealESRGAN_x4plus_anime_6B.pth')
            netscale = 4
            num_in_ch = 3
            num_out_ch = 3
            num_feat = 64
            num_block = 6
            num_grow_ch = 32
        elif model_name == 'realesr-animevideov3':
            model_path = str(script_dir / 'models' / 'realesr-animevideov3.pth')
            netscale = 4
            num_in_ch = 3
            num_out_ch = 3
            num_feat = 64
            num_conv = 16
            use_srvgg = True
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        # Fallback
        if not Path(model_path).exists():
            fallback_path = str(script_dir / '.realesrgan' / f'{model_name}.pth')
            if Path(fallback_path).exists():
                model_path = fallback_path
            else:
                raise FileNotFoundError(f"找不到模型文件: {model_path} 和 {fallback_path}")

        # 初始化模型
        if 'use_srvgg' in locals() and use_srvgg:
            # realesr-animevideov3 使用 SRVGGNetCompact
            model = SRVGGNetCompact(
                num_in_ch=num_in_ch,
                num_out_ch=num_out_ch,
                num_feat=num_feat,
                num_conv=num_conv,
                upscale=netscale,
                act_type='prelu'
            )
        else:
            # 其他模型使用 RRDBNet
            model = RRDBNet(
                num_in_ch=num_in_ch,
                num_out_ch=num_out_ch,
                num_feat=num_feat,
                num_block=num_block,
                num_grow_ch=num_grow_ch,
                scale=netscale
            )

        # 创建 Real-ESRGAN 处理器
        actual_tile = self.tile_size if self.tile_size > 0 else 0
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=actual_tile,
            tile_pad=10,
            pre_pad=0,
            half=False,  # 禁用 FP16 避免精度损失
            device=torch.device('cuda' if self.use_cuda else 'cpu')
        )

        print(f"Real-ESRGAN 初始化完成")
        print(f"  模型: {model_name}")
        print(f"  放大倍数: {netscale}")
        print(f"  去噪强度: {denoise_strength}")
        print(f"  设备: {'CUDA' if self.use_cuda else 'CPU'}")

    def denoise_frame(self, frame):
        """单帧去噪"""
        # 方法1: 使用高斯模糊去噪
        if self.denoise_strength > 0:
            # 计算高斯核大小
            kernel_size = int(5 * self.denoise_strength) * 2 + 1  # 3, 5, 7, 9, 11
            denoised = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            return denoised
        return frame

    def sharpen_frame(self, frame, strength=0.3):
        """锐化图像"""
        kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]]) * strength
        sharpened = cv2.filter2D(frame, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def process_frame(self, frame):
        """处理单帧图像（去噪 + 超分辨率）"""
        # 转换为 RGB（RealESRGAN 期望 RGB 输入）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 先去噪
        denoised = self.denoise_frame(frame_rgb)

        # 超分辨率
        output, _ = self.upsampler.enhance(denoised, outscale=self.scale)

        # 转换回 BGR（OpenCV 期望 BGR 输出）
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output_bgr

    def process_video(self, input_path, output_path, show_progress=True):
        """处理视频"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n视频信息:")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps} FPS")
        print(f"  总帧数: {frame_count}")

        # 创建输出视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_width = width * self.scale
        out_height = height * self.scale
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        processed_frames = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            enhanced = self.process_frame(frame)
            out.write(enhanced)

            processed_frames += 1

            if show_progress and processed_frames % 50 == 0:
                elapsed = time.time() - start_time
                progress = processed_frames / frame_count * 100
                fps_current = processed_frames / elapsed if elapsed > 0 else 0
                print(f"\r进度: {progress:.1f}% ({processed_frames}/{frame_count}) | FPS: {fps_current:.1f}",
                      end='', flush=True)

        cap.release()
        out.release()

        elapsed_total = time.time() - start_time
        avg_fps = processed_frames / elapsed_total if elapsed_total > 0 else 0

        print(f"\n\n处理完成!")
        print(f"  输出: {output_path}")
        print(f"  输出分辨率: {out_width}x{out_height}")
        print(f"  平均 FPS: {avg_fps:.2f}")
        print(f"  总耗时: {elapsed_total:.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Real-ESRGAN 视频超分辨率 + 去噪')
    parser.add_argument('input', type=str, help='输入视频路径')
    parser.add_argument('-o', '--output', type=str, help='输出视频路径')
    parser.add_argument('-s', '--scale', type=int, default=4,
                       choices=[2, 4], help='放大倍数 (2, 4)')
    parser.add_argument('-m', '--model', type=str, default='RealESRGAN_x4plus',
                       choices=['RealESRGAN_x4plus', 'RealESRGAN_x2plus',
                               'RealESRGAN_x4plus_anime_6B', 'realesr-animevideov3'],
                       help='模型名称')
    parser.add_argument('--no-cuda', action='store_true', help='禁用 CUDA')
    parser.add_argument('--tile-size', type=int, default=512,
                       choices=[0, 256, 512, 768, 1024],
                       help='分块大小')
    parser.add_argument('--denoise', type=float, default=0.5,
                       help='去噪强度 (0-1, 0=不去噪, 1=强去噪)')
    parser.add_argument('--no-sharpen', action='store_true', help='禁用后处理锐化')

    args = parser.parse_args()

    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_denoised_{args.scale}x{input_path.suffix}")

    # 创建处理器
    denoiser = RealESRGANDenoiser(
        scale=args.scale,
        model_name=args.model,
        denoise_strength=args.denoise,
        use_cuda=not args.no_cuda,
        tile_size=args.tile_size
    )

    # 处理视频
    denoiser.process_video(args.input, args.output)


if __name__ == '__main__':
    main()
