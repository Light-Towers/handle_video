#!/usr/bin/env python3
"""
RIFE 视频补帧脚本
使用 RIFE 模型进行视频帧插值
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
import time
import urllib.request


class RIFEInterpolator:
    """RIFE 帧插值器"""

    def __init__(self, model_path=None, use_cuda=True):
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path):
        """加载 RIFE 模型"""
        try:
            # 动态导入 RIFE 模型（需要先下载 RIFE 源码）
            sys_path = Path(__file__).parent.parent / 'RIFE'
            if sys_path.exists():
                import sys
                sys.path.insert(0, str(sys_path))
                from model.RIFE_HD import Model
                self.model = Model()
                self.model.load_model(model_path, -1)
                self.model.eval()
                self.model.device()
                print(f"RIFE 模型已加载: {model_path}")
            else:
                print("警告: RIFE 源码未找到，请下载到 handle_video/RIFE 目录")
                print("下载地址: https://github.com/hzwer/arXiv2020-RIFE")
        except Exception as e:
            print(f"加载 RIFE 模型失败: {e}")

    def interpolate(self, frame1, frame2):
        """在两帧之间插值"""
        if self.model is None:
            # 降级方案：简单线性插值
            return cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)

        # RIFE 插值
        with torch.no_grad():
            img1 = torch.from_numpy(frame1).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            img2 = torch.from_numpy(frame2).permute(2, 0, 1).float().unsqueeze(0) / 255.0

            img1 = img1.to(self.device)
            img2 = img2.to(self.device)

            output = self.model.inference(img1, img2)
            output = (output[0] * 255.0).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

            return output


def simple_interpolate_video(input_path, output_path, target_fps, use_rife=False, model_path=None):
    """
    视频补帧

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        target_fps: 目标帧率
        use_rife: 是否使用 RIFE 模型
        model_path: RIFE 模型路径
    """
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {input_path}")

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"输入视频: {width}x{height}, {fps} FPS, {frame_count} 帧, {duration:.1f} 秒")

    # 计算插帧倍数
    factor = int(np.ceil(target_fps / fps))
    print(f"插帧倍数: {factor} ({fps} → {fps * factor} FPS)")

    # 初始化插值器
    if use_rife and model_path:
        interpolator = RIFEInterpolator(model_path, use_cuda=True)
        print(f"使用 RIFE 模型进行插值")
    else:
        interpolator = RIFEInterpolator()
        print(f"使用简单线性插值（未加载 RIFE 模型）")

    # 准备输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = min(fps * factor, target_fps)
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    # 逐帧处理，避免内存溢出
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("无法读取视频帧")

    total_inserted = 0
    total_output = 0
    start_time = time.time()

    frame_idx = 0
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            # 写入最后一帧
            out.write(prev_frame)
            total_output += 1
            break

        # 写入前一帧
        out.write(prev_frame)
        total_output += 1

        # 在两帧之间插入 factor-1 帧
        for j in range(1, factor):
            alpha = j / factor
            if use_rife:
                interp_frame = interpolator.interpolate(prev_frame, curr_frame)
            else:
                # 简单线性插值
                interp_frame = cv2.addWeighted(prev_frame, 1 - alpha, curr_frame, alpha, 0)
            out.write(interp_frame)
            total_inserted += 1
            total_output += 1

        # 更新前一帧
        prev_frame = curr_frame

        frame_idx += 1
        if frame_idx % 20 == 0:
            elapsed = time.time() - start_time
            fps = frame_idx / elapsed if elapsed > 0 else 0
            print(f"进度: {frame_idx}/{frame_count} 帧 | FPS: {fps:.1f}")

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    result_fps = total_output / duration

    print(f"\n处理完成!")
    print(f"  输出: {output_path}")
    print(f"  输出帧率: {result_fps:.1f} FPS")
    print(f"  插入帧数: {total_inserted}")
    print(f"  总帧数: {total_output}")
    print(f"  总耗时: {elapsed:.1f}s")


def download_rife_model(model_dir):
    """下载 RIFE 模型"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = "RIFE_HD.pth"
    model_path = model_dir / model_name

    if model_path.exists():
        print(f"模型已存在: {model_path}")
        return str(model_path)

    print(f"下载 RIFE 模型到: {model_path}")
    url = "https://github.com/hzwer/arXiv2020-RIFE/releases/download/v3.0/RIFE_HD.pth"

    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"模型下载完成")
        return str(model_path)
    except Exception as e:
        print(f"模型下载失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='RIFE 视频补帧')
    parser.add_argument('input', type=str, help='输入视频路径')
    parser.add_argument('-o', '--output', type=str, help='输出视频路径')
    parser.add_argument('-f', '--fps', type=float, default=60,
                       help='目标帧率 (默认: 60)')
    parser.add_argument('--rife', action='store_true',
                       help='使用 RIFE 模型')
    parser.add_argument('--model', type=str,
                       help='RIFE 模型路径')
    parser.add_argument('--download-model', action='store_true',
                       help='下载 RIFE 模型')

    args = parser.parse_args()

    # 自动生成输出路径
    if args.output is None:
        input_path = Path(args.input)
        suffix = '_interpolated' if not args.rife else '_rife_interpolated'
        args.output = str(input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}")

    # 下载模型
    model_path = args.model
    if args.rife and args.download_model:
        model_path = download_rife_model('/workspace/handle_video/models')

    # 执行补帧
    simple_interpolate_video(
        args.input,
        args.output,
        args.fps,
        use_rife=args.rife,
        model_path=model_path
    )


if __name__ == '__main__':
    main()
