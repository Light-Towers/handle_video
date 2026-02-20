#!/usr/bin/env python3
"""
快速视频超分辨率 - 使用 OpenCV 双三次插值
简单快速，适合快速预览
"""

import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm


def process_video(input_path, output_path, scale=4):
    """使用双三次插值进行视频上采样"""

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"输入: {width}x{height} @ {fps}fps")
    print(f"输出: {width*scale}x{height*scale} @ {fps}fps")
    print(f"总帧数: {total_frames}")

    # 使用 mp4v 编码（后续用 ffmpeg 转为 H.264）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width * scale, height * scale)
    )

    print("\n处理中...")
    with tqdm(total=total_frames, desc="进度") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 双三次插值上采样
            upscaled = cv2.resize(
                frame,
                (width * scale, height * scale),
                interpolation=cv2.INTER_CUBIC
            )

            out.write(upscaled)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"\n✅ 完成: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='快速视频超分辨率（双三次插值）')
    parser.add_argument('-i', '--input', required=True, help='输入视频')
    parser.add_argument('-o', '--output', required=True, help='输出视频')
    parser.add_argument('-s', '--scale', type=int, default=4, help='放大倍数')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 文件不存在: {args.input}")
        exit(1)

    process_video(args.input, args.output, args.scale)
