#!/usr/bin/env python3
"""
RIFE 视频补帧脚本（简化版）
"""

import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import sys

# 添加 RIFE 模型路径
rife_path = Path(__file__).parent.parent / 'RIFE'
sys.path.insert(0, str(rife_path))


class RIFEInterpolator:
    """RIFE 补帧器"""

    def __init__(self, model_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)

        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # 加载模型 - 把 model_dir 添加到 Python 路径
        sys.path.insert(0, str(model_dir))

        try:
            from RIFE_HDv3 import Model
            self.model = Model()
            self.model.load_model(model_dir, -1)
            print("Loaded v3.x HD model.")
        except:
            try:
                from RIFE_HDv2 import Model
                self.model = Model()
                self.model.load_model(model_dir, -1)
                print("Loaded v2.x HD model.")
            except:
                from RIFE_HD import Model
                self.model = Model()
                self.model.load_model(model_dir, -1)
                print("Loaded v1.x HD model")

        self.model.eval()
        self.model.device()

    def interpolate(self, img0, img1, scale=1.0):
        """在两帧之间插值"""
        img0 = (torch.from_numpy(np.transpose(img0, (2, 0, 1))).float() / 255.0).unsqueeze(0)
        img1 = (torch.from_numpy(np.transpose(img1, (2, 0, 1))).float() / 255.0).unsqueeze(0)

        img0 = img0.to(self.device)
        img1 = img1.to(self.device)

        with torch.no_grad():
            if scale != 1.0:
                img0 = torch.nn.functional.interpolate(img0, scale_factor=scale, mode='bilinear', align_corners=False)
                img1 = torch.nn.functional.interpolate(img1, scale_factor=scale, mode='bilinear', align_corners=False)

            output = self.model.inference(img0, img1)
            output = output[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0

        return output.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='RIFE Video Interpolation')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--model-dir', type=str, default='/workspace/handle_video/models/rife', help='Model directory')
    parser.add_argument('--exp', type=int, default=1, help='2^exp interpolation')
    parser.add_argument('--scale', type=float, default=1.0, help='Process scale')
    parser.add_argument('--fps', type=int, default=None, help='Target FPS')
    args = parser.parse_args()

    # 默认输出路径
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_rife{video_path.suffix}")

    # 初始化插值器
    interpolator = RIFEInterpolator(args.model_dir)

    # 打开视频
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {width}x{height}, {fps} FPS, {total_frames} frames")

    # 目标 FPS
    if args.fps is None:
        target_fps = fps * (2 ** args.exp)
    else:
        target_fps = args.fps

    print(f"Output: {target_fps} FPS (2^{args.exp} interpolation)")

    # 输出保持原分辨率，scale 仅用于降低推理时的显存占用
    print(f"Output resolution: {width}x{height}")
    if args.scale != 1.0:
        print(f"Note: Processing at {args.scale}x scale to reduce memory")

    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, target_fps, (width, height))

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read video")

    prev_frame = prev_frame[:, :, ::-1].copy()

    total_output = 0
    inserted = 0

    # 逐帧处理
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            # 写入最后一帧
            out.write(prev_frame[:, :, ::-1])
            total_output += 1
            break

        curr_frame = curr_frame[:, :, ::-1].copy()

        # 写入前一帧
        out.write(prev_frame[:, :, ::-1])
        total_output += 1

        # 插值生成中间帧
        for _ in range(2 ** args.exp - 1):
            interp_frame = interpolator.interpolate(prev_frame, curr_frame, args.scale)
            # 如果使用 scale 缩放，需要恢复到原始分辨率
            if args.scale != 1.0:
                interp_frame = cv2.resize(interp_frame, (width, height), interpolation=cv2.INTER_LINEAR)
            out.write(interp_frame[:, :, ::-1])
            total_output += 1
            inserted += 1

        prev_frame = curr_frame

        if total_output % 50 == 0:
            print(f"Progress: {total_output} frames output")

    cap.release()
    out.release()

    print(f"Done! Output: {args.output}")
    print(f"Total frames: {total_output}, Inserted: {inserted}")


if __name__ == '__main__':
    main()
