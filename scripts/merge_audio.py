#!/usr/bin/env python3
"""
合并视频和音频，使用 ffmpeg
"""

import subprocess
import os
import argparse


def merge_audio_video(video_path, original_video_path, output_path):
    """
    合并处理后的视频和原视频的音频
    """
    print(f"视频: {video_path}")
    print(f"原视频(含音频): {original_video_path}")
    print(f"输出: {output_path}")

    # 使用 ffmpeg 合并
    # -map 1:v:0 使用视频文件的视频轨道
    # -map 0:a:0 使用原视频的音频轨道
    # -c:v copy 直接复制视频编码（不重新编码，保持原质量）
    # -c:a aac 音频重新编码为 AAC 格式
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-i', original_video_path,
        '-i', video_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '1:v:0',
        '-map', '0:a:0',
        '-shortest',
        output_path
    ]

    print(f"\n执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return False

    print(f"\n✅ 成功: {output_path}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='合并视频和音频')
    parser.add_argument('-v', '--video', required=True, help='处理后的视频文件')
    parser.add_argument('-o', '--original', required=True, help='原视频文件（包含音频）')
    parser.add_argument('-out', '--output', help='输出文件名（可选）')

    args = parser.parse_args()

    # 默认输出文件名
    if args.output is None:
        base, ext = os.path.splitext(args.video)
        args.output = f"{base}_with_audio{ext}"

    merge_audio_video(args.video, args.original, args.output)
