#!/usr/bin/env python3
"""
Real-ESRGAN 视频超分辨率 - 多线程优化版本
使用多线程流水线提高 GPU 利用率
支持音频拷贝
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
from queue import Queue
from threading import Thread, Lock
import queue
import subprocess
import tempfile


class VideoProcessor:
    def __init__(self, scale=4, model_name='realesr-animevideov3', use_cuda=True, tile_size=0, num_workers=4):
        """
        初始化 Real-ESRGAN 视频处理器

        Args:
            scale: 放大倍数
            model_name: 模型名称
            tile_size: 分块大小 (0=不分块)
            num_workers: 工作线程数量
        """
        self.scale = scale
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.tile_size = tile_size
        self.num_workers = num_workers
        self.model_name = model_name

        script_dir = Path(__file__).parent.parent

        if model_name == 'realesr-animevideov3':
            model_path = str(script_dir / 'models' / 'realesr-animevideov3.pth')
            netscale = 4
            use_srvgg = True
            num_feat = 64
            num_conv = 16
        elif model_name == 'RealESRGAN_x4plus_anime_6B':
            model_path = str(script_dir / 'models' / 'RealESRGAN_x4plus_anime_6B.pth')
            netscale = 4
            use_srvgg = False
            num_feat = 64
            num_block = 6
            num_grow_ch = 32
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        if not Path(model_path).exists():
            fallback_path = str(script_dir / '.realesrgan' / f'{model_name}.pth')
            if Path(fallback_path).exists():
                model_path = fallback_path
            else:
                raise FileNotFoundError(f"找不到模型文件: {model_path}")

        if use_srvgg:
            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3,
                num_feat=num_feat, num_conv=num_conv,
                upscale=netscale, act_type='prelu'
            )
        else:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=num_feat, num_block=num_block,
                num_grow_ch=num_grow_ch, scale=netscale
            )

        actual_tile = self.tile_size if self.tile_size > 0 else 0
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=actual_tile,
            tile_pad=10,
            pre_pad=0,
            half=False,  # 使用 FP32 避免类型不匹配错误
            device=torch.device('cuda' if self.use_cuda else 'cpu')
        )

        print(f"✓ Real-ESRGAN 初始化完成")
        print(f"  模型: {model_name}")
        print(f"  放大倍数: {netscale}")
        print(f"  Tile 大小: {actual_tile}")
        print(f"  工作线程: {num_workers}")
        print(f"  设备: {'CUDA' if self.use_cuda else 'CPU'}")
        print(f"  精度: FP32")

    def process_frame(self, frame):
        """处理单帧"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output, _ = self.upsampler.enhance(frame_rgb, outscale=self.scale)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output_bgr


class VideoReaderThread(Thread):
    def __init__(self, cap, frame_queue, max_queue_size=32):
        super().__init__()
        self.cap = cap
        self.frame_queue = frame_queue
        self.max_queue_size = max_queue_size
        self.frame_count = 0
        self.daemon = True

    def run(self):
        while True:
            if self.frame_queue.qsize() < self.max_queue_size:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame_queue.put((self.frame_count, frame))
                self.frame_count += 1
            else:
                time.sleep(0.01)
        # 发送结束信号
        for _ in range(self.num_workers):
            self.frame_queue.put(None)


class VideoWriterThread(Thread):
    def __init__(self, writer, output_queue):
        super().__init__()
        self.writer = writer
        self.output_queue = output_queue
        self.daemon = True

    def run(self):
        while True:
            item = self.output_queue.get()
            if item is None:
                break
            idx, frame = item
            self.writer.write(frame)
            self.output_queue.task_done()


class VideoProcessorThread(Thread):
    def __init__(self, processor, input_queue, output_queue):
        super().__init__()
        self.processor = processor
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.daemon = True

    def run(self):
        while True:
            item = self.input_queue.get()
            if item is None:
                self.input_queue.task_done()
                break

            idx, frame = item
            try:
                processed = self.processor.process_frame(frame)
                self.output_queue.put((idx, processed))
            except Exception as e:
                print(f"处理帧 {idx} 时出错: {e}")
            finally:
                self.input_queue.task_done()


def copy_audio(input_path, temp_video_path, output_path):
    """
    使用 ffmpeg 将源视频的音频拷贝到处理后的视频中
    直接流拷贝，不重新编码

    Args:
        input_path: 原始输入视频路径
        temp_video_path: 处理后的临时视频路径（无音频）
        output_path: 最终输出路径（带音频）
    """
    print(f"\n正在拷贝音频...")

    # 检查原始视频是否有音频
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'a',
        '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', input_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)

    has_audio = result.returncode == 0 and 'audio' in result.stdout
    print(f"  源视频音频: {'存在' if has_audio else '无'}")

    # 如果没有音频，直接重命名临时文件
    if not has_audio:
        try:
            os.replace(temp_video_path, output_path)
            print(f"✓ 无音频，直接使用处理后的视频")
            return True
        except Exception as e:
            print(f"✗ 文件重命名失败: {e}")
            return False

    # 有音频：直接流拷贝，不重新编码
    # MP4 容器支持 AC3 音频，可以直接拷贝
    cmd = [
        'ffmpeg',
        '-i', temp_video_path,        # 处理后的视频（无音频）
        '-i', input_path,             # 原始视频（有音频）
        '-c:v', 'copy',              # 复制视频流，不重新编码
        '-c:a', 'copy',              # 复制音频流，不重新编码（MP4 支持 AC3）
        '-map', '0:v:0',            # 使用第一个文件的视频流
        '-map', '1:a:0',            # 使用第二个文件的第一个音频流
        '-y',                        # 覆盖输出文件
        output_path
    ]

    try:
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ 音频拷贝完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 音频拷贝失败: {e.stderr}")
        # 如果音频拷贝失败，直接重命名临时文件为最终输出
        try:
            os.replace(temp_video_path, output_path)
            print(f"✗ 音频拷贝失败，使用无音频版本")
            return False
        except Exception as e2:
            print(f"✗ 文件重命名也失败: {e2}")
            return False


def process_video_fast(input_path, output_path, scale=4, model_name='realesr-animevideov3',
                      use_cuda=True, tile_size=0, num_workers=4, copy_audio=True):
    """使用多线程流水线快速处理视频"""

    print(f"\n{'='*60}")
    print(f"多线程视频超分辨率处理")
    if copy_audio:
        print(f"(音频拷贝: 启用)")
    print(f"{'='*60}\n")

    # 初始化处理器
    processor = VideoProcessor(scale=scale, model_name=model_name, use_cuda=use_cuda,
                               tile_size=tile_size, num_workers=num_workers)

    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n输入视频信息:")
    print(f"  路径: {input_path}")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps} FPS")
    print(f"  总帧数: {total_frames}")

    out_width = width * scale
    out_height = height * scale

    # 如果需要拷贝音频，先输出到临时文件
    if copy_audio:
        temp_dir = tempfile.gettempdir()
        temp_filename = f"temp_video_{os.getpid()}_{time.time()}.mp4"
        temp_video_path = os.path.join(temp_dir, temp_filename)
        print(f"\n输出视频信息:")
        print(f"  临时路径: {temp_video_path} (无音频)")
        print(f"  最终路径: {output_path} (带音频)")
    else:
        temp_video_path = output_path
        print(f"\n输出视频信息:")
        print(f"  路径: {output_path}")
    print(f"  分辨率: {out_width}x{out_height}")
    print(f"  帧率: {fps} FPS")

    # 创建队列
    input_queue = Queue(maxsize=32)
    output_queue = Queue(maxsize=32)

    # 创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (out_width, out_height))

    # 启动线程
    reader_thread = VideoReaderThread(cap, input_queue, max_queue_size=32)
    writer_thread = VideoWriterThread(writer, output_queue)

    worker_threads = []
    for i in range(num_workers):
        worker = VideoProcessorThread(processor, input_queue, output_queue)
        worker_threads.append(worker)

    reader_thread.num_workers = num_workers

    # 启动所有线程
    print(f"\n启动处理线程...")
    reader_thread.start()
    writer_thread.start()
    for worker in worker_threads:
        worker.start()

    # 监控进度
    processed_count = 0
    start_time = time.time()
    last_update_time = start_time
    last_count = 0

    print(f"\n开始处理...\n")

    while writer_thread.is_alive():
        time.sleep(0.5)

        current_time = time.time()
        current_count = reader_thread.frame_count

        if current_count > last_count:
            elapsed = current_time - last_update_time
            if elapsed >= 2.0:  # 每2秒更新一次
                fps_current = (current_count - last_count) / elapsed
                total_elapsed = current_time - start_time
                avg_fps = current_count / total_elapsed if total_elapsed > 0 else 0
                progress = current_count / total_frames * 100
                eta = (total_frames - current_count) / avg_fps if avg_fps > 0 else 0

                print(f"\r进度: {progress:.1f}% ({current_count}/{total_frames}) | "
                      f"当前FPS: {fps_current:.1f} | 平均FPS: {avg_fps:.1f} | "
                      f"输入队列: {input_queue.qsize()} | 输出队列: {output_queue.qsize()} | "
                      f"预计剩余: {eta/60:.1f}min", end='', flush=True)

                last_update_time = current_time
                last_count = current_count

        # 额外检查：如果 reader 完成且输出队列为空，强制退出
        if not reader_thread.is_alive() and output_queue.empty():
            print(f"\n检测到所有帧已处理完成")
            # 发送结束信号到输出队列，让 writer 线程可以退出
            for _ in range(num_workers + 1):
                output_queue.put(None)
            break

    # 等待所有线程完成
    for worker in worker_threads:
        worker.join()

    # 所有 worker 完成后，发送结束信号到输出队列
    print(f"\n正在发送结束信号...")
    for _ in range(num_workers + 1):  # 确保有足够多的结束信号
        output_queue.put(None)

    writer_thread.join()
    reader_thread.join()

    cap.release()
    writer.release()

    total_elapsed = time.time() - start_time
    final_avg_fps = total_frames / total_elapsed if total_elapsed > 0 else 0

    print(f"\n\n{'='*60}")
    print(f"✓ 视频处理完成!")
    print(f"{'='*60}")
    print(f"  临时输出: {temp_video_path}")
    print(f"  输出分辨率: {out_width}x{out_height}")
    print(f"  平均 FPS: {final_avg_fps:.2f}")
    print(f"  总耗时: {total_elapsed:.2f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*60}")

    # 拷贝音频
    if copy_audio:
        audio_success = copy_audio(input_path, temp_video_path, output_path)
        if audio_success:
            # 删除临时文件
            try:
                os.remove(temp_video_path)
                print(f"✓ 临时文件已删除: {temp_video_path}")
            except Exception as e:
                print(f"⚠ 临时文件删除失败: {e}")

    print(f"\n{'='*60}")
    print(f"✓ 全部完成!")
    print(f"{'='*60}")
    print(f"  最终输出: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Real-ESRGAN 视频超分辨率 - 多线程优化版 (支持音频拷贝)')
    parser.add_argument('input', type=str, help='输入视频路径')
    parser.add_argument('-o', '--output', type=str, help='输出视频路径')
    parser.add_argument('-s', '--scale', type=int, default=4, choices=[2, 4], help='放大倍数')
    parser.add_argument('-n', '--model', type=str, default='realesr-animevideov3',
                       choices=['realesr-animevideov3', 'RealESRGAN_x4plus_anime_6B'], help='模型名称')
    parser.add_argument('--no-cuda', action='store_true', help='禁用 CUDA')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile 大小 (0=不分块)')
    parser.add_argument('-w', '--workers', type=int, default=4, help='工作线程数量')
    parser.add_argument('--no-audio', action='store_true', help='不拷贝源音频')

    args = parser.parse_args()

    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_{args.model}_enhanced{input_path.suffix}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    process_video_fast(
        input_path=args.input,
        output_path=args.output,
        scale=args.scale,
        model_name=args.model,
        use_cuda=not args.no_cuda,
        tile_size=args.tile,
        num_workers=args.workers,
        copy_audio=not args.no_audio
    )


if __name__ == '__main__':
    main()
