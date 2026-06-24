import cv2
import sys

video_path = sys.argv[1] if len(sys.argv) > 1 else '/workspace/01 Gets Lost in Space.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"无法打开视频: {video_path}")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps if fps > 0 else 0

cap.release()

print(f"视频: {video_path}")
print(f"分辨率: {width}x{height}")
print(f"帧率: {fps:.2f} FPS")
print(f"总帧数: {frame_count}")
print(f"时长: {duration:.1f} 秒")

# Real-ESRGAN 预估时间
fps_realesrgan = 1.6  # 基于之前的测试
estimated_time = frame_count / fps_realesrgan
minutes = int(estimated_time // 60)
seconds = int(estimated_time % 60)

print(f"\nReal-ESRGAN 预估处理时间: {minutes}分{seconds}秒")
