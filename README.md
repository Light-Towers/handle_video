# 视频超分辨率处理项目

基于 GPU 加速的视频超分辨率处理工具集，支持 Real-ESRGAN 和双三次插值两种算法。

## 功能特性

- 🚀 Real-ESRGAN 深度学习超分辨率（高质量）
- ⚡ 双三次插值快速放大（高速）
- 🔄 断点续传支持
- 🔊 音视频合并
- 📊 H.264 编码兼容性优化

## 项目结构

```
handle_video/
├── scripts/                    # 处理脚本
│   ├── video_upscale_realesrgan.py   # Real-ESRGAN 处理
│   ├── video_upscale_bicubic.py      # 双三次插值处理
│   ├── merge_audio.py                 # 音视频合并
│   └── check_video_info.py           # 视频信息检查
├── docs/                       # 文档
│   ├── README.md                      # 项目说明
│   ├── VIDEO_PROCESSING_NOTES.md      # 详细技术笔记
│   └── GPU_HARDWARE_GUIDE.md          # GPU 硬件指南
├── requirements.txt            # Python 依赖
└── .gitignore                 # Git 忽略配置
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### Real-ESRGAN 处理

```bash
# 基本用法
python scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 4

# 启用断点续传
python scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 4 --resume
```

### 双三次插值处理

```bash
python scripts/video_upscale_bicubic.py -i input.mp4 -o output.mp4 -s 4
```

### 音视频合并

```bash
python scripts/merge_audio.py -v processed.mp4 -o original.mp4 -out final.mp4
```

## 性能对比

| 算法 | 4x放大速度 | 质量 | 适用场景 |
|------|----------|------|---------|
| Real-ESRGAN | 1.6 FPS (T4 GPU) | ⭐⭐⭐⭐⭐ | 高质量输出 |
| 双三次插值 | 80 FPS (CPU) | ⭐⭐ | 快速预览 |

## 环境要求

- **GPU**: NVIDIA Tesla T4 或更高（Real-ESRGAN 需要）
- **CUDA**: 12.1+
- **Python**: 3.11+
- **ffmpeg**: 用于音视频处理

## 常见问题

详见 [VIDEO_PROCESSING_NOTES.md](docs/VIDEO_PROCESSING_NOTES.md)

## 许可证

MIT License
