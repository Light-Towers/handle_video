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

### 环境检测

在安装前，可以先检测系统环境：

```bash
bash check_cuda.sh
```

这个脚本会：
- 检测系统 CUDA 版本
- 检测 GPU 型号
- 验证 PyTorch CUDA 版本可用性
- 测试当前环境兼容性
- 给出推荐安装命令

### 方法一：一键安装（推荐）

使用自动安装脚本处理所有依赖和兼容性问题：

```bash
# GPU 环境（自动检测 CUDA 版本）
bash install.sh

# CPU 环境
bash install.sh --cpu
```

安装脚本会自动：
- 检查 Python 版本（需要 3.11+）
- 安装系统依赖（ffmpeg）
- 安装 PyTorch（GPU 或 CPU 版本）
- 安装所有 Python 依赖
- 修复 torchvision API 兼容性问题
- 下载 Real-ESRGAN 模型
- 验证安装是否成功

### 方法二：手动安装

#### 1. 安装系统依赖

```bash
# Ubuntu/Debian
apt-get update && apt-get install -y ffmpeg

# CentOS/RHEL
yum install -y ffmpeg
```

#### 2. 安装 PyTorch

**PyTorch 2.0+（默认包含CUDA支持）：**
```bash
pip install torch==2.1.2 torchvision==0.16.2
```
```

> **版本说明：**
> - 推荐：torch 2.1.2 + torchvision 0.16.2（与 basicsr 1.4.2 兼容，无需修改源码）
> - 支持 CUDA 11.8 和 CUDA 12.x
> - 使用 `install.sh` 脚本会自动检测 CUDA 版本
> - 不推荐：torch 2.7.x + torchvision 0.22.x（需要修改源码）

#### 3. 安装依赖

```bash
# 安装核心包（无依赖模式，避免构建问题）
pip install --no-deps realesrgan==0.3.0 basicsr==1.4.2

# 安装其他依赖
pip install opencv-python==4.10.0.84 numpy==1.26.4 Pillow==11.3.0 scipy==1.17.0 \
    scikit-image==0.26.0 addict==2.4.0 lmdb==1.7.5 PyYAML==6.0.3 \
    requests==2.32.5 future==1.0.0 tqdm==4.67.3
```

#### 4. 安装其他依赖

```bash
pip install realesrgan==0.3.0 basicsr==1.4.2 opencv-python==4.10.0.84 numpy==1.26.4 \
    Pillow==11.3.0 scipy==1.17.0 scikit-image==0.26.0 addict==2.4.0 lmdb==1.7.5 \
    PyYAML==6.0.3 requests==2.32.5 future==1.0.0 tqdm==4.67.3
```

#### 5. 验证安装

```bash
python3 -c "import torch, torchvision, realesrgan, basicsr; print('所有依赖安装成功')"
```

#### 5. 下载 Real-ESRGAN 模型

```bash
mkdir -p models
wget -P models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

### 使用脚本处理视频

```bash
# Real-ESRGAN 基本用法（x4 放大）
python handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 4

# Real-ESRGAN x2 放大（更快）
python handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 2 -m RealESRGAN_x2plus

# Real-ESRGAN 动漫视频
python handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 4 -m RealESRGAN_x4plus_anime_6B

# 启用断点续传
python handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 4 --resume
```

### Real-ESRGAN 参数说明

| 参数 | 说明 | 选项 | 默认值 |
|------|------|------|--------|
| `-s, --scale` | 放大倍数 | 2, 4 | 4 |
| `-m, --model` | 模型选择 | RealESRGAN_x4plus, RealESRGAN_x2plus, RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus |
| `--codec` | 视频编码器 | mp4v, MJPG, vp09 | mp4v |
| `--tile-size` | 分块大小（0=不分块） | 0, 256, 512, 768, 1024 | 512 |
| `--resume` | 启用断点续传 | - | - |

**编码器说明**：
- `mp4v`: MPEG-4 Visual（默认，兼容性好）
- `MJPG`: MJPEG（无压缩，文件大，速度快）
- `vp09`: VP9（压缩率高，Web 优化）

**模型与 scale 兼容性**：
- `RealESRGAN_x4plus`: 支持 2x 或 4x 输出
- `RealESRGAN_x2plus`: 仅支持 2x 输出
- `RealESRGAN_x4plus_anime_6B`: 仅支持 4x 输出

### 双三次插值处理

```bash
python handle_video/scripts/video_upscale_bicubic.py -i input.mp4 -o output.mp4 -s 4
```

### 音视频合并

```bash
python handle_video/scripts/merge_audio.py -v processed.mp4 -o original.mp4 -out final.mp4
```

## 性能对比

| 算法 | 4x放大速度 | 质量 | 适用场景 |
|------|----------|------|---------|
| Real-ESRGAN | 1.6 FPS (T4 GPU) | ⭐⭐⭐⭐⭐ | 高质量输出 |
| 双三次插值 | 80 FPS (CPU) | ⭐⭐ | 快速预览 |

## 环境要求

### 硬件要求
- **GPU**: NVIDIA Tesla T4 或更高（Real-ESRGAN 需要）
- **内存**: 建议 8GB+ GPU 显存

### 软件要求
- **CUDA**: 11.8 或 12.x
- **Python**: 3.11+
- **ffmpeg**: 用于音视频处理
- **操作系统**: Linux / macOS / Windows (WSL)

### CUDA 版本支持

| CUDA 版本 | PyTorch 版本 | torchvision 版本 | 兼容性 |
|-----------|-------------|-----------------|--------|
| 11.8 | 2.1.2 | 0.16.2 | ✅ 支持 |
| 12.x | 2.1.2 | 0.16.2 | ✅ 支持 |

**说明：** 项目同时支持 CUDA 11.8 和 CUDA 12.x，使用 `install.sh` 会自动检测并选择合适的版本。

## 常见问题

详见 [VIDEO_PROCESSING_NOTES.md](docs/VIDEO_PROCESSING_NOTES.md)

### 已知兼容性问题

#### torchvision 版本兼容性

| torch | torchvision | basicsr 1.4.2 | 状态 | 说明 |
|-------|-------------|---------------|------|------|
| 2.1.2 | 0.16.2 | ✓ | ✅ 推荐 | 无需修改源码 |
| 2.7.1 | 0.22.1 | ✗ | ❌ 不推荐 | 需要修改源码 |

**原因：**
- torchvision 0.17+ 移除了 `functional_tensor` 模块
- basicsr 1.4.2 依赖此模块
- 使用推荐版本组合可避免兼容性问题

#### basicsr 构建隔离问题

**问题：** pip 在构建 basicsr 时使用隔离环境，无法访问已安装的 torch

**错误信息：**
```
ModuleNotFoundError: No module named 'torch'
```

**解决方案：**
```bash
pip install basicsr==1.4.2 --no-deps
```

## 许可证

MIT License
