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

### 方法一：一键安装（推荐）

使用自动安装脚本处理所有依赖和兼容性问题：

```bash
# GPU 环境（CUDA 12.x）
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

**GPU 版本（CUDA 12.x）：**
```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
```

**CPU 版本：**
```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

#### 3. 安装依赖

```bash
# 安装核心包（无依赖模式，避免构建问题）
pip install --no-deps realesrgan==0.3.0 basicsr==1.4.2

# 安装其他依赖
pip install opencv-python==4.10.0.84 numpy==1.26.4 Pillow==11.3.0 scipy==1.17.0 \
    scikit-image==0.26.0 addict==2.4.0 lmdb==1.7.5 PyYAML==6.0.3 \
    requests==2.32.5 future==1.0.0 tqdm==4.67.3
```

#### 4. 修复兼容性问题

修改 basicsr 文件以适配新版 torchvision：

```bash
# 找到 basicsr 安装路径
python3 -c "import basicsr; import os; print(os.path.dirname(basicsr.__file__))"

# 编辑文件: {basicsr_path}/data/degradations.py
# 将第 8 行修改为:
# from torchvision.transforms._functional_tensor import rgb_to_grayscale
```

或者使用 sed 自动修复：
```bash
BASICSR_PATH=$(python3 -c "import basicsr; import os; print(os.path.dirname(basicsr.__file__))")
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms._functional_tensor import rgb_to_grayscale/g' \
    "$BASICSR_PATH/data/degradations.py"
```

#### 5. 下载 Real-ESRGAN 模型

```bash
mkdir -p ~/.realesrgan
wget -P ~/.realesrgan https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

### 使用脚本处理视频

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

### 已知兼容性问题

#### torchvision API 变更 (v0.22.x)

**问题：** basicsr 1.4.2 使用的 `torchvision.transforms.functional_tensor` 在 torchvision 0.22+ 中被重命名为 `_functional_tensor`

**错误信息：**
```
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```

**解决方案：**
1. 使用一键安装脚本（自动修复）
2. 或手动修改 degradations.py（详见安装说明）
3. 或降级 torchvision 到 0.15.x 版本

#### basicsr 构建隔离问题

**问题：** pip 在构建 basicsr 时使用隔离环境，无法访问已安装的 torch

**错误信息：**
```
ModuleNotFoundError: No module named 'torch'
```

**解决方案：**
```bash
pip install --no-deps basicsr==1.4.2
```

#### pip 构建依赖冲突

**问题：** basicsr 某些版本依赖 `tb-nightly`，但该包不存在

**解决方案：**
```bash
pip install basicsr==1.4.2 --no-deps
```

## 许可证

MIT License
