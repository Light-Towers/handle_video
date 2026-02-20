#!/bin/bash
# ========================================
# Real-ESRGAN 环境安装脚本
# ========================================
# 用途: 一键安装所有依赖和配置环境
# 使用: bash install.sh [--cpu]
# ========================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检测参数
USE_CPU=false
if [ "$1" = "--cpu" ]; then
    USE_CPU=true
    log_warn "使用 CPU 模式安装 PyTorch"
fi

# ========================================
# 1. 检查 Python 版本
# ========================================
log_info "检查 Python 版本..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MAJOR" -eq 3 -a "$PYTHON_MINOR" -lt 10 ]; then
    log_error "需要 Python 3.10 或更高版本，当前版本: $PYTHON_VERSION"
    exit 1
fi
log_info "Python 版本检查通过: $PYTHON_VERSION"

# ========================================
# 2. 安装系统依赖
# ========================================
log_info "检查系统依赖..."
if ! command -v ffmpeg &> /dev/null; then
    log_warn "ffmpeg 未安装，尝试安装..."
    if command -v apt-get &> /dev/null; then
        apt-get update -qq && apt-get install -y ffmpeg
    elif command -v yum &> /dev/null; then
        yum install -y ffmpeg
    else
        log_error "无法自动安装 ffmpeg，请手动安装"
        exit 1
    fi
    log_info "ffmpeg 安装成功"
else
    log_info "ffmpeg 已安装"
fi

# ========================================
# 3. 安装 PyTorch
# ========================================
log_info "安装 PyTorch..."

if [ "$USE_CPU" = true ]; then
    # CPU 版本
    python3 -m pip install torch==2.1.2 torchvision==0.16.2
    log_info "PyTorch (CPU) 安装成功"
else
    # 自动检测 CUDA 版本
    CUDA_VERSION="12.1"  # 默认使用 CUDA 12.1

    # 检测系统 CUDA 版本
    if command -v nvcc &> /dev/null; then
        SYSTEM_CUDA=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        log_info "检测到系统 CUDA: $SYSTEM_CUDA"

        # 判断使用哪个 CUDA 版本的 PyTorch
        case "$SYSTEM_CUDA" in
            11.8|11.*)
                CUDA_VERSION="11.8"
                log_info "使用 CUDA 11.8 版本的 PyTorch"
                ;;
            *)
                CUDA_VERSION="12.1"
                log_info "使用 CUDA 12.1 版本的 PyTorch"
                ;;
        esac
    else
        # 通过 nvidia-smi 检测
        if command -v nvidia-smi &> /dev/null; then
            NVIDIA_SMI=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
            log_info "通过 nvidia-smi 检测到 CUDA: $NVIDIA_SMI"

            # CUDA 11.x 使用 cu118，否则使用 cu121
            case "$NVIDIA_SMI" in
                11.8|11.*)
                    CUDA_VERSION="11.8"
                    log_info "使用 CUDA 11.8 版本的 PyTorch"
                    ;;
                *)
                    CUDA_VERSION="12.1"
                    log_info "使用 CUDA 12.1 版本的 PyTorch"
                    ;;
            esac
        else
            log_warn "无法检测 CUDA 版本，使用默认 CUDA 12.1"
        fi
    fi

    # 安装 PyTorch（2.0+版本默认包含CUDA支持）
    python3 -m pip install torch==2.1.2 torchvision==0.16.2
    log_info "PyTorch 安装成功"
fi

# ========================================
# 4. 安装其他依赖
# ========================================
log_info "安装 Python 依赖包..."

# 安装不带依赖的核心包（避免构建隔离问题）
log_info "安装核心包 (无依赖模式)..."
python3 -m pip install --no-deps realesrgan==0.3.0 basicsr==1.4.2

# 安装其他依赖
log_info "安装其他依赖..."
python3 -m pip install \
    opencv-python==4.10.0.84 \
    numpy==1.26.4 \
    Pillow==11.3.0 \
    scipy==1.14.1 \
    scikit-image==0.24.0 \
    addict==2.4.0 \
    lmdb==1.7.5 \
    PyYAML==6.0.3 \
    requests==2.32.5 \
    future==1.0.0 \
    tqdm==4.67.3 \
    tb-nightly \
    yapf \
    facexlib>=0.2.5 \
    gfpgan>=1.3.5

log_info "依赖安装完成"

# ========================================
# 5. 验证兼容性
# ========================================
log_info "验证版本兼容性..."
python3 << EOF
import warnings
warnings.filterwarnings('ignore')

import torchvision
print(f"torchvision: {torchvision.__version__}")

# 检查 functional_tensor 模块是否存在
try:
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
    print("✓ torchvision.functional_tensor 模块存在")
except ImportError:
    print("✗ torchvision.functional_tensor 模块不存在")
    exit(1)

# 检查 basicsr 导入
try:
    import basicsr
    print(f"✓ basicsr: {basicsr.__version__}")
except ImportError as e:
    print(f"✗ basicsr 导入失败: {e}")
    exit(1)

# 检查 realesrgan 导入
try:
    import realesrgan
    print(f"✓ realesrgan: {realesrgan.__version__}")
except ImportError as e:
    print(f"✗ realesrgan 导入失败: {e}")
    exit(1)

print("\n版本兼容性验证通过！")
EOF

if [ $? -ne 0 ]; then
    log_error "版本兼容性验证失败"
    exit 1
fi

# ========================================
# 6. 下载 Real-ESRGAN 模型
# ========================================
log_info "检查 Real-ESRGAN 模型..."
REAL_ESRGAN_DIR="$HOME/.realesrgan"
REAL_ESRGAN_MODEL="$REAL_ESRGAN_DIR/RealESRGAN_x4plus.pth"

if [ ! -f "$REAL_ESRGAN_MODEL" ]; then
    log_info "下载 Real-ESRGAN x4plus 模型..."
    mkdir -p "$REAL_ESRGAN_DIR"
    cd "$REAL_ESRGAN_DIR"
    wget -q --show-progress https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
    log_info "模型下载完成"
else
    log_info "模型已存在，跳过下载"
fi

# ========================================
# 7. 验证安装
# ========================================
log_info "验证安装..."
python3 << EOF
import sys
try:
    import torch
    import torchvision
    import cv2
    import numpy as np
    import realesrgan
    import basicsr
    print(f"✓ torch: {torch.__version__}")
    print(f"✓ torchvision: {torchvision.__version__}")
    print(f"✓ opencv: {cv2.__version__}")
    print(f"✓ numpy: {np.__version__}")
    print(f"✓ realesrgan: {realesrgan.__version__}")
    print(f"✓ basicsr: {basicsr.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    log_info "所有依赖验证通过！"
else
    log_error "依赖验证失败"
    exit 1
fi

# ========================================
# 完成
# ========================================
echo ""
log_info "============================================"
log_info "安装完成！"
log_info "============================================"
echo ""
echo "使用方法:"
echo "  Real-ESRGAN 处理:"
echo "    python scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 4"
echo ""
echo "  双三次插值处理:"
echo "    python scripts/video_upscale_bicubic.py -i input.mp4 -o output.mp4 -s 4"
echo ""
echo "  音视频合并:"
echo "    python scripts/merge_audio.py -v processed.mp4 -o original.mp4 -out final.mp4"
echo ""
