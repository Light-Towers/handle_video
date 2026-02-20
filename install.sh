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
if [[ "$1" == "--cpu" ]]; then
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

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 11 ]]; then
    log_error "需要 Python 3.11 或更高版本，当前版本: $PYTHON_VERSION"
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
    python3 -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
else
    # 尝试检测 CUDA 版本
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        log_info "检测到 CUDA: $CUDA_VERSION"
    fi
    python3 -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
fi
log_info "PyTorch 安装成功"

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
    scipy==1.17.0 \
    scikit-image==0.26.0 \
    addict==2.4.0 \
    lmdb==1.7.5 \
    PyYAML==6.0.3 \
    requests==2.32.5 \
    future==1.0.0 \
    tqdm==4.67.3

log_info "依赖安装完成"

# ========================================
# 5. 修复兼容性问题
# ========================================
log_info "修复 torchvision API 兼容性问题..."

# 查找 basicsr 安装路径
BASICSR_PATH=$(python3 -c "import basicsr; print(basicsr.__file__)" 2>/dev/null | sed 's/__init__.py//')
DEGRADATIONS_FILE="$BASICSR_PATH/data/degradations.py"

if [[ -f "$DEGRADATIONS_FILE" ]]; then
    log_info "修改文件: $DEGRADATIONS_FILE"
    if grep -q "from torchvision.transforms.functional_tensor import rgb_to_grayscale" "$DEGRADATIONS_FILE"; then
        sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms._functional_tensor import rgb_to_grayscale/g' "$DEGRADATIONS_FILE"
        log_info "兼容性问题修复成功"
    else
        log_info "兼容性问题已修复或无需修复"
    fi
else
    log_warn "未找到 degradations.py 文件，跳过修复"
fi

# ========================================
# 6. 下载 Real-ESRGAN 模型
# ========================================
log_info "检查 Real-ESRGAN 模型..."
REAL_ESRGAN_DIR="$HOME/.realesrgan"
REAL_ESRGAN_MODEL="$REAL_ESRGAN_DIR/RealESRGAN_x4plus.pth"

if [[ ! -f "$REAL_ESRGAN_MODEL" ]]; then
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
