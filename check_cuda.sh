#!/bin/bash
# ========================================
# CUDA 版本检测和兼容性测试脚本
# ========================================

echo "========================================="
echo "CUDA 环境检测"
echo "========================================="
echo ""

# 检测 nvcc
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "✓ nvcc 版本: $CUDA_VERSION"
else
    echo "✗ nvcc 未安装"
fi

# 检测 nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "✓ nvidia-smi 信息:"
    nvidia-smi --query-gpu=driver_version,cuda_version,name --format=csv,noheader 2>/dev/null | while read line; do
        echo "  - $line"
    done
else
    echo "✗ nvidia-smi 未安装"
fi

echo ""
echo "========================================="
echo "PyTorch CUDA 版本测试"
echo "========================================="
echo ""

# 测试不同 CUDA 版本的 PyTorch 是否可用
echo "测试 CUDA 11.8 版本:"
CUDA118_URL="https://download.pytorch.org/whl/cu118/torch-2.1.2+cu118-cp311-cp311-linux_x86_64.whl"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$CUDA118_URL")
if [ "$HTTP_CODE" = "200" ]; then
    echo "  ✓ PyTorch 2.1.2 (CUDA 11.8) 可用"
else
    echo "  ✗ PyTorch 2.1.2 (CUDA 11.8) 不可用 (HTTP $HTTP_CODE)"
fi

echo ""
echo "测试 CUDA 12.1 版本:"
CUDA121_URL="https://download.pytorch.org/whl/cu121/torch-2.1.2+cu121-cp311-cp311-linux_x86_64.whl"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$CUDA121_URL")
if [ "$HTTP_CODE" = "200" ]; then
    echo "  ✓ PyTorch 2.1.2 (CUDA 12.1) 可用"
else
    echo "  ✗ PyTorch 2.1.2 (CUDA 12.1) 不可用 (HTTP $HTTP_CODE)"
fi

echo ""
echo "========================================="
echo "当前环境 PyTorch 信息"
echo "========================================="
echo ""

python3 << 'EOF'
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA 不可用，运行在 CPU 模式")

# 检测兼容性
print()
print("兼容性测试:")
try:
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
    print("✓ torchvision.functional_tensor 模块可用")
except ImportError:
    print("✗ torchvision.functional_tensor 模块不可用")

try:
    import basicsr
    print(f"✓ basicsr {basicsr.__version__} 导入成功")
except ImportError:
    print("✗ basicsr 导入失败")

try:
    import realesrgan
    print("✓ realesrgan 导入成功")
except ImportError:
    print("✗ realesrgan 导入失败")
EOF

echo ""
echo "========================================="
echo "推荐安装命令"
echo "========================================="
echo ""

# 根据检测到的 CUDA 版本给出推荐
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    if [[ "$CUDA_VERSION" == "11.8" ]] || [[ "$CUDA_VERSION" == "11."* ]]; then
        echo "检测到 CUDA $CUDA_VERSION，推荐使用:"
        echo "  pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118"
    else
        echo "检测到 CUDA $CUDA_VERSION，推荐使用:"
        echo "  pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121"
    fi
else
    echo "无法自动检测 CUDA 版本，请手动选择:"
    echo "  CUDA 11.8:"
    echo "    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118"
    echo "  CUDA 12.x:"
    echo "    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121"
    echo "  CPU:"
    echo "    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu"
fi

echo ""
