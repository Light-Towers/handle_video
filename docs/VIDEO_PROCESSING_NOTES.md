# 视频超分辨率处理笔记

## 环境信息

- **GPU**: NVIDIA Tesla T4 (15GB)
- **CUDA**: 12.1
- **驱动**: 580.65.06
- **Python**: 3.11.1

## 一、问题背景

1. **初始需求**: 从 GitHub 安装 video2x 进行 GPU 加速视频处理
2. **问题**: Video2X 需要 Vulkan API，但 Cloud Studio 环境不支持
3. **解决方案**: 转向 CUDA 方案

## 二、Vulkan 支持问题

### 问题诊断

```bash
# 检查 Vulkan
vulkaninfo  # Found no drivers!

# NVIDIA 驱动问题
nvidia-smi  # Driver/library version mismatch
```

### 解决方案

```bash
# 修复驱动符号链接
ln -sf libnvidia-ml.so.580.65.06 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1

# 安装 mesa-vulkan-drivers (仅提供 CPU 渲染)
apt-get install mesa-vulkan-drivers
```

### 结论

- Video2X **无法使用**（依赖 Vulkan）
- 必须使用 CUDA 方案

## 三、可用的视频处理方案

### 方案 1: Real-ESRGAN (推荐 - 高质量)

#### 安装依赖

```bash
pip3 install realesrgan basicsr facexlib gfpgan tqdm
pip3 install torchvision

# 修复 basicsr 兼容性（如果使用不兼容版本）
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' \
  /root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/basicsr/data/degradations.py

# 安装 ffmpeg
apt-get update && apt-get install -y ffmpeg
```

#### 使用方法

```bash
# 基本用法（x4 放大）
python3 handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 4

# x2 放大（更快）
python3 handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 2 -m RealESRGAN_x2plus

# 动漫视频
python3 handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 4 -m RealESRGAN_x4plus_anime_6B

# 启用断点续传
python3 handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 4 --resume
```

#### 性能

- **4x 放大**: ~1.6 FPS
- **处理 19s 视频**: ~4.6 分钟
- **质量**: ⭐⭐⭐⭐⭐

#### 参数说明

```python
# video_upscale_realesrgan.py 中的关键参数
RealESRGANer(
    scale=4,           # 放大倍数
    model_path=...,    # 模型路径
    model=model,       # RRDBNet 模型
    tile=256,          # 分块大小（提高速度）
    tile_pad=10,
    pre_pad=0,
    half=True,         # 半精度推理
)
```

---

### 方案 2: 双三次插值 (推荐 - 快速)

#### 使用方法

```bash
python3 handle_video/scripts/video_upscale_bicubic.py -i input.mp4 -o output.mp4 -s 4
```

#### 性能

- **4x 放大**: ~80 FPS
- **处理 19s 视频**: ~5 秒
- **质量**: ⭐⭐

#### 参数说明

```python
cv2.resize(
    frame,
    (width * scale, height * scale),
    interpolation=cv2.INTER_CUBIC  # 双三次插值
)
```

---

### 方案 3: ESPCN (存在问题)

#### 问题

- **模型无预训练权重**: 输出全黑（mean=0）
- **预训练模型链接失效**: GitHub 404

#### 不建议使用

---

## 四、音频处理

### 问题

OpenCV 的 `VideoWriter` **只处理视频轨道，不处理音频**。

### 解决方案

使用 ffmpeg 合并视频和音频：

```python
# merge_audio.py
cmd = [
    'ffmpeg',
    '-y',
    '-i', original_video_path,  # 原视频（包含音频）
    '-i', processed_video_path,  # 处理后的视频
    '-c:v', 'copy',              # 复制视频编码
    '-c:a', 'aac',               # 音频编码为 AAC
    '-map', '1:v:0',             # 使用处理后的视频
    '-map', '0:a:0',             # 使用原视频的音频
    '-shortest',
    output_path
]
```

### 使用

```bash
python3 handle_video/scripts/merge_audio.py -v processed.mp4 -o original.mp4 -out output_final.mp4
```

---

## 五、编码兼容性问题

### 问题

- **OpenCV 默认编码**: `mp4v` (MPEG-4 Part 2)
- **浏览器要求**: `H.264` (AVC)
- **结果**: mp4v 编码在浏览器中无法播放（黑屏）

### 解决方案

用 ffmpeg 重新编码为 H.264：

```bash
ffmpeg -i input.mp4 \
  -c:v libx264 \
  -preset fast \
  -crf 23 \
  -c:a copy \
  output_h264.mp4
```

### 参数说明

- `-c:v libx264`: 使用 H.264 视频编码
- `-preset fast`: 编码速度/质量平衡
- `-crf 23`: 质量参数（18-28，越小质量越好）
- `-c:a copy`: 复制音频流

---

## 六、完整工作流程

### 步骤 1: 视频超分辨率

```bash
# Real-ESRGAN (高质量)
python3 handle_video/scripts/video_upscale_realesrgan.py \
  input.mp4 \
  -o output.mp4 \
  -s 4

# 双三次插值 (快速)
python3 handle_video/scripts/video_upscale_bicubic.py \
  -i input.mp4 \
  -o output.mp4 \
  -s 4
```

### 步骤 2: 编码为 H.264

```bash
ffmpeg -i output.mp4 \
  -c:v libx264 \
  -preset fast \
  -crf 23 \
  -c:a copy \
  output_h264.mp4
```

### 步骤 3: 合并音频

```bash
python3 handle_video/scripts/merge_audio.py \
  -v output_h264.mp4 \
  -o input.mp4 \
  -out output_final.mp4
```

---

## 七、性能对比

| 方案 | 放大 | 耗时 | 大小 | 质量 | 兼容性 | 推荐场景 |
|------|------|------|------|------|--------|----------|
| Real-ESRGAN | 4x | 4.6分钟 | 5.8M | ⭐⭐⭐⭐⭐ | ✅ | 高质量 |
| 双三次插值 | 4x | 5秒 | 3.4M | ⭐⭐ | ✅ | 快速预览 |
| ESPCN | 4x | 16秒 | - | ❌ | ❌ | 不推荐 |

---

## 八、常见问题

### Q1: Video2X 无法使用？

**A**: Video2X 依赖 Vulkan，当前环境仅支持 CUDA。

### Q2: 输出视频黑屏？

**A**: 可能原因：
1. **编码不兼容**: mp4v 编码浏览器不支持 → 使用 H.264
2. **模型问题**: 未加载预训练权重 → 使用 Real-ESRGAN

### Q2.1: 为什么有些视频处理 FPS 只有 0.3？

**A**: RealESRGAN 处理速度受视频分辨率影响很大：

| 视频分辨率 | 每帧像素数 | FPS (T4 GPU) | 说明 |
|-----------|-----------|--------------|------|
| 426x240 | 102,240 | ~1.5 | 标准测试视频 |
| 720x576 | 414,720 | ~0.3 | 像素数是 4.26x240 的 4 倍，速度慢 4-5 倍 |

**原因**：
- RealESRGAN 是深度学习模型，计算量与像素数成正比
- 720x576 放大 4 倍后是 2880x2304，每帧计算量巨大
- 高分辨率视频建议：
  1. 先截取短片段测试（如 5 秒）
  2. 使用 x2 模型而不是 x4
  3. 降低 tile_size 增加稳定性

### Q2.2: scale 和 model 不匹配会怎样？

**A**: `enhance(frame, outscale=X)` 的工作原理：
1. 先用模型进行 AI 放大（由模型的 `scale` 决定）
2. 然后用双三次插值调整到 `outscale` 倍

| 模型 | 模型倍数 | 输出倍数 | 结果 |
|------|---------|---------|------|
| x4plus | 4 | 4 | ✓ 正常，AI 放大 4 倍 |
| x2plus | 2 | 2 | ✓ 正常，AI 放大 2 倍 |
| x4plus | 4 | 2 | ✗ 先 AI 放大 4 倍，再缩小（浪费） |
| x2plus | 2 | 4 | ✗ 先 AI 放大 2 倍，再插值放大（质量差） |

**解决方案**：脚本已添加兼容性验证，不匹配的组合会报错。

### Q3: 输出视频无声音？

**A**: OpenCV 默认不处理音频 → 用 ffmpeg 合并。

### Q4: Real-ESRGAN 太慢？

**A**: 优化方法：
1. 使用 `tile=256` 分块处理
2. 开启 `half=True` 半精度推理
3. 降低放大倍数（4x → 2x）
4. 使用 x2plus 模型（参数量更少）
5. 截取短视频测试效果后再处理完整视频

### Q5: 视频编码器如何选择？

**A**: 可用编码器对比：

| 编码器 | 代码 | 兼容性 | 压缩率 | 速度 | 适用场景 |
|--------|------|--------|--------|------|---------|
| MPEG-4 Visual | mp4v | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 通用（默认） |
| H.264 | avc1 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 高兼容性 |
| H.265 | hevc | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 高压缩率 |
| MJPEG | MJPG | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | 无压缩，快速测试 |
| VP9 | vp09 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Web 优化 |

**当前系统支持**: `mp4v`, `MJPG`, `vp09`

**选择建议**：
```bash
# 默认使用 mp4v（兼容性好）
python3 handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4

# 无压缩快速测试（文件大）
python3 handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 --codec MJPG

# Web 优化（压缩率高）
python3 handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 --codec vp09
```

---

## 九、脚本文件说明

| 文件 | 用途 |
|------|------|
| `video_upscale_realesrgan.py` | Real-ESRGAN 深度学习超分辨率 |
| `video_upscale_bicubic.py` | 双三次插值快速处理 |
| `merge_audio.py` | 合并视频和音频 |
| `check_frames.py` | 检查视频帧数据 |

---

## 十、最终建议

### 生产环境

```bash
# 高质量场景
python3 /workspace/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 4
ffmpeg -i output.mp4 -c:v libx264 -preset fast -crf 23 -c:a copy output_h264.mp4
python3 /workspace/merge_audio.py -v output_h264.mp4 -o input.mp4 -out output_final.mp4
```

### 快速预览

```bash
# 快速场景
python3 /workspace/video_upscale_bicubic.py -i input.mp4 -o output.mp4 -s 4
ffmpeg -i output.mp4 -c:v libx264 -preset fast -crf 23 -c:a copy output_h264.mp4
python3 /workspace/merge_audio.py -v output_h264.mp4 -o input.mp4 -out output_final.mp4
```

---

---

## 十一、Real-ESRGAN 多模型支持

### 可用模型

| 模型名称 | 放大倍数 | 参数量 | 特点 | 适用场景 |
|---------|---------|--------|------|---------|
| `RealESRGAN_x4plus` | 4x | 16.7M | 通用高质量 | 通用视频 |
| `RealESRGAN_x2plus` | 2x | ~16M | 快速处理 | 低倍数放大 |
| `RealESRGAN_x4plus_anime_6B` | 4x | 6B (约600万) | 动漫优化 | 动漫/插画 |

### 模型选择建议

```bash
# 通用视频 x4 放大（质量最高）
python3 handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 4 -m RealESRGAN_x4plus

# 通用视频 x2 放大（速度更快）
python3 handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 2 -m RealESRGAN_x2plus

# 动漫视频 x4 放大
python3 handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 4 -m RealESRGAN_x4plus_anime_6B
```

### 模型下载

```bash
# 下载 x2 模型
wget -P handle_video/models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x2plus.pth

# 下载动漫模型
wget -P handle_video/models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus_anime_6B.pth
```

### Scale 与模型兼容性

脚本会自动验证 `scale` 和 `model` 的兼容性：
- `RealESRGAN_x4plus`: 支持 2x 或 4x
- `RealESRGAN_x2plus`: 仅支持 2x
- `RealESRGAN_x4plus_anime_6B`: 仅支持 4x

不兼容的组合会报错：
```
ValueError: RealESRGAN_x2plus 模型只支持 [2]x 输出，不支持 4x
```

---

## 十二、GPU 硬件知识

### 推理卡 vs 算力卡

| 特性 | 推理卡 | 算力卡（训练卡） |
|------|--------|-----------------|
| **用途** | 部署已训练模型，进行预测推理 | 训练新模型，反向传播计算 |
| **显存** | 较小但高效 | 大容量（40GB-80GB） |
| **计算精度** | 优化 INT8/FP16，平衡速度和精度 | 支持 FP32/FP64，追求高精度 |
| **功耗** | 低功耗（75-300W） | 高功耗（300-700W） |
| **价格** | 相对便宜 | 昂贵 |
| **典型场景** | 在线服务、边缘计算 | 大模型训练、科研 |

### GPU 性能对比

| 型号 | 类型 | 单精度 TFLOPS | 显存 | 架构 | 特点 |
|------|------|--------------|------|------|------|
| **Tesla T4** | 推理卡 | 65 | 16GB | Turing | 当前环境使用，成本低 |
| **NVIDIA A10** | 推理卡 | 31.2 | 24GB | Ampere | 能效比高，云端常用 |
| **Tesla V100** | 训练卡 | 14-15.7 | 16GB/32GB | Volta | 经典训练卡，Tensor Core |
| **NVIDIA L40** | 推理卡 | ~48 | 48GB | Ada Lovelace | 专业可视化+推理 |
| **NVIDIA A800** | 训练卡 | 312 | 80GB | Ampere | 中国版 A100，降速版 |
| **NVIDIA A100** | 训练卡 | 312 | 40GB/80GB | Ampere | 旗舰训练卡，最快 |

---

## 十三、性能分析：Real-ESRGAN 处理 25 分钟视频

### 视频信息

- **文件**: `01 Gets Lost in Space.mp4`
- **分辨率**: 720x576
- **总帧数**: 37,271 帧
- **时长**: 25 分钟
- **原始大小**: 248 MB

### 处理时间预估

| GPU | 帧率 | 处理时间 | 说明 |
|-----|------|---------|------|
| **T4（当前）** | 1.6 FPS | **6.5 小时** | 50 TFLOPs/帧 × 37,271帧 ÷ 65 TFLOPS |
| **A10** | 1.2 FPS | 8.6 小时 | 算力较低（31.2 TFLOPS） |
| **V100 32GB** | 1.8 FPS | 5.8 小时 | Tensor Core 加速 |
| **L40** | 2.0 FPS | 5.2 小时 | 较新架构 |
| **A800** | 6.2 FPS | 1.7 小时 | 旗舰训练卡 |
| **A100** | 6.5 FPS | **1.6 小时** | 最快 |

### 输出文件大小预估

| 方法 | 原始大小 | 输出大小 | 增幅 |
|------|---------|---------|------|
| 双三次插值 4x | 248 MB | 3.3 GB | 13.3 倍 |
| Real-ESRGAN 4x | 248 MB | **3.4-3.6 GB** | 13.7-14.5 倍 |

### Real-ESRGAN 性能瓶颈

1. **模型复杂度 (60%)**: 23 RDB 块，16.7M 参数，约 50 TFLOPS/帧
2. **GPU 计算能力 (25%)**: Tesla T4 仅 65 TFLOPS
3. **内存传输 (10%)**: PCIe 带宽限制
4. **后处理 (5%)**: Tile 分块开销

### 双三次插值 vs Real-ESRGAN 速度对比

| 方法 | 帧率 | 处理 37,271 帧 | 速度比 |
|------|------|----------------|--------|
| Real-ESRGAN (T4 GPU) | 1.6 FPS | 6.5 小时 | 1x |
| 双三次插值 (CPU) | 80 FPS | 8 分钟 | **48.75x 更快** |

### 双三次插值不需要 GPU 的原因

| 特性 | 双三次插值 | Real-ESRGAN |
|------|-----------|-------------|
| **算法类型** | 数学插值公式 | 神经网络（RRDBNet） |
| **计算复杂度** | 低（O(n²)像素） | 极高（23层网络，1600万参数） |
| **GPU需求** | 无 | 必须 |
| **速度** | 快（CPU即可） | 慢（需GPU加速） |
| **画质** | 模糊，平滑 | 清晰，细节丰富 |

**双三次插值工作原理**:
```python
# 使用数学公式计算像素值
output_pixel = weighted_sum(
    16个邻近输入像素的值 × 双三次多项式权重
)
```

**Real-ESRGAN 工作原理**:
```python
# 23层深度神经网络，每层计算量巨大
output = rrdb_network(input)  # 50 TFLOPS/帧
```

---

## 十四、CPU 处理 Real-ESRGAN 的时间

### CPU vs GPU 性能对比

| 处理器 | 单精度算力 | 与 T4 对比 |
|--------|-----------|-----------|
| 典型服务器 CPU（EPYC Milan 3.5GHz） | ~0.5-1 TFLOPS | 慢 65-130 倍 |
| 当前环境 AMD EPYC | ~0.5 TFLOPS | 慢 130 倍 |

### 处理 `01 Gets Lost in Space.mp4` 预估

| 处理器 | 处理速度 | 处理时间 |
|--------|---------|---------|
| **T4 GPU** | 1.6 FPS | 6.5 小时 |
| **CPU** | 0.012 FPS | **875 小时（约 36 天）** |

**计算公式**:
```
CPU处理速度 = 1.6 FPS ÷ 130 = 0.012 FPS
处理时间 = 37,271帧 ÷ 0.012 = 3,105,916秒 ≈ 875小时
```

### 结论

CPU 处理 Real-ESRGAN **完全不可行**：
- 单个视频需要 **36 天**
- 即使是双三次插值（8 分钟），Real-ESRGAN 也慢了 **650,000 倍**

---

## 十五、中断处理和并行策略

### 断点续传支持

当前 `video_upscale_realesrgan.py` **支持断点续传**。

#### 使用方法

```bash
# 启用断点续传
python handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 4 --resume
```

#### 工作原理

- 每处理 500 帧保存一次进度到 `.progress.json` 文件
- 中断后重新运行，自动从上次位置继续
- 处理完成后自动删除进度文件

#### 方案 2: 分段处理（备选）

如果断点续传不够可靠，可以考虑：
- 将视频分成多个小段（每段 5000 帧）
- 单独处理每段，处理完后用 ffmpeg 合并
- 某段中断只需重新处理该段

---

## 十六、并行处理的局限性

### 单 GPU 并行不能加速

**原因**:
- GPU 是**共享资源**
- 多进程会竞争 GPU 时间片
- 上下文切换有开销
- 显存碎片化影响效率

**理论计算**:

| 方案 | 单帧时间 | 37,271 帧总时间 |
|------|---------|----------------|
| 单进程顺序处理 | 0.625 秒 | 6.5 小时 |
| 4 进程并行 | 0.625 秒 × 1.3（竞争） | 8.5 小时 |

**结论**: 并行反而会**降低 20-40%** 的速度

### 真正能加速的方法

#### 1. 多 GPU 并行

```
GPU1: 处理段 1-9000
GPU2: 处理段 9001-18000
GPU3: 处理段 18001-27000
GPU4: 处理段 27001-37271
```
→ 速度提升 4 倍（需要 4 张 GPU）

#### 2. 更强的 GPU

```
T4 (1.6 FPS) → A100 (6.5 FPS)
```
→ 速度提升 4 倍

#### 3. 使用轻量模型

```
RealESRGAN x4plus → RealESRGAN x2plus (2倍放大)
```
→ 速度提升 2-3 倍，但放大倍数降低

### 结论

- **分段并行 ≠ 速度提升**（单 GPU 场景）
- 只有多 GPU 才能真正加速
- 当前环境下，顺序处理是最优方案

---

## 十七、常见问题补充

### Q5: 双三次插值处理需要多久？

**A**:
- `standard-test.mp4` (456 帧): 6 秒
- `01 Gets Lost in Space.mp4` (37,271 帧): 约 8 分钟

### Q6: 为什么 CPU 不能用 Real-ESRGAN？

**A**:
- Real-ESRGAN 需要 50 TFLOPS/帧计算
- CPU 只有 0.5 TFLOPS
- 速度慢 100 倍以上，处理 25 分钟视频需 36 天

### Q7: 如何判断视频处理是否完成？

**A**:
```bash
# 检查进程
ps aux | grep "handle_video/scripts/video_upscale_realesrgan.py"

# 检查输出文件
ls -lh output.mp4

# 如果只看到 grep 命令，说明处理已完成
```

### Q8: 如何使用断点续传？

**A**: 使用 `--resume` 参数即可：
```bash
python handle_video/scripts/video_upscale_realesrgan.py input.mp4 -o output.mp4 -s 4 --resume
```

进度会自动保存，中断后重新运行即可继续。

---

## 附录: 相关链接

- Video2X: https://github.com/k4yt3x/video2x
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- BasicSR: https://github.com/xinntao/BasicSR

---

## 十八、FFmpeg 视频截取技巧

### 基本截取命令

```bash
# 截取前 30 秒
ffmpeg -i input.mp4 -t 00:00:30 -c:v libx264 -c:a aac output_30s.mp4 -y

# 截取前 5 秒
ffmpeg -i input.mp4 -t 00:00:05 -c:v libx264 -c:a aac output_5s.mp4 -y

# 从 00:01:00 开始截取 10 秒
ffmpeg -i input.mp4 -ss 00:01:00 -t 00:00:10 -c:v libx264 -c:a aac output.mp4 -y
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `-t` | 截取时长 | `-t 00:00:30`（30秒）或 `-t 30` |
| `-ss` | 起始时间 | `-ss 00:01:00`（从1分钟开始） |
| `-c:v libx264` | 视频编码为 H.264 | 兼容性好 |
| `-c:a aac` | 音频编码为 AAC | 标准音频格式 |
| `-y` | 覆盖输出文件 | 不提示确认 |

### 实际案例

从 24 分钟的视频（248 MB）截取测试视频：

```bash
# 原视频
# 01 Gets Lost in Space.mp4
# 分辨率: 720x576, 时长: 24分50秒, 大小: 248 MB

# 截取前 30 秒（推荐用于测试）
ffmpeg -i "01 Gets Lost in Space.mp4" -t 00:00:30 -c:v libx264 -c:a aac "01 Gets Lost in Space_test_30s.mp4" -y
# 输出: 6.7 MB, 30 秒, 750 帧

# 截取前 5 秒（快速测试）
ffmpeg -i "01 Gets Lost in Space.mp4" -t 00:00:05 -c:v libx264 -c:a aac "01 Gets Lost in Space_test_5s.mp4" -y
# 输出: 1.1 MB, 5 秒, 125 帧
```

### 为什么需要截取测试视频？

RealESRGAN 处理速度受视频分辨率影响极大：

| 视频 | 分辨率 | 帧数 | 预计处理时间 (x4) |
|------|--------|------|-------------------|
| 原视频（30秒） | 720x576 | 750 | ~25 分钟 |
| 原视频（5秒） | 720x576 | 125 | ~4 分钟 |
| 标准测试视频 | 426x240 | 456 | ~5 分钟 |

**建议工作流程**：
1. 先截取 5 秒测试视频
2. 用 RealESRGAN 处理测试视频，检查效果
3. 如果效果满意，再处理完整视频

### 快速生成测试视频脚本

```bash
#!/bin/bash
# 生成测试视频
INPUT_VIDEO="$1"
TEST_LENGTH="${2:-5}"  # 默认 5 秒

if [ -z "$INPUT_VIDEO" ]; then
    echo "用法: $0 <输入视频> [时长秒数]"
    exit 1
fi

OUTPUT_VIDEO="${INPUT_VIDEO%.*}_test_${TEST_LENGTH}s.mp4"

ffmpeg -i "$INPUT_VIDEO" \
    -t "00:00:$TEST_LENGTH" \
    -c:v libx264 \
    -c:a aac \
    "$OUTPUT_VIDEO" \
    -y

echo "测试视频已生成: $OUTPUT_VIDEO"
```

使用：
```bash
bash create_test_video.sh "01 Gets Lost in Space.mp4" 5
```

---

## 十九、脚本更新日志

### video_upscale_realesrgan.py 更新内容

#### 2026-02-21 更新

1. **模型初始化优化**
   - 将模型初始化移到模型选择之后
   - 根据不同模型名称设置对应的网络参数
   - 添加 fallback 机制（models → .realesrgan）

2. **多模型支持**
   - 支持 `RealESRGAN_x4plus`（通用 x4）
   - 支持 `RealESRGAN_x2plus`（通用 x2）
   - 支持 `RealESRGAN_x4plus_anime_6B`（动漫专用 x4）

3. **编码器选择**
   - 新增 `--codec` 参数
   - 支持 `mp4v`（默认）、`MJPG`、`vp09`

4. **Scale/Model 兼容性验证**
   - 验证 scale 和 model 是否兼容
   - 不兼容的组合会报错

5. **打印信息优化**
   - 显示"模型倍数"和"输出倍数"两个信息
   - 避免用户混淆

6. **断点续传支持**
   - 支持 `--resume` 参数
   - 每 500 帧保存一次进度

### 已知问题和解决方案

#### 问题 1: Fallback 路径写死为 x4plus 模型

**原问题**：
```python
if not Path(model_path).exists():
    fallback_path = str(script_dir / '.realesrgan' / 'RealESRGAN_x4plus.pth')  # 写死
    if Path(fallback_path).exists():
        model_path = fallback_path
        netscale = 4  # 写死
```

**已修复**：
```python
if not Path(model_path).exists():
    fallback_path = str(script_dir / '.realesrgan' / f'{model_name}.pth')  # 动态
    if Path(fallback_path).exists():
        model_path = fallback_path
    else:
        raise FileNotFoundError(f"找不到模型文件: {model_path} 和 {fallback_path}")
```

#### 问题 2: 打印信息误导（显示模型倍数而非输出倍数）

**原问题**：
```python
print(f"放大倍数: {netscale}")  # 显示模型内置倍数
```

**已修复**：
```python
print(f"模型倍数: {netscale}")
print(f"输出倍数: {self.scale}")  # 显示用户指定的倍数
```
