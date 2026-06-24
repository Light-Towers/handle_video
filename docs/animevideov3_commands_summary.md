# realesr-animevideov3 视频处理命令总结

## 完整处理流程

### 1. 仅使用 realesr-animevideov3 模型进行超分（无降噪、锐化、补帧）

```bash
cd /workspace/handle_video/scripts
python video_upscale_realesrgan_denoise.py \
  /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s.mp4 \
  -o /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s_animevideov3_only.mp4 \
  -m realesr-animevideov3 \
  --denoise 0
```

**参数说明：**
- `-m realesr-animevideov3`: 指定使用 realesr-animevideov3 模型
- `--denoise 0`: 禁用去噪（设置为 0 表示不做额外去噪处理）
- `-o`: 输出文件路径

---

### 2. 添加音频到增强后的视频

```bash
ffmpeg -i /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s_animevideov3_only.mp4 \
  -i /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s.mp4 \
  -c:v copy \
  -c:a aac \
  -map 0:v:0 \
  -map 1:a:0 \
  -shortest \
  /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s_animevideov3_only_audio.mp4 \
  -y
```

**参数说明：**
- `-i`: 输入文件（第一个是增强后的视频，第二个是原视频）
- `-c:v copy`: 直接复制视频流（不重新编码）
- `-c:a aac`: 音频使用 AAC 编码
- `-map 0:v:0`: 使用第一个输入的视频流
- `-map 1:a:0`: 使用第二个输入的音频流
- `-shortest`: 以最短的流为准
- `-y`: 覆盖输出文件

---

### 3. 使用 RIFE 进行帧插值补帧（2x 补帧，25fps → 50fps）

```bash
cd /workspace/handle_video/scripts
python rife_interpolate_simple.py \
  --video /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s_animevideov3_audio.mp4 \
  --output /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s_animevideov3_rife.mp4 \
  --model-dir /workspace/handle_video/RIFE/train_log \
  --scale 0.5
```

**参数说明：**
- `--video`: 输入视频路径
- `--output`: 输出视频路径
- `--model-dir`: RIFE 模型目录
- `--scale 0.5`: 使用 0.5x 缩放进行推理以减少内存占用

---

### 4. 为 RIFE 补帧后的视频添加音频

```bash
ffmpeg -i /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s_animevideov3_rife.mp4 \
  -i /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s.mp4 \
  -c:v copy \
  -c:a aac \
  -map 0:v:0 \
  -map 1:a:0 \
  -shortest \
  /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s_animevideov3_rife_final.mp4 \
  -y
```

---

## 完整流程（一键执行）

如果你想一次性执行完整流程（超分 + 音频 + 补帧 + 音频）：

```bash
# 步骤1: 使用 realesr-animevideov3 进行超分
cd /workspace/handle_video/scripts && \
python video_upscale_realesrgan_denoise.py \
  /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s.mp4 \
  -o /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s_animevideov3_only.mp4 \
  -m realesr-animevideov3 \
  --denoise 0

# 步骤2: 添加音频
ffmpeg -i /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s_animevideov3_only.mp4 \
  -i /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s.mp4 \
  -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest \
  /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s_animevideov3_audio.mp4 \
  -y

# 步骤3: RIFE 补帧
cd /workspace/handle_video/scripts && \
python rife_interpolate_simple.py \
  --video /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s_animevideov3_audio.mp4 \
  --output /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s_animevideov3_rife.mp4 \
  --model-dir /workspace/handle_video/RIFE/train_log \
  --scale 0.5

# 步骤4: 最终添加音频
ffmpeg -i /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s_animevideov3_rife.mp4 \
  -i /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s.mp4 \
  -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest \
  /workspace/data/videos/01\ Gets\ Lost\ in\ Space_test_2m20s_animevideov3_rife_final.mp4 \
  -y
```

---

## 脚本参数说明

### video_upscale_realesrgan_denoise.py 参数

```bash
python video_upscale_realesrgan_denoise.py [输入视频] [选项]

选项:
  -o, --output       输出视频路径
  -s, --scale        放大倍数 (2 或 4)，默认: 4
  -m, --model        模型名称，默认: RealESRGAN_x4plus
                     可选: realesr-animevideov3,
                            RealESRGAN_x4plus,
                            RealESRGAN_x2plus,
                            RealESRGAN_x4plus_anime_6B
  --denoise          去噪强度 (0-1)，默认: 0.5
                     0 = 不去噪
                     1 = 强去噪
  --tile-size        分块大小，默认: 512
                     可选: 0, 256, 512, 768, 1024
  --no-cuda          禁用 CUDA
  --no-sharpen       禁用后处理锐化
```

---

## 常用命令组合

### 仅超分（推荐）
```bash
python video_upscale_realesrgan_denoise.py \
  输入视频.mp4 \
  -m realesr-animevideov3 \
  --denoise 0
```

### 超分 + 去噪
```bash
python video_upscale_realesrgan_denoise.py \
  输入视频.mp4 \
  -m realesr-animevideov3 \
  --denoise 0.3
```

### 最高质量超分（不分块）
```bash
python video_upscale_realesrgan_denoise.py \
  输入视频.mp4 \
  -m realesr-animevideov3 \
  --denoise 0 \
  --tile-size 0
```

### 快速处理
```bash
python video_upscale_realesrgan_denoise.py \
  输入视频.mp4 \
  -m realesr-animevideov3 \
  --denoise 0 \
  --tile-size 256
```

---

## 处理结果

### animevideov3_only_audio.mp4
- 分辨率: 2880x2304 (4x 超分)
- 帧率: 25 FPS
- 时长: 10 秒
- 帧数: 250 帧
- 处理: 仅 realesr-animevideov3 超分
- 包含音频: ✅

### animevideov3_rife_final.mp4
- 分辨率: 2880x2304 (4x 超分)
- 帧率: 50 FPS (2x 补帧)
- 时长: 10 秒
- 帧数: 499 帧
- 处理: realesr-animevideov3 超分 + RIFE 补帧
- 包含音频: ✅

---

## 文件位置

**脚本：**
- `/workspace/handle_video/scripts/video_upscale_realesrgan_denoise.py`
- `/workspace/handle_video/scripts/rife_interpolate_simple.py`

**模型：**
- `/workspace/handle_video/models/realesr-animevideov3.pth`
- `/workspace/handle_video/RIFE/train_log/` (RIFE 模型)

**输入输出：**
- 输入: `/workspace/data/videos/01 Gets Lost in Space_test_2m20s.mp4`
- 输出: `/workspace/data/videos/01 Gets Lost in Space_test_2m20s_*.mp4`

---

## 处理时间参考

使用 realesr-animevideov3 处理 10 秒视频（250 帧）：
- 处理时间: 约 105 秒
- 平均速度: 约 2.4 FPS
- 输出分辨率: 2880x2304

使用 RIFE 补帧处理 10 秒视频（250 帧 → 499 帧）：
- 处理时间: 约 20-30 秒
- 输出帧率: 50 FPS

---
