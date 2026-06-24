# 音视频操作命令手册

> 整理自会话历史中所有涉及的音视频操作命令，按功能类别分类，记录每条命令的具体功能、参数及使用场景。
>
> 整理时间：2026-06-24

---

## 目录

1. [视频信息检查](#1-视频信息检查)
2. [视频元数据查询](#2-视频元数据查询)
3. [视频合并与拼接](#3-视频合并与拼接)
4. [静音音频添加](#4-静音音频添加)
5. [图片转视频](#5-图片转视频)
6. [视频重新编码](#6-视频重新编码)
7. [字幕烧录](#7-字幕烧录)
8. [音频提取](#8-音频提取)
9. [SRT 转 TXT](#9-srt-转-txt)
10. [视频截图](#10-视频截图)
11. [文件管理与清理](#11-文件管理与清理)
12. [最佳实践](#12-最佳实践)

---

## 1. 视频信息检查

### 1.1 完整视频信息查询

```bash
ffprobe -v error -show_format -show_streams "input.mkv"
```

- **功能**：查看视频的完整格式信息和所有流信息
- **参数**：
  - `-v error`：只显示错误输出
  - `-show_format`：显示容器格式信息
  - `-show_streams`：显示所有流信息
- **场景**：完整检查视频文件，确认编码、分辨率、时长、音频等信息

### 1.2 检查视频流信息

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,dar -of default=noprint_wrappers=1 "input.mkv"
```

- **功能**：查看视频流编码、分辨率、宽高比
- **参数**：
  - `-select_streams v:0`：仅选择第一个视频流
  - `-show_entries stream=codec_name,width,height,dar`：显示指定字段
- **场景**：快速确认视频分辨率是否变化（如从 2880×2304 变为 1344×768）

### 1.3 检查音频流信息

```bash
ffprobe -v error -select_streams a -show_entries stream=codec_name,channels "input.mkv"
```

- **功能**：查看音频流编码和声道数
- **场景**：检查视频是否包含音频轨道、音频编码格式

### 1.4 检查音频是否存在

```bash
ffprobe -v error -select_streams a -show_entries stream=codec_name -of default=noprint_wrappers=1 "input.mkv"
```

- **功能**：快速检查是否有音频流
- **场景**：排查播放器死机问题——发现 OP 合并后音频丢失

### 1.5 查看流数量

```bash
ffprobe -v error -show_format -show_streams "input.mkv" | grep -E "(duration|codec_name|channels|Stream)"
```

- **功能**：简洁摘要
- **场景**：快速查看 `nb_streams` 确认是否为单流视频

---

## 2. 视频元数据查询

### 2.1 查看文件大小

```bash
ls -lh "file.mkv"
```

- **功能**：查看文件大小（人类可读格式）

### 2.2 查看文件创建时间

```bash
ls -l --time-style=full-iso "file.mkv"
```

- **功能**：查看文件的完整时间戳
- **场景**：确定视频文件是什么时候生成的

### 2.3 图片尺寸查询

```bash
identify "image.png"
```

或使用 Python：

```bash
python3 -c "from PIL import Image; img = Image.open('image.png'); print(f'尺寸: {img.size[0]}x{img.size[1]}')"
```

- **功能**：查看原始图片尺寸
- **场景**：确认 OP/ED 图片初始尺寸（1344×768），用于计算缩放参数

---

## 3. 视频合并与拼接

### 3.1 Concat Demuxer 方式（推荐，无重新编码）

```bash
# 1. 创建文件列表
cat > /tmp/merge_list.txt << 'EOF'
file '/path/to/video1.mkv'
file '/path/to/video2.mkv'
file '/path/to/video3.mkv'
EOF

# 2. 合并
ffmpeg -f concat -safe 0 -i /tmp/merge_list.txt -c copy "output.mkv" -y
```

- **功能**：无缝拼接多个视频，不重新编码
- **参数**：
  - `-f concat`：使用 concat 分隔器
  - `-safe 0`：允许使用绝对路径
  - `-c copy`：复制流，不重新编码
- **⚠️ 重要前提**：所有视频必须有**完全相同**的编码参数（编码格式、分辨率、帧率、音频编码等）
- **场景**：合并 OP + 主视频 + ED1 + ED2 + ED3

### 3.2 ❌ 不推荐：Concat Filter 方式

```bash
ffmpeg -i "video1.mkv" -i "video2.mkv" -filter_complex \
  "[0:v][1:v]concat=n=2:v=1:a=0[outv]" \
  -map "[outv]" -map 1:a -c:v libx264 -preset medium -crf 23 -c:a copy "output.mkv" -y
```

- **问题**：需要重新编码，速度极慢（对于高分辨率视频容易失败/killed）
- **场景**：仅在流参数不一致时使用

### 3.3 合并失败原因总结

| 失败现象 | 原因 | 解决方案 |
|---------|------|---------|
| 音频丢失 | 第一个文件无音频，concat demuxer 忽略后续音频 | 确保所有文件都有音频流 |
| 分辨率错误 | OP/ED 缩放到目标尺寸时不匹配 | 使用正确的 scale + pad 滤镜 |
| 播放器死机 | 缺少音频 + 分辨率异常 | 合并前检查所有文件的流信息 |
| 编码进程 killed | 高分辨率视频重新编码耗资源 | 使用 `-c copy` 代替重新编码 |

---

## 4. 静音音频添加

### 4.1 为无声视频添加静音音频轨道

```bash
ffmpeg -i "video_without_audio.mkv" \
  -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
  -c:v copy -c:a ac3 -b:a 128k -shortest \
  "video_with_audio.mkv" -y
```

- **功能**：为无音频视频添加静音轨道，确保与有音频视频合并时不丢失音频
- **参数**：
  - `-f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000`：生成立体声 48kHz 静音源
  - `-c:v copy`：视频流直接复制不编码
  - `-c:a ac3 -b:a 128k`：音频编码为 AC3，128kbps
  - `-shortest`：以较短的流为准结束输出
- **场景**：OP/ED 图片视频本身无音频，合并前需添加静音轨道

### 4.2 批量添加静音音频

```bash
ffmpeg -i "op.mkv" -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
  -c:v copy -c:a ac3 -b:a 128k -shortest "op_with_audio.mkv" -y && \
ffmpeg -i "ed_1.mkv" -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
  -c:v copy -c:a ac3 -b:a 128k -shortest "ed_1_with_audio.mkv" -y && \
ffmpeg -i "ed_2.mkv" -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
  -c:v copy -c:a ac3 -b:a 128k -shortest "ed_2_with_audio.mkv" -y && \
ffmpeg -i "ed_3.mkv" -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
  -c:v copy -c:a ac3 -b:a 128k -shortest "ed_3_with_audio.mkv" -y
```

- **场景**：一次性为多个片头片尾视频添加音频

---

## 5. 图片转视频

### 5.1 ❌ 错误方式：放大 + 裁剪（内容被裁切）

```bash
ffmpeg -loop 1 -i "image.png" \
  -vf "scale=2880:2304:force_original_aspect_ratio=increase,crop=2880:2304,fps=25" \
  -t 10 -pix_fmt yuv420p \
  -c:v libx264 -preset ultrafast -an \
  "output.mkv" -y
```

- **问题**：图片从 1344×768 放大到 2880×2304 时两侧被裁切，内容不完整
- **原理**：
  - `force_original_aspect_ratio=increase`：先放大到超出目标尺寸
  - `crop=2880:2304`：再从中心裁剪，导致两边内容丢失

### 5.2 ✅ 正确方式：缩放 + 黑边填充（保留完整内容）

```bash
ffmpeg -loop 1 -i "image.png" \
  -vf "scale=2880:1620:force_original_aspect_ratio=decrease,pad=2880:2304:(ow-iw)/2:(oh-ih)/2:black,fps=25" \
  -t 10 -pix_fmt yuv420p \
  -c:v libx264 -preset ultrafast -an \
  "output.mkv" -y
```

- **功能**：将图片转为视频，保持完整内容，用黑边填充
- **参数详解**：
  - `-loop 1`：循环单张图片
  - `scale=2880:1620:force_original_aspect_ratio=decrease`：缩放到不超出目标宽度
  - `pad=2880:2304:(ow-iw)/2:(oh-ih)/2:black`：居中 + 黑边填充
  - `-t 10`：生成 10 秒视频
  - `-pix_fmt yuv420p`：标准像素格式（兼容性好）
  - `-c:v libx264 -preset ultrafast`：快速 H.264 编码
  - `-an`：不包含音频
- **计算示例**：
  - 原始图片：1344×768，宽高比 1.75
  - 目标：2880×2304，宽高比 1.25
  - 缩放后：4032×2304 → 上下居中，两侧添加黑边

### 5.3 Preset 速度对比

| Preset | 编码速度 | 压缩效率 | 输出大小 | 用途 |
|--------|---------|---------|---------|------|
| `ultrafast` | 最快（2-5倍速） | 最低 | 最大 | 调试/静态图片转视频 |
| `fast` | 快 | 中等 | 中等 | 一般用途 |
| `medium` | 中等（默认） | 中等 | 中等 | 最终输出推荐 |
| `slow` | 慢 | 高 | 小 | 最终高质量输出 |

- **ultrafast 原理**：跳过复杂编码决策——无 B 帧、无多参考帧、最简运动估计、跳过率失真优化（RDO）

---

## 6. 视频重新编码

### 6.1 MKV 转 MP4（实时转码）

```bash
ffmpeg -i "input.mkv" \
  -c:v libx264 -preset fast -crf 23 \
  -c:a aac -b:a 128k \
  "output.mp4" -y
```

- **功能**：将 MKV 视频转为 MP4 容器
- **参数**：
  - `-c:v libx264`：视频编码器
  - `-preset fast`：编码速度预设
  - `-crf 23`：恒定质量参数（0=无损，23=默认，51=最差）
  - `-c:a aac -b:a 128k`：音频转为 AAC 128kbps
- **示例耗时**：10 秒 2880×2304 视频转换约需 13-14 秒（0.76x 实时）

### 6.2 重新编码（快速测试版）

```bash
ffmpeg -i "input.mkv" \
  -c:v libx264 -preset ultrafast -crf 28 \
  -c:a aac -b:a 128k -threads 4 \
  "output.mp4" -y
```

- **参数**：
  - `-preset ultrafast`：最快编码（牺牲文件大小）
  - `-crf 28`：较低质量（更快）
  - `-threads 4`：限制 CPU 线程
- **场景**：快速测试输出

### 6.3 高兼容性 MP4 编码

```bash
ffmpeg -i "input.mkv" \
  -c:v libx264 -preset fast -crf 23 \
  -profile:v baseline -level 3.0 \
  -c:a aac -b:a 128k \
  -movflags +faststart \
  "output.mp4" -y
```

- **参数**：
  - `-profile:v baseline -level 3.0`：基线配置文件，兼容最广泛
  - `-movflags +faststart`：moov atom 前置，支持流式播放
- **⚠️ 注意**：高分辨率视频（如 2880×2304）可能因资源不足被 killed

---

## 7. 字幕烧录

### 7.1 将 ASS 字幕烧录到视频

```bash
ffmpeg -i "input.mkv" \
  -vf "subtitles=filename='subtitle.ass':force_style='FontName=Arial,FontSize=18,PrimaryColour=&HFFFFFF,OutlineColour=&H00000000,BorderStyle=1'" \
  -c:v libx264 -preset ultrafast -crf 30 \
  -c:a aac -b:a 96k -threads 4 \
  "output.mp4" -y
```

- **功能**：将 ASS/SSA 字幕硬编码烧录到视频中
- **参数**：
  - `subtitles=filename='...'`：指定字幕文件路径
  - `force_style='...'`：覆盖字幕样式

---

## 8. 音频提取

### 8.1 从视频提取音频为 MP3

```bash
ffmpeg -i "input.mkv" -vn -c:a libmp3lame -b:a 128k "output.mp3" -y
```

- **功能**：从视频文件中提取音频轨道并转为 MP3
- **参数**：
  - `-vn`：排除视频流
  - `-c:a libmp3lame`：使用 MP3 编码器
  - `-b:a 128k`：音频比特率 128kbps
- **典型输出**：24:50 视频 → ~23MB MP3

### 8.2 提取音频为原始格式（不转码）

```bash
ffmpeg -i "input.mkv" -vn -c:a copy "output.ac3"
```

- **功能**：直接复制音频流，不重新编码
- **场景**：需要保持原始音频质量

---

## 9. SRT 转 TXT

### 9.1 提取纯台词文本

```bash
sed '/^[0-9]*$/d; /^[[:space:]]*$/d; /-->/d' "input.srt" > "output.txt"
```

- **功能**：从 SRT 字幕文件中提取纯文本台词
- **参数详解**：
  - `/^[0-9]*$/d`：删除纯数字行（行号）
  - `/^[[:space:]]*$/d`：删除空白行
  - `/-->/d`：删除时间戳行
- **场景**：将字幕转为纯文本台词列表

**典型效果**：
- `02 For Lunch.srt`（1501行）→ `02 For Lunch_dialogue.txt`（821行）
- `03 Inside Ralphie.srt`（1491行）→ `03 Inside Ralphie_dialogue.txt`（813行）

---

## 10. 视频截图

### 10.1 从视频指定时间提取一帧

```bash
ffmpeg -i "input.mkv" -ss 00:00:01 -vframes 1 -vf scale=200:-1 -f image2 "output.png" -y
```

- **功能**：提取视频中指定时间点的一帧并缩放
- **参数**：
  - `-ss 00:00:01`：跳转到第 1 秒
  - `-vframes 1`：只提取 1 帧
  - `-vf scale=200:-1`：缩放宽度到 200px，高度自动计算
- **场景**：检查视频帧的实际内容

---

## 11. 文件管理与清理

### 11.1 删除错误/废弃的视频文件

```bash
# 删除指定文件
rm -f "wrong_file.mkv" "another_wrong.mkv"

# 删除一类文件
rm -f ed_1_fixed.mkv ed_1_with_audio.mkv ed_2_fixed.mkv ed_2_with_audio.mkv
```

### 11.2 临时文件管理策略

**建议组织方式**：
```
项目目录/
├── final/            ← 仅存放最终版本文件
├── temp/             ← 临时生成的文件（完成后可整体删除）
└── source/           ← 原始文件
```

### 11.3 查看目录中特定类型文件

```bash
ls -lh /path/to/dir/ | grep -E "\.(mkv|mp4)$"
```

### 11.4 批量查找文件

```bash
find /workspace -iname "*02*For Lunch*" -name "*.mkv" -o -iname "*02*For Lunch*" -name "*.mp4"
```

---

## 12. 最佳实践

### 12.1 视频合并流程（完整模板）

```bash
# === 准备工作 ===
# 1. 检查所有文件的流信息，确保编码一致
for f in op.mkv main.mkv ed_1.mkv ed_2.mkv ed_3.mkv; do
  echo "=== $f ==="
  ffprobe -v error -select_streams a -show_entries stream=codec_name "$f"
  ffprobe -v error -select_streams v -show_entries stream=width,height,codec_name "$f"
done

# 2. 为无音频的视频添加静音音轨
for f in op.mkv ed_1.mkv ed_2.mkv ed_3.mkv; do
  output="${f%.mkv}_with_audio.mkv"
  ffmpeg -i "$f" -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
    -c:v copy -c:a ac3 -b:a 128k -shortest "$output" -y
done

# 3. 创建合并列表
cat > /tmp/merge.txt << EOF
file '/path/to/op_with_audio.mkv'
file '/path/to/main.mkv'
file '/path/to/ed_1_with_audio.mkv'
file '/path/to/ed_2_with_audio.mkv'
file '/path/to/ed_3_with_audio.mkv'
EOF

# 4. 合并
ffmpeg -f concat -safe 0 -i /tmp/merge.txt -c copy "final.mkv" -y

# 5. 验证
ffprobe -v error -show_format -show_streams "final.mkv"
```

### 12.2 图片转视频 + 合并流程

```bash
# 1. 图片转视频（保留完整内容，黑边填充）
ffmpeg -loop 1 -i "op.png" \
  -vf "scale=TARGET_W:TARGET_H_WITH_ASPECT:force_original_aspect_ratio=decrease,\
       pad=TARGET_W:TARGET_H:(ow-iw)/2:(oh-ih)/2:black,fps=25" \
  -t 10 -pix_fmt yuv420p -c:v libx264 -preset ultrafast -an \
  "op_video.mkv" -y

# 2. 添加静音音频
ffmpeg -i "op_video.mkv" \
  -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
  -c:v copy -c:a ac3 -b:a 128k -shortest \
  "op_final.mkv" -y

# 3. 与其他视频合并
# ... 使用 concat demuxer
```

### 12.3 故障排查清单

| 检查项 | 命令 |
|-------|------|
| 视频能否播放 | 播放测试 |
| 音频是否存在 | `ffprobe -select_streams a input.mkv` |
| 分辨率是否正确 | `ffprobe -select_streams v -show_entries stream=width,height` |
| 音视频同步 | 对比 `duration` 值 |
| 编码是否一致 | 检查所有文件的 `codec_name` |
| 文件是否完整 | 对比预期时长与实际时长 |

### 12.4 常见 Preset 对比

| preset | 速度乘数 | 文件大小 | 质量 | 推荐场景 |
|--------|---------|---------|------|---------|
| ultrafast | ~5x | 140% | 略低 | 快速预览、静态图转视频 |
| veryfast | ~3x | 120% | 良好 | 快速编码 |
| fast | ~2x | 115% | 很好 | 一般使用 |
| medium | 1x | 100% | 很好 | **最终输出推荐** |
| slow | 0.5x | 90% | 最好 | 高质量存档 |
| veryslow | 0.25x | 85% | 最好 | 终极质量 |

### 12.5 文件命名规范建议

```
{集号} {标题}_Final.mkv              ← 最终成品
{集号} {标题}.mkv                     ← 原始视频
{集号} {标题}_with_subs.mkv           ← 烧录字幕版
op_final_correct.mkv                  ← 片头（带音频）
ed_{1-3}_final_correct.mkv            ← 片尾（带音频）
```

---

## 附录：命令速查表

| 操作 | 命令模板 |
|------|---------|
| 查看视频信息 | `ffprobe -v error -show_format -show_streams "file.mkv"` |
| 提取音频 | `ffmpeg -i "file.mkv" -vn -c:a libmp3lame -b:a 128k "file.mp3" -y` |
| 添加静音音频 | `ffmpeg -i "file.mkv" -f lavfi -i anullsrc=... -c:v copy -c:a ac3 -shortest "out.mkv" -y` |
| 图片转视频 | `ffmpeg -loop 1 -i "img.png" -vf "scale=...:force_original_aspect_ratio=decrease,pad=..." -t 10 -c:v libx264 -preset ultrafast -an "out.mkv" -y` |
| Concat 合并 | `ffmpeg -f concat -safe 0 -i list.txt -c copy "out.mkv" -y` |
| MKV 转 MP4 | `ffmpeg -i "in.mkv" -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k "out.mp4" -y` |
| SRT 转 TXT | `sed '/^[0-9]*$/d; /^[[:space:]]*$/d; /-->/d' "in.srt" > "out.txt"` |
| 查看文件大小 | `ls -lh "file.mkv"` |
