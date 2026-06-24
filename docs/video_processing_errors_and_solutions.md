# 视频处理错误记录与解决方案

## 1. ASS字幕样式问题

### 问题：字幕显示为斜体，看不清楚
**原因**：
- 迅雷影音可能强制使用自己的样式，覆盖了ASS文件设置
- 字体大小设置过大（初始设为28号）

**解决方案**：
- 将字体改为Arial，大小设为18号
- 标红文字设为22号（红色+粗体）

---

## 2. 字幕颜色不显示问题

### 问题：迅雷影音中字幕不显示颜色
**原因**：
- FFmpeg对某些ASS样式标记支持不完整
- 颜色恢复标记使用不当

**尝试的方案**：
1. `{\b0\fs18}` - 只关闭粗体和设置字体大小，**颜色未恢复**（错误）
2. `{\r}` - **正确方案**，完全重置所有样式

**解决方案**：
```
{\1c&H0000FF\b1\fs22}关键词{\r}
```
- 使用 `{\r}` 重置标记，而不是单独恢复每个属性

---

## 3. 音画不同步问题

### 问题：截取视频后音画不同步
**原因**：
- `-ss` 参数放在 `-i` 前面，导致时间戳不准确

**错误命令**：
```bash
ffmpeg -ss 00:01:10 -i input.mkv ...
```

**正确命令**：
```bash
ffmpeg -i input.mkv -ss 00:01:10 ...
```
- 将 `-ss` 参数放在输入文件后面，使用精确的帧级时间戳

---

## 4. 字幕文件未被修改

### 问题：执行ffmpeg命令后，ASS字幕文件内容丢失
**原因**：
- 从SRT重新生成ASS文件，覆盖了之前修改的内容
- 没有在修改前备份

**教训**：
- **修改文件前必须备份**
- 保存修改记录，以便重新应用

---

## 5. 视频拼接格式错误

### 问题：concat滤镜参数配置错误
**错误**：
```bash
-filter_complex "[0:v]format=yuv420p,fps=25[v1];[v1][1:v]concat=n=1:v=1:a=0[outv]"
```
- `n=1` 只匹配1个输入，实际有2个

**解决方案**：
- 不使用concat滤镜，直接生成独立视频片段
- 使用concat协议（`-f concat`）合并已存在的视频文件

---

## 6. 图片转视频失败

### 问题：生成的MKV文件大小为0
**原因**：
1. 图片尺寸（1654×2339）与视频不一致（2880×2304）
2. 没有正确设置像素格式

**错误命令**：
```bash
ffmpeg -loop 1 -i input.png -t 10 -c:v libx264 output.mkv
```

**正确命令**：
```bash
ffmpeg -loop 1 -i input.png \
  -vf "scale=2880:2304:force_original_aspect_ratio=increase,crop=2880:2304,fps=25" \
  -t 10 -pix_fmt yuv420p \
  -c:v libx264 -preset ultrafast -an output.mkv
```

**关键点**：
- 使用 `scale` 和 `crop` 滤镜调整尺寸到目标分辨率
- 设置 `-pix_fmt yuv420p` 确保兼容性
- 使用 `-an` 禁用音频（静态图片不需要）
- `-preset ultrafast` 加快处理速度

---

## 7. 进程被杀死

### 问题：ffmpeg进程被系统杀死
**原因**：
- 内存不足
- 同时处理多个任务

**解决方案**：
- 逐个处理视频片段，避免并发
- 使用 `-preset ultrafast` 降低内存占用

---

## 8. Heredoc语法问题

### 问题：使用heredoc创建文件列表失败
**错误**：
```bash
cat > /tmp/list.txt << 'EOF'
file 'path1.mkv'
file 'path2.mkv'
EOF
```
- heredoc的结束标记 `EOF` 前不能有空格

**解决方案**：
```bash
echo "file '/path/to/file1.mkv'" > /tmp/list.txt
echo "file '/path/to/file2.mkv'" >> /tmp/list.txt
```
- 使用多个echo命令追加写入，更可靠

---

## 9. 播放器死机问题

### 问题：生成的视频在播放器中死机
**原因**：
1. **缺少音频轨道**：合并视频时，OP和ED片段没有音频，导致concat demuxer丢失了主视频的音频
2. **分辨率错误**：之前的合并导致视频从2880×2304被错误缩放到1344×768
3. **编码不一致**：不同的编码参数导致播放器兼容性问题

**错误示例**：
- `01 Gets Lost in Space_with_OP.mkv` 只有视频流，没有音频流
- 分辨率从2880×2304变成1344×768

**解决方案**：

### 步骤1：为OP和ED添加静音音频
```bash
# 为OP添加静音音频
ffmpeg -i "/workspace/data/videos/ai/op_intro_fixed.mkv" \
  -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
  -c:v copy -c:a ac3 -b:a 128k -shortest \
  "/workspace/data/videos/ai/op_intro_with_audio.mkv"

# 为ED添加静音音频（3个）
ffmpeg -i "/workspace/data/videos/ai/ed_1_fixed.mkv" \
  -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
  -c:v copy -c:a ac3 -b:a 128k -shortest \
  "/workspace/data/videos/ai/ed_1_with_audio.mkv"

# ed_2 和 ed_3 同理
```

**关键点**：
- 使用 `anullsrc` 生成静音音频源
- 保持音频参数一致：stereo, 48000Hz
- 使用 `-shortest` 确保音频长度与视频匹配

### 步骤2：使用concat demuxer合并
```bash
# 创建文件列表
cat > /tmp/final_merge.txt << 'EOF'
file '/workspace/data/videos/ai/op_intro_with_audio.mkv'
file '/workspace/data/videos/ai/01 Gets Lost in Space_with_subs.mkv'
file '/workspace/data/videos/ai/ed_1_with_audio.mkv'
file '/workspace/data/videos/ai/ed_2_with_audio.mkv'
file '/workspace/data/videos/ai/ed_3_with_audio.mkv'
EOF

# 合并视频
ffmpeg -f concat -safe 0 -i /tmp/final_merge.txt \
  -c copy "/workspace/data/videos/ai/01 Gets Lost in Space_Final.mkv" -y
```

**关键点**：
- 确保所有输入文件都有视频和音频流
- 使用 `-c copy` 直接复制流，不重新编码
- 验证分辨率一致性（2880×2304）

### 步骤3：验证最终视频
```bash
# 检查视频信息
ffprobe -v error -show_format -show_streams "/workspace/data/videos/ai/01 Gets Lost in Space_Final.mkv"

# 确认以下信息：
# - nb_streams=2（1视频+1音频）
# - width=2880, height=2304
# - 音视频时长基本一致（差值<0.1秒）
```

**最终结果**：
- 时长：25分31秒
- 分辨率：2880×2304
- 编码：H.264视频 + AC3音频
- 结构：OP(10s) → 主视频(24:50) → ED1(10s) → ED2(10s) → ED3(10s)

---

## 10. 视频重新编码失败

### 问题：尝试重新编码为MP4时进程被杀死
**原因**：
- 视频分辨率过高（2880×2304），编码参数过于严格
- 使用 `profile:v baseline -level 3.0` 不支持如此高分辨率

**错误命令**：
```bash
ffmpeg -i input.mkv -c:v libx264 -preset fast -crf 23 \
  -profile:v baseline -level 3.0 -c:a aac -b:a 128k \
  -movflags +faststart output.mp4
```

**解决思路**：
- 对于超高清视频，使用 `copy` 模式避免重新编码
- 或使用更宽松的profile设置（如High Profile）
- 分段处理或降低目标分辨率

---

## 最佳实践总结

### 字幕制作
1. **字体大小**：
   - 默认：18号
   - 标红：22号（红色+粗体）
2. **恢复标记**：始终使用 `{\r}` 而非单独属性
3. **字体选择**：Arial（英文），Microsoft YaHei（中文）

### 视频处理
1. **备份重要文件**：修改前总是备份
2. **时间戳截取**：`-ss` 放在 `-i` 之后
3. **分辨率统一**：所有视频片段使用相同分辨率（2880×2304）
4. **逐步处理**：避免并发任务，逐个处理

### 视频合并
1. **音频轨道一致性**：确保所有输入文件都有音频流
   - 有音频的片段保持原样
   - 无音频的片段（如OP/ED图片）添加静音音频
2. **分辨率一致性**：所有片段使用相同分辨率
3. **编码一致性**：使用相同编码格式（如H.264 + AC3）
4. **使用concat demuxer**：`-f concat -i list.txt -c copy`
   - 避免使用concat filter，除非需要重新编码
   - `-c copy` 直接复制流，速度快且无质量损失

### FFmpeg常用参数
| 参数 | 用途 |
|------|------|
| `-ss` | 截取时间点（放在-i后） |
| `-t` | 持续时长 |
| `-c:v libx264` | 视频编码器 |
| `-preset ultrafast` | 加快处理 |
| `-pix_fmt yuv420p` | 像素格式 |
| `-an` | 无音频 |
| `-f concat` | 合并协议 |
| `-c copy` | 直接复制流（不重新编码） |
| `-shortest` | 以最短流为准 |
| `-lavfi anullsrc` | 生成静音音频 |

### 添加静音音频的完整模板
```bash
ffmpeg -i "video_without_audio.mkv" \
  -f lavfi -i "anullsrc=channel_layout=stereo:sample_rate=48000" \
  -c:v copy -c:a ac3 -b:a 128k -shortest \
  "video_with_audio.mkv"
```

---

## ASS字幕标记速查

| 标记 | 作用 | 示例 |
|------|------|------|
| `\1c&H0000FF` | 主颜色（红色） | `{\1c&H0000FF}text` |
| `\b1` | 开启粗体 | `{\b1}text` |
| `\fs22` | 字体大小22 | `{\fs22}text` |
| `\r` | 重置所有样式 | `{\r}` |
| `\N` | 强制换行 | `line1\Nline2` |

颜色代码（BGR格式）：
- `&H0000FF` = 红色
- `&H00FF00` = 绿色
- `&HFF0000` = 蓝色
- `&HFFFFFF` = 白色

---

## 故障排查清单

当视频合并或处理出现问题时，按以下顺序检查：

1. **检查音频流**
   ```bash
   ffprobe -v error -select_streams a -show_entries stream=codec_name input.mkv
   ```

2. **检查分辨率**
   ```bash
   ffprobe -v error -select_streams v -show_entries stream=width,height input.mkv
   ```

3. **检查时长**
   ```bash
   ffprobe -v error -show_entries format=duration input.mkv
   ```

4. **验证文件完整性**
   ```bash
   ls -lh input.mkv  # 检查文件大小是否正常
   ```

5. **测试单独播放**
   - 确保每个片段都能单独正常播放
   - 特别注意OP和ED片段
