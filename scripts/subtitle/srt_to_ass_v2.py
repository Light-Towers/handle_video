#!/usr/bin/env python3
"""
将 SRT 字幕转换为带颜色和加粗效果的 ASS 字幕
重点词汇会自动识别并高亮显示
"""

import re
import json
import os


def load_keywords(keyword_file):
    """
    从 JSON 文件加载关键词配置
    """
    with open(keyword_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 转换格式为脚本内部格式
    keyword_categories = {}
    for category, data in config.items():
        keyword_categories[category] = {
            'name': data.get('name', category),
            'color': data['color'],
            'bold': data.get('bold', True),
            'description': data.get('description', ''),
            'words': data['words']
        }

    return keyword_categories


def parse_srt_time(time_str):
    """
    解析 SRT 时间格式: 00:01:11,760 -> (hours, minutes, seconds, milliseconds)
    """
    # SRT格式: HH:MM:SS,mmm
    match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', time_str)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        milliseconds = int(match.group(4))
        return hours, minutes, seconds, milliseconds
    return 0, 0, 0, 0


def time_to_ass(hours, minutes, seconds, milliseconds):
    """
    将时间转换为 ASS 时间格式: H:MM:SS.cc (百分之一秒)
    """
    # 毫秒转百分之一秒 (四舍五入)
    centiseconds = round(milliseconds / 10)
    if centiseconds >= 100:
        centiseconds = 99
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


# 重点词汇列表 - 这些会被高亮
KEYWORD_CATEGORIES = {
    'planets': {
        'name': '行星名称',
        'color': '&H00FFFF',
        'bold': True,
        'description': '太阳系中的行星和天体',
        'words': ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter',
                 'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Planet', 'Planets']
    },
    'space': {
        'name': '空间术语',
        'color': '&H0000FF',
        'bold': True,
        'description': '与太空、宇宙相关的专业术语',
        'words': ['Space', 'Orbit', 'Orbiting', 'Outer Space', 'Solar System',
                 'Universe', 'Galaxy', 'Star', 'Asteroid', 'Meteorite']
    },
    'science': {
        'name': '科技术语',
        'color': '&H00FF00',
        'bold': True,
        'description': '科学领域的专业词汇',
        'words': ['Gravity', 'Atmosphere', 'Temperature', 'Gas', 'Clouds',
                 'Solid', 'Liquid', 'Acid', 'Oxygen', 'Carbon']
    },
    'highlight': {
        'name': '重点强调',
        'color': '&H0080FF',
        'bold': True,
        'description': '需要重点强调的词汇',
        'words': ['Alien', 'Aliens', 'LIFE', 'Proof', 'Prove', 'Adventure']
    }
}


def highlight_keywords(text):
    """
    在文本中标记重点词汇
    返回带 ASS 标签的文本
    """
    # 收集所有需要高亮的词及其类别
    all_words = {}
    for category, config in KEYWORD_CATEGORIES.items():
        for word in config['words']:
            all_words[word.lower()] = config

    # 按词长降序排序,避免部分匹配问题
    sorted_words = sorted(all_words.items(), key=lambda x: len(x[0]), reverse=True)

    # 构建匹配模式
    result = text

    # 先找到所有需要标记的位置(从原始文本)
    highlights = []
    for word_lower, config in sorted_words:
        pattern = r'\b' + re.escape(word_lower) + r'\b'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # 检查是否与已有高亮重叠
            start = match.start()
            end = match.end()
            overlapped = False
            for h_start, h_end, _, _, _ in highlights:
                if not (end <= h_start or start >= h_end):
                    overlapped = True
                    break
            if not overlapped:
                highlights.append((start, end, match.group(), config['color'], config['bold']))

    # 按起始位置排序,从后往前替换
    highlights.sort(key=lambda x: x[0], reverse=True)

    for start, end, original, color, bold in highlights:
        bold_tag = '\\b1' if bold else '\\b0'
        tagged = f'{{\\1c{color}{bold_tag}}}{original}{{\\r}}'
        result = result[:start] + tagged + result[end:]

    return result


def parse_srt(srt_content):
    """
    解析 SRT 内容,返回字幕条目列表
    每个条目: (index, start_time, end_time, text)
    """
    entries = []
    lines = srt_content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # 跳过空行
        if not line:
            i += 1
            continue

        # 序号行
        if line.isdigit():
            index = int(line)
            i += 1

            # 时间码行
            if i < len(lines):
                time_line = lines[i].strip()
                i += 1

                # 解析时间
                time_match = re.match(
                    r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
                    time_line
                )
                if time_match:
                    start_time = time_match.group(1)
                    end_time = time_match.group(2)

                    # 读取字幕文本(可能多行)
                    text_lines = []
                    while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                        text_lines.append(lines[i].rstrip())
                        i += 1

                    text = '\\N'.join(text_lines)  # ASS用\\N换行
                    entries.append((index, start_time, end_time, text))

        i += 1

    return entries


def generate_ass_header():
    """
    生成 ASS 文件头
    """
    header = """[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    return header


def convert_srt_to_ass(srt_content):
    """
    将 SRT 内容转换为 ASS 格式
    """
    # 解析 SRT
    entries = parse_srt(srt_content)

    # 生成 ASS 内容
    ass_content = generate_ass_header()

    dialogue_count = 0
    for index, start_time_str, end_time_str, text in entries:
        # 解析时间
        sh, sm, ss, sms = parse_srt_time(start_time_str)
        eh, em, es, ems = parse_srt_time(end_time_str)

        # 转换为 ASS 时间格式
        start_ass = time_to_ass(sh, sm, ss, sms)
        end_ass = time_to_ass(eh, em, es, ems)

        # 高亮关键词
        highlighted_text = highlight_keywords(text)

        # 生成 Dialogue 行
        dialogue_count += 1
        ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{highlighted_text}\n"

    return ass_content, dialogue_count


def main():
    # 输入和输出文件路径
    srt_file = '/workspace/台词/srt/01 Gets Lost in Space.srt'
    ass_file = '/workspace/台词/srt/01 Gets Lost in Space.ass'
    keyword_file = '/workspace/keywords.json'

    print("=" * 60)
    print("SRT 转 ASS 字幕转换工具")
    print("=" * 60)
    print(f"\n输入文件: {srt_file}")
    print(f"输出文件: {ass_file}")
    print(f"关键词配置: {keyword_file}")

    # 检查关键词文件是否存在
    if os.path.exists(keyword_file):
        print(f"\n正在从 JSON 文件加载关键词配置...")
        KEYWORD_CATEGORIES.update(load_keywords(keyword_file))
        print(f"已加载 {len(KEYWORD_CATEGORIES)} 个分类")
    else:
        print(f"\n未找到关键词配置文件 {keyword_file}")
        print("使用内置关键词配置...")

    # 读取 SRT 文件
    print("\n正在读取 SRT 文件...")
    try:
        with open(srt_file, 'r', encoding='utf-8') as f:
            srt_content = f.read()
    except UnicodeDecodeError:
        # 尝试其他编码
        with open(srt_file, 'r', encoding='gb18030', errors='ignore') as f:
            srt_content = f.read()
        print("使用 GB18030 编码读取")

    print(f"文件大小: {len(srt_content)} 字节")

    # 转换为 ASS
    print("\n正在转换为 ASS 格式...")
    print("关键词高亮规则:")
    for category, config in KEYWORD_CATEGORIES.items():
        name = config.get('name', category)
        color = config['color']
        word_count = len(config['words'])
        print(f"  {name} ({category}): {word_count} 个词 (颜色: {color})")

    ass_content, dialogue_count = convert_srt_to_ass(srt_content)

    # 保存 ASS 文件
    print(f"\n正在保存 ASS 文件...")
    with open(ass_file, 'w', encoding='utf-8-sig') as f:
        f.write(ass_content)

    print(f"\n转换完成!")
    print(f"生成 {dialogue_count} 条字幕")
    print(f"输出文件: {ass_file}")
    print("\n提示: 将 .ass 文件与 .mkv 视频放在同一目录,播放器会自动加载!")
    print(f"\n提示: 编辑 {keyword_file} 可自定义关键词配置")


if __name__ == '__main__':
    main()
