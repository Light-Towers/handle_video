#!/usr/bin/env python3
"""
从 SRT 字幕文件中提取纯文本
"""

import re
import sys


def extract_text_from_srt(srt_file_path):
    """从SRT文件中提取字幕文本"""
    # 优先尝试 GBK 编码（中文Windows常见编码）
    encodings_to_try = ['gbk', 'gb18030', 'utf-8', 'latin-1']

    for encoding in encodings_to_try:
        try:
            with open(srt_file_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"使用编码: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        # 如果所有编码都失败，使用 errors='ignore'
        with open(srt_file_path, 'r', encoding='latin-1', errors='ignore') as f:
            content = f.read()

    # 使用正则表达式提取字幕文本
    # SRT格式: 序号 -> 时间码 -> 字幕文本 -> 空行
    lines = content.split('\n')
    text_lines = []

    for line in lines:
        line = line.strip()
        # 跳过序号行（纯数字）
        if line.isdigit():
            continue
        # 跳过时间码行（包含 --> 的行）
        if '-->' in line:
            continue
        # 跳过空行
        if not line:
            continue
        # 添加文本行
        text_lines.append(line)

    return text_lines


def save_text(text_lines, output_file):
    """将提取的文本保存到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in text_lines:
            f.write(line + '\n')


def main():
    # 输入和输出文件路径
    srt_file = '/workspace/data/videos/04 Gets Eaten.srt'
    output_file = '/workspace/data/videos/04 Gets Eaten.txt'

    print(f"正在读取字幕文件: {srt_file}")
    text_lines = extract_text_from_srt(srt_file)

    print(f"提取到 {len(text_lines)} 行文本")
    print(f"正在保存到: {output_file}")
    save_text(text_lines, output_file)

    print("完成!")
    print("\n前10行预览:")
    for i, line in enumerate(text_lines[:10], 1):
        print(f"{i}. {line}")


if __name__ == '__main__':
    main()
