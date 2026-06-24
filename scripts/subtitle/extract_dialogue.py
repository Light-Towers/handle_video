#!/usr/bin/env python3
"""
从 SRT 字幕文件中提取纯文本对话
"""

import re


def extract_dialogue_from_srt(srt_file_path):
    """从SRT文件中提取字幕文本"""
    encodings_to_try = ['utf-8', 'gb18030', 'gbk', 'latin-1']

    for encoding in encodings_to_try:
        try:
            with open(srt_file_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"使用编码: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        with open(srt_file_path, 'r', encoding='latin-1', errors='ignore') as f:
            content = f.read()

    # 使用正则表达式提取字幕文本
    lines = content.split('\n')
    text_lines = []

    for line in lines:
        line = line.strip()
        if line.isdigit():
            continue
        if '-->' in line:
            continue
        if not line:
            continue
        text_lines.append(line)

    return text_lines


def main():
    # 输入和输出文件路径
    srt_file = '/workspace/台词/srt/01 Gets Lost in Space.srt'
    output_file = '/workspace/台词/txt/01 Gets Lost in Space_dialogue.txt'

    print(f"正在读取字幕文件: {srt_file}")
    text_lines = extract_dialogue_from_srt(srt_file)

    print(f"提取到 {len(text_lines)} 行文本")
    print(f"正在保存到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in text_lines:
            f.write(line + '\n')

    print("完成!")
    print("\n前10行预览:")
    for i, line in enumerate(text_lines[:10], 1):
        print(f"{i}. {line}")


if __name__ == '__main__':
    main()
