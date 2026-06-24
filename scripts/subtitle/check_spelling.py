#!/usr/bin/env python3
"""扫描 SRT 文件中的拼写错误"""
import re
from collections import Counter

# 读取 SRT 文件
with open('/workspace/data/台词/srt/01 Gets Lost in Space.srt', 'r', encoding='utf-8') as f:
    content = f.read()

# 提取纯文本内容
text_lines = []
for line in content.split('\n'):
    if re.match(r'^\d+$', line):
        continue
    if re.match(r'^\d{2}:\d{2}:\d{2}', line):
        continue
    if line.strip():
        text_lines.append(line)

# 统计单词
words = []
for line in text_lines:
    words.extend(re.findall(r'\b[a-zA-Z]+\b', line.lower()))

word_count = Counter(words)

print("=" * 70)
print("SRT 文件拼写问题扫描结果")
print("=" * 70)

# 超长单词
long_words = [(w, c) for w, c in word_count.items() if len(w) > 15]
print(f"\n【超长单词】({len(long_words)} 个)")
print("-" * 70)
for word, count in sorted(long_words, key=lambda x: x[1], reverse=True):
    print(f"  {word:35} {count:3} 次")

# 查找连写的情况
print("\n【疑似连写错误】(驼峰命名)")
print("-" * 70)
for line in text_lines:
    matches = re.findall(r'\b[a-z]+[A-Z][a-z]+\b', line)
    if matches:
        for m in matches:
            print(f"  {m:35} -> {line.strip()[:60]}")

# 查找特定错误
print("\n【特定错误模式】")
print("-" * 70)
errors = {
    'totallyoutrageous': 'totally outrageous',
    'meteorttes': 'meteorites',
    'meteoRlTES': 'meteorites',
}

for line in text_lines:
    for wrong, correct in errors.items():
        if wrong.lower() in line.lower():
            print(f"  发现错误: {wrong:20} -> {correct:20}")
            print(f"    上下文: {line.strip()}")
