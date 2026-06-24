#!/usr/bin/env python3
"""全面扫描 SRT 文件中的拼写问题"""
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

print("=" * 70)
print("SRT 文件全面拼写问题扫描")
print("=" * 70)

# 1. 大小写混乱问题（如 REALLy）
print("\n【1. 大小写混乱问题】")
print("-" * 70)
mixed_case = []
for line in text_lines:
    # 查找模式：前大写后小写混合的常见词
    if re.search(r'\b[A-Z]{2,}[a-z]+\b', line):
        for match in re.finditer(r'\b[A-Z]{2,}[a-z]+\b', line):
            word = match.group()
            # 排除正常的大写缩写词
            if word not in ['NASA', 'Ms.', 'Mr.', 'Dr.', 'St.']:
                if len(word) > 4:
                    mixed_case.append((word, line.strip()))

if mixed_case:
    seen = set()
    for word, context in mixed_case:
        if word not in seen:
            print(f"  {word:25} -> {context[:60]}")
            seen.add(word)
else:
    print("  未发现")

# 2. 超长单词（可能是连写）
print("\n【2. 超长单词（>20字符）】")
print("-" * 70)
words = []
for line in text_lines:
    words.extend(re.findall(r'\b[a-zA-Z]+\b', line))

word_count = Counter(words)
long_words = [(w, c) for w, c in word_count.items() if len(w) > 20]

if long_words:
    for word, count in sorted(long_words, key=lambda x: x[1], reverse=True):
        print(f"  {word:35} {count:3} 次")
else:
    print("  未发现")

# 3. 可能的连写（驼峰命名）
print("\n【3. 疑似连写错误（驼峰命名）】")
print("-" * 70)
camel_case = []
for line in text_lines:
    matches = re.findall(r'\b[a-z]+[A-Z][a-z]+\b', line)
    if matches:
        for m in matches:
            # 排除正常的人名
            if m not in ['Frizzle', 'Carlos', 'Arnold', 'Keesha', 'Phoebe',
                        'Dorothy', 'Ann', 'Ralphie', 'Wanda', 'Tim']:
                camel_case.append((m, line.strip()))

if camel_case:
    seen = set()
    for word, context in camel_case:
        if word not in seen:
            print(f"  {word:25} -> {context[:60]}")
            seen.add(word)
else:
    print("  未发现")

# 4. 小写开头的专有名词
print("\n【4. 小写开头的专有名词】")
print("-" * 70)
proper_nouns = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter',
                'saturn', 'uranus', 'neptune', 'pluto', 'frizzle', 'ralphie',
                'arnold', 'wanda', 'keesha', 'phoebe', 'carlos', 'tim']

lowercase_proper = []
for line in text_lines:
    for pn in proper_nouns:
        pattern = r'\b' + pn + r'\b'
        if re.search(pattern, line, re.IGNORECASE):
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                word = match.group()
                if word.islower() and word in pn:
                    lowercase_proper.append((word, line.strip()))

if lowercase_proper:
    seen = set()
    for word, context in lowercase_proper:
        if word not in seen:
            print(f"  {word:25} -> {context[:60]}")
            seen.add(word)
else:
    print("  未发现")

# 5. 重复字母（如 meteorr)
print("\n【5. 重复字母错误】")
print("-" * 70)
duplicate_letters = []
for line in text_lines:
    # 查找连续3个以上相同字母
    if re.search(r'(.)\1{2,}', line.lower()):
        for match in re.finditer(r'(.)\1{2,}', line.lower()):
            word = match.group()
            # 排除正常的如 'lll' 等
            if len(word) >= 3:
                duplicate_letters.append((word, line.strip()))

if duplicate_letters:
    seen = set()
    for word, context in duplicate_letters:
        if word not in seen:
            print(f"  {word:25} -> {context[:60]}")
            seen.add(word)
else:
    print("  未发现")

# 6. 常见拼写错误模式
print("\n【6. 常见错误模式】")
print("-" * 70)
error_patterns = {
    r'([aeiou])\1+': '元音重复过多',  # 如 reeally
    r'([bcdfghjklmnpqrstvwxyz])\1{3,}': '辅音重复过多',  # 如 mmmmmm
    r'\b[a-z]+[^aeioubcdfghjklmnpqrstvwxyz\s\']+[a-z]+\b': '包含特殊字符',
}

for pattern, desc in error_patterns.items():
    for line in text_lines:
        if re.search(pattern, line):
            print(f"  {desc:20} -> {line.strip()[:60]}")
            break

# 7. 全大写单词（除了正常强调的）
print("\n【7. 全大写单词】")
print("-" * 70)
all_caps = []
for line in text_lines:
    matches = re.findall(r'\b[A-Z]{2,}\b', line)
    # 排除缩写和正常词
    skip = ['OK', 'NASA', 'NO', 'YES', 'AND', 'THE', 'BUT', 'FOR', 'WITH']
    if matches:
        for m in matches:
            if m not in skip:
                all_caps.append((m, line.strip()))

if all_caps:
    seen = set()
    for word, context in all_caps[:10]:  # 只显示前10个
        if word not in seen:
            print(f"  {word:25} -> {context[:60]}")
            seen.add(word)
else:
    print("  未发现")
