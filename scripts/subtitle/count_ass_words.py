import re

# 读取 ASS 文件
with open('/workspace/台词/srt/01 Gets Lost in Space.ass', 'r', encoding='utf-8') as f:
    content = f.read()

# 提取所有高亮的单词
pattern = r'\{\\1c&H[0-9A-F]*\\b[01]\}([^{]+)\{\\r\}'
matches = re.findall(pattern, content)

# 统计每个单词出现的次数
from collections import Counter
word_count = Counter(matches)

# 按颜色分类
color_pattern = r'\{\\1c(&H[0-9A-F]*)\\b[01]\}([^{]+)\{\\r\}'
all_matches = re.findall(color_pattern, content)

colors = {
    '&H00FFFF': {'name': '行星名称', 'words': []},
    '&H0000FF': {'name': '空间术语', 'words': []},
    '&H00FF00': {'name': '科技术语', 'words': []},
    '&HFF00FF': {'name': '物理术语', 'words': []},
    '&H00FF80': {'name': '化学术语', 'words': []},
    '&HFF8000': {'name': '生物术语', 'words': []},
    '&H808000': {'name': '地质术语', 'words': []},
    '&H0080FF': {'name': '重点强调', 'words': []},
}

for color, word in all_matches:
    if color in colors:
        colors[color]['words'].append(word)

# 打印结果
print("=" * 70)
print("ASS 字幕文件中标记的单词统计")
print("=" * 70)

for color_code in sorted(colors.keys()):
    if colors[color_code]['words']:
        print(f"\n【{colors[color_code]['name']}】({color_code})")
        print("-" * 70)
        for word, count in Counter(colors[color_code]['words']).most_common():
            print(f"  {word:25} {count:3} 次")

print("\n" + "=" * 70)
print(f"总计: {len(word_count)} 个不同单词, {len(matches)} 个标记")
print("=" * 70)
