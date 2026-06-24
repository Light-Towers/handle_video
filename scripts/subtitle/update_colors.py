import re

# 读取文件
with open('/workspace/data/台词/srt/01 Gets Lost in Space.ass', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 需要标记为红色的词汇（按长度降序排序，避免部分匹配）
red_words = [
    'solar system',
    'asteroid belt',
    'sulphuric acid',
    'sunblock',
    'artificial',
    'navigator',
    'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto',
    'alien',
    'outrageous',
    'rotting',
    'astronaut',
    'orbit',
    'gravity',
    'meteorite',
    'marvelous',
    'rust',
    'chore',
    'surrender',
    'hint',
    'blotch',
    'riddle',
    'fiction',
    'temporary'
]

# 处理每一行
result_lines = []
for line in lines:
    if not line.startswith('Dialogue:'):
        # 非台词行，直接添加
        result_lines.append(line)
        continue

    # 台词行：删除所有现有的颜色标记
    clean_line = line

    # 删除所有 {\1c&H...} 标记（包括后面的 \b1 或其他标记）
    clean_line = re.sub(r'\{\\1c&H[0-9A-Fa-f]+\\b[01]\}', '', clean_line)
    clean_line = re.sub(r'\{\\1c&H[0-9A-Fa-f]+\}', '', clean_line)
    # 删除所有 {\r} 标记
    clean_line = re.sub(r'\{\\r\}', '', clean_line)
    # 删除所有 {\b1} 和 {\b0} 标记
    clean_line = re.sub(r'\{\\b[01]\}', '', clean_line)

    # 给指定词汇添加红色标记
    # 红色颜色代码: {\1c&H0000FF\b1}word{\r}
    for word in sorted(red_words, key=len, reverse=True):
        # 使用单词边界匹配
        pattern = r'\b' + re.escape(word) + r'\b'
        clean_line = re.sub(pattern, r'{\\1c&H0000FF\\b1}' + word + r'{\\r}', clean_line)

    result_lines.append(clean_line)

# 保存文件
with open('/workspace/data/台词/srt/01 Gets Lost in Space.ass', 'w', encoding='utf-8') as f:
    f.writelines(result_lines)

print("完成！已删除所有颜色标记并给指定词汇添加红色标记")
print(f"处理了 {len(red_words)} 个词汇")
