# 读取SRT文件
with open('/workspace/data/台词/srt/01 Gets Lost in Space.srt', 'r', encoding='utf-8') as f:
    content = f.read()

# 应用所有修改
replacements = [
    # 第7行
    ('ok, Ralphie', 'Ok, Ralphie'),

    # 第11行
    ("lt's going to be a model of the solar system.", "It's going to be a model of the solar system."),

    # 第16行
    ('l hate to tell you', 'I hate to tell you'),

    # 第20-21行
    ('sun was so', 'the sun was so'),
    ('flagpole!', 'the flagpole!'),

    # 第25行
    ('And you had all nine planets?', 'And you had all nine planets?'),

    # 第45-46行
    ('l betcha your class', 'I bet your class'),
    ('ALlENS that live on', 'aliens that live on'),

    # 第54行
    ('ONLy planet', 'the only planet'),

    # 第63行
    ("'cause l got straight A's", 'because I got straight A'),

    # 第72行
    ('lf you already know', 'If you already know'),

    # 第86行
    ('and l quote', 'and I quote'),

    # 第90行
    ('what l said', 'what I said'),

    # 继续添加更多修改...
]

# 应用替换
for old, new in replacements:
    content = content.replace(old, new)

# 保存修改后的文件
with open('/workspace/data/台词/srt/01 Gets Lost in Space.srt', 'w', encoding='utf-8') as f:
    f.write(content)

print("SRT文件修改完成")
print(f"应用了 {len(replacements)} 处修改")
