import re

# 读取文件
with open('/workspace/台词/srt/01 Gets Lost in Space.ass', 'r', encoding='utf-8') as f:
    content = f.read()

# 需要标记为红色的词汇（按长度降序排序）
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

# 第一阶段：删除所有现有的颜色标记
content = re.sub(r'\{\\1c&H[0-9A-Fa-f]+\\b[01]\}', '', content)
content = re.sub(r'\{\\1c&H[0-9A-Fa-f]+\}', '', content)
content = re.sub(r'\{\\r\}', '', content)
content = re.sub(r'\{\\b[01]\}', '', content)

# 第二阶段：应用所有文本修改（按对话历史记录）
replacements = [
    # 第13行
    (',0,0,0,0,,ok, Ralphie', ',0,0,0,0,,Ok, Ralphie'),

    # 第14行
    ("lt's going to be a model of the ", "It's going to be a model of the "),
    ('Neat, Janet?', 'Neat, huh, Janet?'),

    # 第15行
    ('l hate to tell you', 'I hate to tell you'),

    # 第16行
    ('but when my class built one, the sun was so', 'but when my class built one, the sun was so'),
    ('flagpole!', 'the flagpole!'),

    # 第22行
    ('l betcha your class', 'I bet your class'),
    ('ALlENS that live on', 'aliens that live on'),

    # 第24行
    ('ONLy planet', 'the only planet'),

    # 第26行
    ("'cause l got straight A's", 'because I got straight A'),

    # 第28行
    ('lf you already know', 'If you already know'),

    # 第31行
    ('and l quote', 'and I quote'),

    # 第32行
    ('what l said', 'what I said'),

    # 第39行
    ("l'd say", "I'd say"),

    # 第42行
    ('l wonder', 'I wonder'),

    # 第43行
    ('lnside', 'Inside'),

    # 第45行
    ('NORMAL', 'normal'),

    # 第46行
    ("THAT'S", "That's"),

    # 第47行
    ('lt\'s closed for the day.', 'Closed today.'),

    # 第49行
    ('Well, looks like we\'ll have to go back to', 'Well, looks like we\'ll just have to go back to'),

    # 第50行
    ('Oh no...', 'Oh, no!'),

    # 第54行
    ('than--', 'than your'),

    # 第57行
    ('Stop bus ,Ms. Frizzle', 'Stop the bus, Ms. Frizzle'),

    # 第59行
    ('lsn\'t there... you know... some place ELSE...', 'Isn\'t there... you know... someplace else...'),

    # 第64行
    ('ARNOLD! Why didn\'t l think', 'ARNOLD! Why didn\'t I think'),

    # 第67行
    ('A FlELD TRlP!', 'A FIELD TRIP!'),

    # 第68行
    ('lnto outer space?!?!', 'Into outer space?!?!'),

    # 第70行
    ('planetarium', 'planetarium'),

    # 第76行
    ('Wonderful! Let', 'Wonderful! Let the'),

    # 第79行
    ('sunblock 8,000 sun-goggles?', 'sunblock 8,000 sun-goggles?'),

    # 第80行
    ('l wonder', 'I wonder'),

    # 第83行
    ('and are now on our way to', 'and now we are on our way to'),

    # 第85行
    ('When l tell my class', 'When I tell my class'),

    # 第90行
    ('l wonder', 'I wonder'),

    # 第93行
    ('Bet l can', 'Bet I can'),

    # 第95行
    ('l won first place in my class\' jumping contest.', 'I won first place in my class\'s jumping contest.'),

    # 第96行
    ('Wait\'ll l tell my class l won the', 'Wait until I tell my class I won the'),

    # 第100行
    ('How on Earth... l mean... How\'m l going to', 'How on Earth... I mean... How am I going to'),

    # 第102行
    ('lt\'s too', 'It\'s too'),

    # 第103行
    ('REALLY', 'way below'),

    # 第106行
    ('SOMETHlNG had to make tracks this', 'SOMETHING had to make tracks this'),

    # 第107行
    ('meteorites', 'a meteorite'),

    # 第110行
    ('lf the ones that HlT the planet are called', 'If the ones that hit the planet are called'),

    # 第114行
    ('What are you doing, janet, watch it!', 'Janet, what are you doing? Hey! Watch it!'),

    # 第116行
    ('When l show this to my class, it\'ll PROVE l', 'When I show this to my class, it\'ll prove I'),

    # 第119行
    ('the Sun!', 'the Sun!'),

    # 第120行
    ('l can\'t wait! l can\'t wait! l can\'t wait!', 'I can\'t wait! I can\'t wait! I can\'t wait!'),

    # 第122行
    ('These clouds are pretty!', 'Wow! What a view!'),

    # 第123行
    ('l better sit down, l guess.', 'Uh, I think I\'ll go...sit down now.'),

    # 第125行
    ('Hey, l feel like l weigh the same here as l do', 'Hey, I feel like I weigh the same here as I do'),

    # 第130行
    ('cool the place down', 'cool the place down'),

    # 第132行
    ('Sulphuric Acid?', 'Sulphuric Acid?'),

    # 第135行
    ('l like rocks', 'I like rocks'),

    # 第137行
    ('lf this doesn\'t prove l was on', 'If this doesn\'t prove I was on'),

    # 第146行
    ('How magnificent!', 'How marvelous!'),

    # 第147行
    ('lt looks like', 'It looks like'),

    # 第149行
    ('coloured by', 'colored by'),

    # 第150行
    ('iceburg', 'cliffs of ice'),

    # 第151行
    ('lce climbing', 'Ice climbing'),

    # 第156行
    ('No, thanks, janet. I\'d stay at home if l knew we gotta do this.', 'Janet, stop! Maybe I should have stayed home today.'),

    # 第158行
    ('l wonder if', 'I wonder if'),
    ('weren\'t', 'wasn\'t'),

    # 第159行
    ('as l always say', 'as I always say'),

    # 第160行
    ('best place for ice cream', 'best place for I scream!'),

    # 第161行
    ('lce cream? Where?', 'Ice cream? Where?'),

    # 第162行
    ('Here! lCE CREAM!!', 'Here! I scream! Woo woo woo woo! Woo woo woo!'),

    # 第164行
    ('lCE CREAM!!', 'I scream!'),

    # 第165行
    ('the last ice block', 'That\'s the last ice block'),

    # 第167行
    ('On Earth, l had', 'On Earth, I had'),

    # 第171行
    ('ALlEN!', 'Alien!'),

    # 第175行
    ('l knew it was him the whole time', 'I knew it was him the whole time'),

    # 第179行
    ('l don\'t think', 'I don\'t think'),

    # 第181行
    ('That\'s no potato. lt\'s an asteroid, Ralphie', 'That\'s no potato. It\'s an asteroid, Ralphie'),

    # 第183行
    ('lt\'s part of the', 'It\'s part of the'),

    # 第186行
    ('l gotta have an asteroid!', 'Asteroid! Ha ha! I\'ve got to have one. Please. It will only take me a second.'),

    # 第192行
    ('lt\'s gone!', 'It\'s gone!'),

    # 第193-194
    ('LOST lN SPACE!!', 'LOST IN SPACE!!'),

    # 第195行
    ('As l always say', 'As I always say'),

    # 第198行
    ('l mean pliers!', 'I mean pliers!'),

    # # 第204行
    ('Keep your claw on that button, Liz. you\'ve given me the most WONDERFUL idea.', 'Up, up and away.'),

    # # 第205
    ('LOST lN SPACE', 'LOST IN SPACE'),

    # # 第206
    ('WlTHOUT A TEACHER!!!', 'WITHOUT A TEACHER!!!'),

    # # 第209
    ("l'd never leave you. l'm right here!", "I'd never leave you. I'm right here!"),

    # # 第210
    ("l'll give you a hint. l'm headed", "I'll give you a hint. I'm headed"),

    # # 第216
    ('l wonder if Ms Frizzle is down there', 'I wonder if Ms. Frizzle is down there'),

    # # 第220
    ('lt\'s a storm', 'It\'s a storm'),

    # # 第224
    ('if l could just get', 'if I could just get'),

    # # 第225
    ('it will prove l\'d been to', 'it will prove I\'d been to'),

    # # # 第227
    ('IS', 'is'),

    # # # 第229
    ("l've got to have", "I've got to have"),

    # # # # 第231
    ('Janet! Pull up! Pull up!', 'Janet! Pull up! Pull up!'),

    # # # # 第232
    ('lt\'s all yours', 'It\'s all yours'),

    # # # # # # 第234
    ('lf you visit', 'If you visit'),

    # # # # # # 第235
    ('l knew l should have stayed', 'I knew I should have stayed'),

    # # # # # # 第238
    ('l got them some of', 'I got them some of'),

    # # # # # # 第240
    ('planet l\'m on', 'planet I\'m on'),

    # # # # # # 第244
    ('lt\'s beautiful!', 'It\'s beautiful!'),

    # # # # # # 第246
    ('lt\'s got to be', 'It\'s got to be'),

    # # # # # # 第251
    ('lt could be', 'It could be'),

    # # # # # # 第253
    ('Hey, l\'m just', 'Hey, I\'m just'),

    # # # # # # 第254
    ('Let\'s-Find-Ms\\NFrizzle-Without-A-Map', 'Let\'s-Find-Ms. Frizzle-Without-A-Map'),

    # # # # # # 第259
    ('Liz and l can\'t wait', 'Liz and I can\'t wait'),

    # # # # # # 第264
    ('y eah, she can see', 'yeah, she can see'),

    # # # # # # 第269
    ('you can tell', 'You can tell'),

    # # # # # # 第273
    ('l have to stay', 'I have to stay'),

    # # # # # # 第277
    ('WAlT! l need proof!', 'WAIT! I need proof!'),

    # # # # # # 第279
    ("lt's my favourite planet!", "It's my favourite planet!"),

    # # # # # # 第283
    ('favourite color', 'favorite color'),

    # # # # # # 第291
    ('l\'ll tell you', 'I\'ll tell you'),

    # # # # # # 第300
    ('you sure can see', 'You sure can'),
    ('l wonder where', 'I wonder where'),

    # # # # # # 第307
    ('lt was a good hint, if l do say so myself', 'It was a good hint, if I do say so myself'),

    # # # # # # 第308
    ('l got enough stuff', 'I got enough stuff'),

    # # # # # # 第313
    ('There\'s no way l\'m going', 'There\'s no way I\'m going'),

    # # # # # # 第314
    ('There\'s no way l\'m going', 'There\'s no way I\'m going'),

    # # # # # # 第315
    ('lt\'s PROOF!', 'It\'s PROOF!'),

    # # # # # # 第316
    ('l\'ll believe you. They\'ll believe you.', 'I\'ll believe you. They\'ll believe you.'),

    # # # # # # 第317
    ('Janet! you want PROOF! l\'ll give you\\NPROOF!', 'Janet! You want PROOF! I\'ll give you\\NPROOF!'),

    # # # # # # 第318
    ('Here\'s proof', 'Here\'s proof'),

    # # # # # # 第323
    ('Arnold, it\'s the least l can do. lf it weren\'t for\\Nyou, l\'d still be on', 'Arnold, it\'s the least I can do. If it weren\'t for\\Nyou, I\'d still be on'),

    # # # # # # 第324
    ('l don\'t need to prove', 'I don\'t need to prove'),

    # # # # # # 第325
    ('you know it. l know it. lf no one else believes', 'You know it. I know it. If no one else believes'),

    # # # # # # 第338
    ('As l always say', 'As I always say'),

    # # # # # # 第339-343
    ('ls it the magic school bus?', 'Is it the magic school bus?'),

    # # # # # # # # 第348
    ('l\'ll give you that one, but only because you call\\Nit a MAGlC school bus.', 'I\'ll give you that one, but only because you call\\Nit a MAGIC school bus.'),

    # # # # # # # # # 第350
    ('lf we did it in real time', 'If we did it in real time'),

    # # # # # # # # # # 第359
    ('l happen to know', 'I happen to know'),

    # # # # # # # # # # 第360
    ('you\'re one bright kid!', 'You\'re one bright kid!'),

    # # # # # # # # # # 第362
    ('when the bus was coasting', 'when the bus was coasting'),
    ('they were weightless, like\\Nastronauts', 'they were weightless, like\\Nastronauts'),

    # # # # # # # # # # # 第364
    ('Uhhhh.. Well, will you?ahh?believe an artificial\\Ngravity geoscopic confabulator?', 'Uh... Well, would you, uh... believe an artificial\\Ngravity geoscopic confabulator?'),

    # # # # # # # # # # # # 第366
    ('OK. you got me', 'OK. You got me'),

    # # # # # # # # # # # # # 第367
    ('don\'t think l didn\'t notice', "don't think I didn't notice"),

    # # # # # # # # # # # # # # 第369
    ('y eah. Like', 'yeah. Like'),

    # # # # # # # # # # # # # # # 第371
    ('you wouldn\'t want', 'You wouldn\'t want'),

    # # # # # # # # # # # # # # # # # 第374
    ('you think she\'d', 'You think she\'d'),

    # # # # # # # # # # # # # # # # # # 第376
    ('why l called', 'why I called'),

    # # # # # # # # # # # # # # # # # # # 第377
    ('y es?', 'Yes?'),

    # # # # # # # # # # # # # # # # # # # # 第378
    ('l\'m STlLL waiting', 'I\'m STILL waiting'),

    # 其他大小写修正
    ('WAlT', 'WAIT'),
    ('MAGlC', 'MAGIC'),
    ('MS. FRlZZLE', 'MS. FRIZZLE'),
    ('SOMETHlNG', 'SOMETHING'),
    ('STlLL', 'STILL'),
    ('FlCTlON', 'FICTION'),
    ('STlLL', 'STILL'),
]

# 应用所有文本替换
for old, new in replacements:
    content = content.replace(old, new)

# 第三阶段：给指定词汇添加红色标记
for word in sorted(red_words, key=len, reverse=True):
    pattern = r'\b' + re.escape(word) + r'\b'
    content = re.sub(pattern, r'{\\1c&H0000FF\\b1}' + word + r'{\\r}', content)

# 保存文件
with open('/workspace/台词/srt/01 Gets Lost in Space.ass', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"完成！")
print(f"应用了 {len(replacements)} 处文本修改")
print(f"标记了 {len(red_words)} 个词汇为红色")
