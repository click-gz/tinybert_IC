#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能助手多轮对话意图识别数据集生成器 - 最终完整版
整合了基础模板、边界案例、真实日常表达

特点:
- 12个意图类别：
  0: 其他
  1: 聊天，包括问候、兴趣、哲学话题等，问题求助/技术支持
  2: 天气查询
  3: 新闻资讯
  4: 音乐播放
  5: 提醒设置
  6: 设备状态查询/设置
  7: 通话视频
  8: 跌倒监测确认
  9: 空间操作
  10: 视觉问答
  11: 餐厅推荐
- 包含基础模板、口语化表达、边界案例
- 真实场景、情绪驱动、上下文指代
- 自动划分train/dev/test
"""

import json
import random
import os
import argparse

random.seed(42)

# 意图类别定义
INTENT_LABELS = {
    0: "其他",
    1: "聊天，包括问候、兴趣、哲学话题等，问题求助/技术支持",
    2: "天气查询",
    3: "新闻资讯",
    4: "音乐播放",
    5: "提醒设置",
    6: "设备状态查询/设置",
    7: "通话视频",
    8: "跌倒监测确认",
    9: "空间操作",
    10: "视觉问答",
    11: "餐厅推荐"
}


FILLER_WORDS = [
    "嗯", "额", "那个", "我想", "其实", "说真的", "欸", "这个", "哎呀"
]

# 老年人特有的语气词和表达
ELDERLY_FILLERS = [
    "哎", "哎呀", "唉", "这个这个", "那个那个", "我说", "你说", "诶呀"
]

SENTENCE_ENDINGS = ["啊", "呀", "呢", "吧", "啦", "嘛", "哟"]

REQUEST_POLITESSES = [
    "麻烦", "请", "能不能", "帮忙", "劳驾", "请你", "麻烦你", "帮个忙"
]

FOLLOW_UP_ACKS = [
    "好的", "行", "谢谢你", "辛苦了", "拜托了", "收到", "好好好", "知道了"
]


USER_CLOSINGS = {
    0: ["算了", "不说了", "没事了", "随便吧", "我走了"],
    1: ["先聊到这", "我去忙啦", "改天再聊", "哈哈好吧", "那就这样"],
    2: ["知道了", "好嘞", "明白了", "行我准备下", "那就带伞"],
    3: ["辛苦了", "够了谢谢", "等下我再听", "知道了", "先这样"] ,
    4: ["不错就这首", "声音挺好", "先这样播着", "谢谢DJ", "我听会"] ,
    5: ["麻烦你", "记好了哈", "我等提醒", "行没问题", "收到提醒"] ,
    6: ["感谢", "那就这样", "我再观察", "行先别动", "先这样保持"] ,
    7: ["接通就行", "好了我自己来", "先挂了", "等会我再打", "行谢谢"],
    8: ["我没事放心", "我自己可以", "辛苦了", "别挂电话", "谢了"],
    9: ["到时候告诉我", "记得带回来", "麻烦快点", "好我等你", "谢谢"] ,
    10:["发我识别结果", "明白了", "好嘞", "辛苦了", "待会再看"] ,
    11:["就这家吧", "行我去试", "等会就去", "谢谢推荐", "好的下单"]
}


def maybe_add_noise(text, elderly_mode=False):
    """为用户话术增加口语化噪声，让表达更自然
    
    Args:
        text: 原始文本
        elderly_mode: 是否使用老年人语言特点（更多重复、停顿）
    """
    text = text.strip()
    if not text:
        return text
    
    # 选择语气词来源
    filler_pool = ELDERLY_FILLERS if elderly_mode and random.random() < 0.5 else FILLER_WORDS
    
    # 可能在开头或结尾加入语气词
    if random.random() < 0.35:  # 老年人更容易加语气词
        filler = random.choice(filler_pool)
        if random.random() < 0.5 and len(text) > 2:
            text = f"{filler}{text}"
        else:
            ending = random.choice(SENTENCE_ENDINGS)
            if not text.endswith(tuple(SENTENCE_ENDINGS)):
                text = f"{text}{ending}"
    
    # 可能追加礼貌或确认语
    if random.random() < 0.25:
        addon = random.choice([
            "，行不行", "，拜托了", "，谢谢", "，快点", "，好不好", "，要不然算了",
            "，可以吗", "，帮帮忙"
        ])
        if addon not in text:
            text = f"{text}{addon}"
    
    # 老年人模式：增加重复的概率
    repeat_prob = 0.15 if elderly_mode else 0.1
    if random.random() < repeat_prob and len(text) > 4:
        words = list(text)
        insert_pos = random.randint(1, len(words) - 1)
        words.insert(insert_pos, words[insert_pos - 1])
        text = "".join(words)
    
    return text


def ensure_user_last_turn(turns, label):
    if not turns:
        return turns
    if turns[-1]["speaker"] != "user":
        candidates = USER_CLOSINGS.get(label, USER_CLOSINGS.get(1, ["那就这样"]))
        turns.append({"speaker": "user", "text": random.choice(candidates)})
    return turns


def post_process_turns(turns, elderly_mode=False):
    """后处理对话轮次，添加口语化噪声
    
    Args:
        turns: 对话轮次列表
        elderly_mode: 是否使用老年人语言特点
    """
    processed = []
    for turn in turns:
        text = turn["text"]
        if turn["speaker"] == "user":
            text = maybe_add_noise(text, elderly_mode)
        processed.append({"speaker": turn["speaker"], "text": text})
    return processed


def create_dialogue(dialogue_id, turns, label, elderly_mode=False):
    """创建对话数据
    
    Args:
        dialogue_id: 对话ID
        turns: 对话轮次
        label: 意图标签
        elderly_mode: 是否使用老年人语言特点
    """
    finalized_turns = ensure_user_last_turn(turns, label)
    return {
        "dialogue_id": dialogue_id,
        "turns": post_process_turns(finalized_turns, elderly_mode),
        "label": label
    }


# ============================================================================
# Label 0: 其他（不属于其他类别的对话）
# ============================================================================

def generate_label_0_data(num_samples=500):
    """生成其他类别对话 - 不属于其他11个明确类别的内容"""
    random_topics = [
        # 无意义的测试
        "测试", "试试", "在吗", "喂喂喂", "你好你好",
        # 不完整的指令
        "帮我", "那个", "等等", "嗯嗯", "算了",
        # 模糊的询问
        "什么意思", "不懂", "啊", "哦", "是吗",
        # 错误输入或无效指令
        "asdf", "123", "。。。", "？？？", "emmm",
        # 超出能力范围的请求
        "帮我写作业", "替我上班", "帮我找对象", "给我钱",
        # 争议性或不适当的话题
        "你有多聪明", "你会不会取代人类", "你有感情吗",
    ]
    
    system_fallback = [
        "抱歉，我没太理解您的意思。", "您可以换个说法吗？",
        "我可能帮不上这个忙。", "这个问题超出了我的能力范围。",
        "我还在学习中，可能回答不好。", "您能再详细说说吗？"
    ]
    
    clarification_questions = [
        "您是想做什么呢？", "能具体说说吗？", "我不太明白您的意思。",
        "可以换个方式表达吗？", "这个我可能帮不上忙。"
    ]
    
    ambiguous_responses = [
        "好像不太对", "这个怎么说呢", "我也不确定", "可能吧", "不知道",
        "随便", "都行", "无所谓", "看着办"
    ]
    
    data = []
    for i in range(num_samples):
        scenario = random.random()
        turns = []
        
        if scenario < 0.2:
            # 单个无意义或测试输入
            turns.append({"speaker": "user", "text": random.choice(random_topics)})
            turns.append({"speaker": "system", "text": random.choice(system_fallback)})
            turns.append({"speaker": "user", "text": random.choice(ambiguous_responses)})
        elif scenario < 0.4:
            # 不完整或中断的对话
            turns.append({"speaker": "user", "text": random.choice([
                "帮我", "那个啥", "我想要", "能不能", "可以吗"
            ])})
            turns.append({"speaker": "system", "text": random.choice(clarification_questions)})
            turns.append({"speaker": "user", "text": random.choice([
                "算了", "没事了", "不用了", "我再想想", "改天说"
            ])})
        elif scenario < 0.6:
            # 超出能力范围的请求
            turns.append({"speaker": "user", "text": random.choice([
                "帮我写论文", "替我做决定", "你觉得我该选哪个",
                "帮我预测未来", "你能帮我赚钱吗", "能给我人生建议吗"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "这个可能超出我的能力了。", "这种事情我帮不上忙。",
                "建议您咨询专业人士。", "这个决定还是您自己做比较好。"
            ])})
            if random.random() < 0.5:
                turns.append({"speaker": "user", "text": random.choice([
                    "好吧", "那算了", "我自己想想", "知道了"
                ])})
        elif scenario < 0.8:
            # 关于AI本身的哲学问题
            turns.append({"speaker": "user", "text": random.choice([
                "你真的有意识吗", "你会不会取代人类", "你有感情吗",
                "你害怕死亡吗", "你的存在意义是什么", "你是真实的吗"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "我只是一个AI助手，帮您处理日常事务。",
                "这是个有趣的哲学问题，但我只是个程序。",
                "我不具备真正的意识，只是按照程序运行。",
                "我的目标就是帮助您更方便地生活。"
            ])})
            turns.append({"speaker": "user", "text": random.choice([
                "听起来挺复杂", "好吧明白了", "还是有点玄乎", "行吧"
            ])})
        else:
            # 混合类型或多意图不明确
            turns.append({"speaker": "user", "text": random.choice([
                "帮我查查天气然后播放音乐顺便订个餐厅",
                "我想要那个什么东西你知道的",
                "就是那个啥来着",
                "你懂我意思吗"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "您说的有点多，能一个一个来吗？",
                "我没太理解，能详细说说吗？",
                "您具体想做哪个？"
            ])})
            turns.append({"speaker": "user", "text": random.choice([
                "算了不弄了", "太麻烦了", "我自己来吧", "不管了"
            ])})
        
        data.append(create_dialogue(f"d_0_{i+1:04d}", turns, 0))
    return data


# ============================================================================
# Label 1: 聊天，包括问候、兴趣、哲学话题等，问题求助/技术支持
# ============================================================================

def generate_label_1_data(num_samples=500):
    """生成聊天和问题求助类对话"""
    # 聊天部分
    chitchat_topics = [
        "你好啊", "最近忙不忙", "你会思考吗", "你的梦想是什么",
        "今天过得好吗", "陪我聊聊天", "好无聊啊", "你开心吗",
        "今天天气真好", "这天气真舒服", "难得这么好的天气",
        "今天心情不错", "感觉有点累", "有点烦躁", "最近好像没动力",
        "今天超嗨", "有点小激动", "下班后干嘛好", "你会唱歌吗"
    ]
    
    # 老年人特有的聊天话题
    elderly_topics = [
        "陪我说说话", "一个人在家挺闷的", "今天孩子们都没来",
        "腿有点疼", "睡不着觉", "今天胃口不太好", "血压今天有点高",
        "想起以前的事了", "年轻的时候啊", "老伴去世好几年了",
        "孙子今天打电话了", "女儿说周末来看我", "儿子工作太忙了",
        "楼下老王又去跳广场舞了", "隔壁老李今天去体检", "想出去遛弯",
        "今天早上打太极了", "下午要去老年活动中心", "和老姐妹约好下棋",
        "这药吃了感觉不舒服", "最近总是忘事", "耳朵有点不好使",
        "眼睛看不清了", "手脚有点不灵活", "晚上睡觉容易醒"
    ]
    
    # 情感陪伴相关
    loneliness_expressions = [
        "一个人在家太安静了", "好想念孩子们", "要是老伴还在就好了",
        "总是想起以前", "现在的日子过得慢", "有点孤单",
        "邻居都不怎么来往", "老朋友都走得差不多了"
    ]
    
    # 健康关怀话题
    health_concerns = [
        "今天吃药了没有", "血压测了吗", "血糖怎么样", "记得按时吃药",
        "腰疼得厉害", "膝盖不舒服", "头有点晕", "胸口有点闷",
        "今天走了多少步", "要多喝水", "别忘了量血压"
    ]
    
    mood_followups = [
        "最近工作压得喘不过气", "周末打算去看电影", "感觉需要放个假",
        "我在想要不要去健身", "最近迷上做饭了", "想养只猫",
        "我在考虑换工作", "今天老板又开会了",
        # 老年人的后续
        "想去公园走走", "打算去买点菜", "下午要午休一会",
        "晚上想看看新闻", "准备给孙子打个电话"
    ]
    
    social_invites = [
        "我们有空一起去看风景好不好", "要不要一起出去玩", "周末一起去爬山吧",
        "有空一起吃个饭", "改天一起逛街", "找个时间一起聊聊",
        "要不要组团去旅游", "下次一起去看演唱会", "咱们约个时间见面",
        "一起去公园散步怎么样", "有空一起打球", "周末想去郊游有兴趣吗",
        "找时间聚聚", "好久不见了一起聚聚", "要不要一起去看展览"
    ]
    
    life_sharing = [
        "今天去了个很棒的地方", "最近在学画画", "我买了个新游戏",
        "发现一家好吃的店", "今天遇到件有趣的事", "最近在追一部剧",
        # 老年人的生活分享
        "今天在公园遇到老朋友了", "菜市场的菜又涨价了", "刚才和孙子视频了",
        "邻居家的猫又来我这了", "今天做了拿手菜"
    ]
    system_replies = [
        "您好！我在呢。", "我在这儿，陪你聊聊。", "听起来你需要聊聊。",
        "我在，随时陪你。", "您好呀，怎么啦？"
    ]
    empathy_replies = [
        "听起来有点累，要不要休息一下？", "辛苦啦，要不要喝点热饮放松下？",
        "嗯嗯，我懂的。", "我在，慢慢说。", "那你想做点什么放松呢？"
    ]
    
    # 问题求助/技术支持部分
    problems = [
        ("电脑", ["开不了机", "死机了", "蓝屏", "运行很慢", "黑屏", "自动重启", "风扇老响", "键盘失灵"]),
        ("手机", ["卡顿", "发热严重", "电池耗电快", "屏幕失灵", "无法充电", "黑屏", "摄像头模糊", "信号差"]),
        ("WiFi", ["连不上", "信号弱", "频繁断线", "速度很慢", "搜不到网络", "老断开", "密码老错"]),
        ("软件", ["打不开", "闪退", "安装失败", "卸载不掉", "更新出错", "运行报错", "卡在加载", "提示缺少组件"]),
        ("家电", ["冰箱不制冷", "洗衣机不转", "空调漏水", "电视黑屏", "扫地机器人卡住"]),
    ]
    troubleshooting_steps = [
        "请尝试断电一分钟后再开机", "建议您清理一下缓存再试试",
        "我可以把详细指引发到您的手机", "需要我安排售后上门吗",
        "我们可以远程协助，是否授权我连接设备", "先确保电源线和网线连接正常",
        "建议先备份重要数据，准备重置"
    ]
    emotional_reactions = [
        "我真的急用", "能不能快点", "别老让我重启", "你直接帮我搞定",
        "麻烦快点处理", "是不是要换新的了", "别再问了直接给方案"
    ]

    data = []
    for i in range(num_samples):
        scenario = random.random()
        turns = []

        # 增加老年人特有场景的比例
        if scenario < 0.1:
            # 情感陪伴场景（老年人孤独感）
            turns.append({"speaker": "user", "text": random.choice(loneliness_expressions)})
            turns.append({"speaker": "system", "text": random.choice([
                "我一直陪着您呢。", "您说说，我听着。", "别难过，我陪您聊聊。",
                "您想聊什么都可以。", "我在这儿陪您。"
            ])})
            turns.append({"speaker": "user", "text": random.choice([
                "谢谢你啊", "有你真好", "你真贴心", "陪我说说话就好"
            ])})
            if random.random() < 0.5:
                turns.append({"speaker": "system", "text": random.choice([
                    "要不要我给您放点音乐？", "咱们聊聊开心的事吧。",
                    "您可以跟我说说以前的事。"
                ])})
        elif scenario < 0.2:
            # 老年人健康关怀场景
            turns.append({"speaker": "user", "text": random.choice(elderly_topics)})
            turns.append({"speaker": "system", "text": random.choice([
                "要多注意休息啊。", "身体怎么样？", "要不要提醒您吃药？",
                "记得按时吃药。", "需要我帮您做点什么吗？"
            ])})
            if random.random() < 0.6:
                turns.append({"speaker": "user", "text": random.choice([
                    "没事就是说说", "老毛病了", "年纪大了不中用了",
                    "等会我去休息会", "一会儿吃药"
                ])})
        elif scenario < 0.28:
            # 老年人回忆往事场景
            turns.append({"speaker": "user", "text": random.choice([
                "想起以前的事了", "年轻的时候啊", "那时候日子虽然苦但是开心",
                "以前我们那个年代", "想起老伴了"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "跟我说说呗。", "那时候是什么样的？", "我听您讲。",
                "一定有很多故事吧。"
            ])})
            turns.append({"speaker": "user", "text": random.choice([
                "那时候大家都穷但是快乐", "现在变化太大了", "不说了都是往事了",
                "说起来话就长了", "老了老了"
            ])})
        elif scenario < 0.35:
            # 社交邀约场景
            turns.append({"speaker": "user", "text": random.choice(social_invites)})
            turns.append({"speaker": "system", "text": random.choice([
                "听起来不错呀！", "好主意！", "可以啊，什么时候？",
                "我很乐意陪你聊这个。", "那一定很有意思。"
            ])})
            if random.random() < 0.5:
                turns.append({"speaker": "user", "text": random.choice([
                    "那就这么定了", "到时候再约", "我也很期待", "有你在就好"
                ])})
        elif scenario < 0.42:
            # 生活分享场景
            turns.append({"speaker": "user", "text": random.choice(life_sharing)})
            turns.append({"speaker": "system", "text": random.choice([
                "哇，跟我说说！", "听起来很棒！", "是吗，怎么样？"
            ])})
            turns.append({"speaker": "user", "text": random.choice(mood_followups)})
        elif scenario < 0.5:
            # 简单问候
            turns.append({"speaker": "user", "text": random.choice(["喂", "嗨", "有人吗", "小助手在吗"])} )
            turns.append({"speaker": "system", "text": random.choice(system_replies)})
            turns.append({"speaker": "user", "text": random.choice(chitchat_topics)})
        elif scenario < 0.5:
            # 情绪表达
            turns.append({"speaker": "user", "text": random.choice(chitchat_topics)})
            turns.append({"speaker": "system", "text": random.choice(empathy_replies)})
            turns.append({"speaker": "user", "text": random.choice(mood_followups)})
            if random.random() < 0.5:
                turns.append({"speaker": "system", "text": random.choice([
                    "周末可以出去走走放松下。", "要不要听点音乐调节心情？",
                    "可以找朋友聊聊，也不错哦。"
                ])})
        elif scenario < 0.6:
            # 天气闲聊（不是真正的天气查询）
            turns.append({"speaker": "user", "text": random.choice([
                "今天天气真不错", "外面风太大了", "雾霾严重吗", "晴空万里心情好"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "是啊，最近天气确实很反复。", "哈哈，是不是想出去玩？",
                "看起来你心情不错呢。"
            ])})
            if random.random() < 0.5:
                turns.append({"speaker": "user", "text": random.choice([
                    "算了随便聊聊", "就是感慨一下", "我只是随便说说"
                ])})
        # 以下是问题求助/技术支持场景（40%）
        elif scenario < 0.75:
            # 简单技术问题
            category, issues = random.choice(problems)
            issue = random.choice(issues)
            brand = random.choice(["华硕", "戴尔", "联想", "苹果", "小米", "华为", "惠普", "荣耀", "三星"])
            turns.append({"speaker": "user", "text": random.choice([
                f"我的{brand}{category}{issue}", f"这个{category}最近老{issue}",
                f"帮我看看{brand}{category}怎么{issue}", f"{category}突然就{issue}了怎么办"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "收到，我看看怎么帮您处理", "我记下来了，需要了解更多情况",
                "别担心，我来一起排查"
            ])})
            turns.append({"speaker": "user", "text": random.choice([
                "就突然这样", "昨天还好好的", "开机几秒就卡住", "啥都没动就坏了"
            ])})
        else:
            # 复杂技术问题（多轮）
            category, issues = random.choice(problems)
            issue = random.choice(issues)
            brand = random.choice(["华硕", "戴尔", "联想", "苹果", "小米", "华为", "惠普", "荣耀", "三星"])
            turns.append({"speaker": "user", "text": f"求救，{category}{issue}"})
            turns.append({"speaker": "system", "text": random.choice([
                "先别急，我先确认一下情况", "我来和您一步步排查"
            ])})
            turns.append({"speaker": "user", "text": random.choice([
                "开机就黑屏", "提示英文我看不懂", "插电就一直闪", "连接不上网络"
            ])})
            turns.append({"speaker": "system", "text": random.choice(troubleshooting_steps)})
            if random.random() < 0.5:
                turns.append({"speaker": "user", "text": random.choice(emotional_reactions)})

        turns.append({"speaker": "user", "text": random.choice(USER_CLOSINGS[1])})
        # 30%的样本使用老年人语言特点
        use_elderly_mode = random.random() < 0.3
        data.append(create_dialogue(f"d_1_{i+1:04d}", turns, 1, elderly_mode=use_elderly_mode))
    return data


# ============================================================================
# Label 2: 天气查询（增强口语化和自然表达）
# ============================================================================

def generate_label_2_data(num_samples=500):
    """生成天气查询对话，包含口语化和间接表达"""
    cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "重庆", "武汉", "苏州", "厦门"]
    districts = ["朝阳", "浦东", "天河", "南山", "滨江", "高新区", "江汉"]
    times = ["今天", "明天", "后天", "周末", "晚上", "早上", "中午"]
    events = ["晨跑", "带娃出去", "去出差", "户外婚礼", "露营", "骑行", "登山"]
    temp_descriptions = ["太热", "有点冷", "闷得慌", "挺凉快", "干燥", "湿漉漉"]
    indirect_phrases = [
        "出门需要带伞吗", "要穿厚衣服吗", "适合户外运动吗",
        "今天穿什么衣服合适", "需要防晒吗", "会不会很冷",
        "要不要开空调", "空气质量咋样", "能晒衣服吗"
    ]
    informal_queries = [
        "今儿太阳毒不毒", "明天是不是还下雨", "帮我看看周末的天气",
        "后天温度能不能降点", "最近风力咋样", "湿度高不高",
        "晚上出去会不会冷", "明早会不会结冰"
    ]
    context_intros = [
        "等会要出门", "我打算带娃去公园", "下午有个户外活动",
        "晚上要加班回家晚", "朋友婚礼在室外", "明天打算去骑行",
        "准备周末露营", "明早要赶高铁"
    ]
    system_followups = [
        "是准备做什么活动吗？", "具体想了解哪个城市或区域呢？",
        "需要我给你生活建议吗？", "要不要顺便看看空气质量？"
    ]
    system_infos = [
        "我这边数据显示今天有阵雨", "温度大概在23度左右",
        "风力不大，大约二级南风", "湿度有点高，体感会闷热",
        "紫外线比较强，记得防晒", "晚间温差有点大，出门带件薄外套"
    ]
    follow_suggestions = [
        "建议备把小伞以防万一哦", "可以穿轻薄的长袖，防晒又不热",
        "还挺适合户外的，不过注意补水", "可能有雷阵雨，小心安排",
        "空气质量一般，戴口罩的话更安全", "晚上稍微有点凉，带件外套稳妥"
    ]

    data = []
    for i in range(num_samples):
        scenario = random.random()
        turns = []

        if scenario < 0.3:
            city = random.choice(cities)
            time = random.choice(times)
            expression = random.choice([
                f"{time}{city}天气怎么样", f"帮我看看{city}{time}要不要带伞",
                f"想了解下{city}{time}的温度", f"{time}{city}会不会下雨"
            ])
            turns.append({"speaker": "user", "text": expression})
        elif scenario < 0.55:
            turns.append({"speaker": "user", "text": random.choice(context_intros)})
            turns.append({"speaker": "system", "text": random.choice(system_followups)})
            city = random.choice(cities)
            time = random.choice(times)
            detail = random.choice([
                f"帮我查下{city}{time}天气", f"看看{time}{city}{random.choice(temp_descriptions)}吗",
                f"{districts[random.randrange(len(districts))]}那边{time}会不会下雨",
                random.choice(indirect_phrases)
            ])
            turns.append({"speaker": "user", "text": detail})
            if random.random() < 0.4:
                turns.append({"speaker": "system", "text": random.choice(system_infos)})
                turns.append({"speaker": "user", "text": random.choice([
                    "那需要带伞吗", "温差大不大", "穿短袖行不行", "小孩会不会冷"
                ])})
        elif scenario < 0.8:
            turns.append({"speaker": "user", "text": random.choice([
                "外面雨好大啊", "阳光晒得受不了", "风是不是挺大啊",
                "好像要打雷了", "今天天气怪怪的"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "要不要我给你查一下实时天气？", "出门要注意安全哦。",
                "我这边也看到有预警了。"
            ])})
            turns.append({"speaker": "user", "text": random.choice(informal_queries)})
            turns.append({"speaker": "system", "text": random.choice(system_infos)})
            if random.random() < 0.5:
                turns.append({"speaker": "system", "text": random.choice(follow_suggestions)})
        else:
            city = random.choice(cities)
            event = random.choice(events)
            turns.append({"speaker": "user", "text": random.choice([
                f"计划{event}，{city}天气咋样", f"等会要{event}，会不会下雨",
                f"我晚上要赶车，帮忙看看天气情况"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "我来帮你查下最新数据。", "稍等，我看看实时预报。",
                "我这边给你搜一下。"
            ])})
            turns.append({"speaker": "system", "text": random.choice(system_infos)})
            turns.append({"speaker": "user", "text": random.choice([
                "那需要准备什么", "有雷暴预警吗", "温差大不大", "要不要推迟"
            ])})
            turns.append({"speaker": "system", "text": random.choice(follow_suggestions)})

        data.append(create_dialogue(f"d_2_{i+1:04d}", turns, 2))
    return data


# ============================================================================
# Label 3: 新闻资讯
# ============================================================================

def generate_label_3_data(num_samples=500):
    """生成新闻资讯查询对话"""
    categories = ["科技", "体育", "娱乐", "财经", "国际", "本地", "生活", "社会"]
    push_styles = ["播报", "推送", "念一下", "讲讲", "总结下", "快速说"]
    follow_needs = [
        "顺便说下股市", "有没有突发", "要紧的有没有", "重点说科技",
        "先说国内的", "帮我过滤一下广告", "简单概括就行"
    ]
    boundary_requests = [
        "今天大新闻", "有没有什么值得关注的", "先说比赛结果",
        "有没有明星八卦", "财报怎么样", "有没有疫情消息"
    ]
    device_contexts = [
        "开车", "做饭", "通勤", "健身", "加班", "刚起床"
    ]
    system_prompts = [
        "想听哪方面的？", "需要我按类别分吗？", "要不要我按时间顺序讲？",
        "是想听国内还是国际？", "要不要我给你讲重点摘要？"
    ]

    data = []
    for i in range(num_samples):
        scenario = random.random()
        turns = []
        category = random.choice(categories)

        if scenario < 0.25:
            turns.append({"speaker": "user", "text": random.choice([
                f"播报一下{category}新闻", f"想听{category}方面的最新消息",
                f"{category}最近啥情况", f"帮我总结下{category}热点"
            ])})
        elif scenario < 0.5:
            turns.append({"speaker": "user", "text": random.choice([
                "新闻来一段", "给我推点新闻", "讲讲今天的大事"
            ])})
            turns.append({"speaker": "system", "text": random.choice(system_prompts)})
            turns.append({"speaker": "user", "text": random.choice([
                f"重点说{category}", f"我想听{category}和{random.choice(categories)}",
                random.choice(boundary_requests)
            ])})
            if random.random() < 0.4:
                turns.append({"speaker": "system", "text": random.choice([
                    "要不要再顺带提一下财经？", "我可提供图文摘要哦。",
                    "要关注突发快讯吗？"
                ])})
        elif scenario < 0.75:
            turns.append({"speaker": "user", "text": random.choice([
                f"我在{random.choice(device_contexts)}，播点新闻", "开车路上想听新闻",
                "忙着做饭，你帮我讲讲新闻"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "明白，要不要我说重点的？", "我来快速帮您摘要。",
                "您想先听哪一类？"
            ])})
            turns.append({"speaker": "user", "text": random.choice([
                f"先说{category}", "财经和国际重点讲", "别太长，重点说",
                random.choice(follow_needs)
            ])})
            if random.random() < 0.5:
                turns.append({"speaker": "system", "text": random.choice([
                    "需要我把链接发给你吗？", "我可以帮你标记稍后阅读。",
                    "要不要我做个语音播报？"
                ])})
        else:
            turns.append({"speaker": "user", "text": random.choice([
                "最近有什么值得关注的新闻", "今天热点是什么",
                "有没有什么突发消息", "重要新闻帮我留意一下"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "我帮你查看一下各种频道。", "稍等，我看看最新资讯。",
                "我先整理一下要点给你。"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                f"目前{category}方面有新进展。", f"体育这边有最新赛况。",
                "本地也有一些活动资讯。", "国际上有新的政策发布。"
            ])})
            turns.append({"speaker": "user", "text": random.choice([
                "重点说国内", "挑重要的说", "发我链接", "帮我收藏",
                "晚点提醒我看详细的"
            ])})

        data.append(create_dialogue(f"d_3_{i+1:04d}", turns, 3))
    return data


# ============================================================================
# Label 4: 音乐播放（真实日常表达）
# ============================================================================

def generate_label_4_data(num_samples=500):
    """生成音乐播放对话，包含口语化和简短表达"""
    singers = ["周杰伦", "邓紫棋", "薛之谦", "林俊杰", "陈奕迅"]
    
    # 老年人喜欢的歌手和音乐
    elderly_singers = ["邓丽君", "刘德华", "费玉清", "蔡琴", "李谷一", "阎维文", "宋祖英"]
    elderly_music_types = [
        "民歌", "红歌", "京剧", "豫剧", "黄梅戏", "越剧", "评书", "相声",
        "老歌", "怀旧歌曲", "革命歌曲", "地方戏", "样板戏"
    ]
    classic_songs = [
        "月亮代表我的心", "我的祖国", "在那遥远的地方", "茉莉花", "红梅赞",
        "唱支山歌给党听", "北京的金山上", "敖包相会"
    ]
    
    genres = ["舒缓", "摇滚", "爵士", "古典", "电子", "民谣", "轻音乐", "纯音乐"]
    moods = ["心情不好", "想嗨", "有点丧", "想睡觉", "运动用", "想跳舞", "放松", "怀旧"]
    situations = ["开车", "做饭", "跑步", "学习", "和朋友聚会", "晚上加班", "午休", "散步", "打太极"]
    music_requests = [
        "放首歌", "来首歌", "放点音乐", "听歌", "放歌", "来点节奏感的",
        "心情不好，放点舒缓的", "开心，来点欢快的", "累了，听点放松的",
        "换一首", "下一首", "暂停", "继续", "声音大点", "小声点",
        "再放一遍", "这首不好听", "换个风格", "放点国外的", "有没有轻音乐"
    ]
    feature_requests = [
        "调节音量", "换个音箱播放", "设置睡眠定时", "同步到电视",
        "把喜欢的收藏", "随机播放", "循环播放", "播放列表顺序播"
    ]
    system_confirmations = [
        "好的，马上播放。", "收到，为您安排。", "了解，正在切换曲目。",
        "我来帮您调一下。", "马上切换音量。"
    ]

    data = []
    for i in range(num_samples):
        num_user_turns = random.randint(1, 5)
        turns = []
        
        # 30%的样本使用老年人音乐偏好
        is_elderly_preference = random.random() < 0.3
        
        if num_user_turns == 1:
            if random.random() < 0.4:
                request = random.choice(music_requests)
            elif is_elderly_preference:
                # 老年人音乐请求
                request = random.choice([
                    f"播放{random.choice(elderly_singers)}的歌",
                    f"来点{random.choice(elderly_music_types)}",
                    f"放{random.choice(classic_songs)}",
                    "放点老歌", "来点怀旧的", "放点民歌", "来段评书",
                    "放点京剧", "来点红歌"
                ])
            else:
                singer = random.choice(singers)
                request = f"播放{singer}的歌"
            turns.append({"speaker": "user", "text": request})
            
        elif num_user_turns == 2:
            turns.append({"speaker": "user", "text": random.choice(["放歌", "听歌", "放点音乐"])})
            turns.append({"speaker": "system", "text": "想听什么类型的？"})
            if is_elderly_preference:
                turns.append({"speaker": "user", "text": random.choice([
                    "随便", "轻松的", f"{random.choice(elderly_singers)}的",
                    "老歌就行", "来点怀旧的", "民歌吧", "评书也行"
                ])})
            else:
                turns.append({"speaker": "user", "text": random.choice([
                    "随便", "轻松的", f"{random.choice(singers)}的"
                ])})
        elif num_user_turns == 3:
            mood = random.choice(moods)
            follow = random.choice(["放首歌", "来点音乐", "推荐一下", "给我点歌"])
            turns.append({"speaker": "user", "text": random.choice([
                f"{mood}，{follow}",
                f"我在{random.choice(situations)}，适合听啥",
                f"想听{random.choice(genres)}风格的"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "我帮你挑几首。", "想要中文还是英文？", "我推荐一个列表给你。"
            ])})
            turns.append({"speaker": "user", "text": random.choice([
                f"先放{random.choice(singers)}的", "随便来点节奏快的",
                "轻音乐就行", "帮我放点怀旧的"
            ])})
        else:
            turns.append({"speaker": "user", "text": "放歌"})
            turns.append({"speaker": "system", "text": random.choice(system_confirmations)})
            turns.append({"speaker": "user", "text": random.choice([
                f"播放{random.choice(singers)}的歌", f"来点{random.choice(genres)}",
                f"放点适合{random.choice(situations)}的"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "正在为您挑选...", "马上来", "我给您排个歌单。"
            ])})
            turn_extra = random.random()
            if turn_extra < 0.4:
                turns.append({"speaker": "user", "text": random.choice([
                    "声音大点", "再小点", "有没纯音乐", "下一首",
                    "循环这首", "切到现场版"
                ])})
                turns.append({"speaker": "system", "text": random.choice(system_confirmations)})
            elif turn_extra < 0.75:
                turns.append({"speaker": "user", "text": random.choice(feature_requests)})
                turns.append({"speaker": "system", "text": random.choice([
                    "设置好了。", "已经同步。", "好的，给您收藏了。"
                ])})
            else:
                turns.append({"speaker": "user", "text": random.choice([
                    "这首有点吵", "换成轻松点的", "有没有纯钢琴", "随便推荐个热歌"
                ])})
                turns.append({"speaker": "system", "text": random.choice([
                    "我给你换一首。", "我推荐你听这首热门的。",
                    "要不要试试我新整理的歌单？"
                ])})
        
        data.append(create_dialogue(f"d_4_{i+1:04d}", turns, 4))
    return data


# ============================================================================
# Label 5: 提醒设置（真实场景化）
# ============================================================================

def generate_label_5_data(num_samples=500):
    """生成提醒设置对话"""
    times = [
        "早上7点", "上午9点", "中午12点", "下午3点", "晚上8点",
        "明天一早", "后天下午", "周一", "周末", "十分钟后", "半小时后",
        "早饭前", "早饭后", "午饭前", "午饭后", "晚饭前", "晚饭后", "睡觉前"
    ]
    events = [
        "开会", "吃药", "运动", "起床", "打电话给妈妈", "参加家长会",
        "缴水电费", "去机场", "给老板发报告", "提醒喝水", "接孩子"
    ]
    
    # 老年人特有的提醒事项
    elderly_events = [
        "吃降压药", "吃降糖药", "吃心脏病药", "测血压", "测血糖",
        "去医院复查", "去社区医院拿药", "打电话给儿子", "打电话给女儿",
        "孙子放学", "老年活动中心有活动", "和老姐妹约好下棋",
        "楼下跳广场舞", "去公园打太极", "午睡", "晒太阳",
        "喝水", "眼药水", "泡脚", "按摩", "看新闻联播"
    ]
    simple_requests = [
        "设置提醒", "提醒我一下", "别忘了提醒我", "到时候叫我", "设置一个闹钟",
        "帮我记一下", "提醒我不要忘了", "安排个提醒"
    ]
    clarification_prompts = [
        "什么时候提醒您？", "要提醒什么事情呢？", "需要重复提醒吗？",
        "要不要我提前几分钟提醒？", "是一次性还是每天都提醒？"
    ]
    acknowledgement = [
        "好的，帮您记下了。", "提醒设置好了。", "我已经帮您加到日程里。",
        "收到，届时会提醒您。", "设置完成，有需要随时告诉我。"
    ]
    user_followups = [
        "提前十分钟提醒", "每天都提醒", "只提醒一次", "改成晚上八点",
        "稍微提前一点", "帮我同步到手机", "提醒的时候播个音乐"
    ]

    data = []
    for i in range(num_samples):
        scenario = random.random()
        turns = []

        time = random.choice(times)
        # 40%的样本使用老年人特有的提醒事项
        event = random.choice(elderly_events) if random.random() < 0.4 else random.choice(events)

        if scenario < 0.25:
            request = random.choice(simple_requests)
            turns.append({"speaker": "user", "text": request})
            turns.append({"speaker": "system", "text": random.choice(clarification_prompts)})
            turns.append({"speaker": "user", "text": f"{time}提醒我{event}"})
            turns.append({"speaker": "system", "text": random.choice(acknowledgement)})
        elif scenario < 0.5:
            turns.append({"speaker": "user", "text": f"{time}提醒我{event}"})
            turns.append({"speaker": "system", "text": random.choice(acknowledgement)})
            if random.random() < 0.6:
                turns.append({"speaker": "user", "text": random.choice(user_followups)})
                turns.append({"speaker": "system", "text": random.choice([
                    "好的，已经调整。", "帮您更新好了。", "同步完成。"
                ])})
        elif scenario < 0.75:
            turns.append({"speaker": "user", "text": random.choice([
                "我要设置一个提醒", "帮我记一下明天的事", "下午三点很重要，提醒我"
            ])})
            turns.append({"speaker": "system", "text": random.choice(clarification_prompts)})
            turns.append({"speaker": "user", "text": f"{time}提醒我{event}"})
            turns.append({"speaker": "system", "text": random.choice([
                random.choice(acknowledgement),
                "我可以提前十五分钟再提醒一次，要不要？"
            ])})
            if random.random() < 0.5:
                turns.append({"speaker": "user", "text": random.choice([
                    "提前十五分钟", "不用重复", "再提醒我一次",
                    "到点的时候用铃声提醒"
                ])})
                turns.append({"speaker": "system", "text": random.choice([
                    "拿下，已经设置。", "按照你的要求调整好了。",
                    "好，提醒方式改成铃声。"
                ])})
        else:
            turns.append({"speaker": "user", "text": random.choice([
                f"等会提醒我{event}",
                f"我怕忘记{event}",
                "最近老忘事，帮我记一下"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "具体什么时间？", "要提醒几次？", "需要设置为重复提醒吗？"
            ])})
            turns.append({"speaker": "user", "text": f"{time}提醒我"})
            turns.append({"speaker": "system", "text": random.choice([
                random.choice(acknowledgement),
                "要不要同步到日历里？"
            ])})
            if random.random() < 0.4:
                turns.append({"speaker": "user", "text": random.choice([
                    "同步一下", "不用了", "顺便提醒喝水" , "再加一个晚上十点的"
                ])})
                turns.append({"speaker": "system", "text": random.choice([
                    "好的，再补一个提醒。", "明白，保持一次。",
                    "同步完成。"
                ])})

        data.append(create_dialogue(f"d_5_{i+1:04d}", turns, 5))
    return data


# ============================================================================
# Label 6: 设备状态查询（口语化）
# ============================================================================

def generate_label_6_data(num_samples=500):
    """生成设备状态查询对话 - 聚焦机器人本体的电量和音量"""
    
    # 电量相关查询
    battery_queries = [
        "还有电吗", "快没电了吧", "电量还够吗", "还能用多久",
        "电量剩多少", "要充电了吗", "电池怎么样", "电够不够",
        "能用到晚上吗", "电量百分之多少", "还有多少电",
        "是不是该充电了", "电快用完了吧"
    ]
    
    # 音量相关查询和设置
    volume_queries = [
        "太吵了", "太大声了", "小声点", "声音调大", "静音",
        "音量多少", "声音太小", "大点声", "调小声",
        "音量调到最大", "能不能小声点", "声音刚好",
        "把声音关了", "声音开大点"
    ]
    
    # 混合查询（既可能问电量也可能问音量）
    general_queries = [
        "你现在状态怎么样", "检查一下", "还好吗",
        "工作正常吗", "有没有问题"
    ]
    
    # 电量相关的操作
    battery_actions = [
        "开启省电模式", "去充电", "回去充电", "记得充电",
        "电量低了提醒我", "快没电了去充电吧"
    ]
    
    # 音量相关的操作
    volume_actions = [
        "把音量调到50%", "音量调大一点", "音量调小一点",
        "静音", "取消静音", "音量调到最大", "音量调到最小",
        "声音合适就行"
    ]
    
    # 系统回复
    battery_responses = [
        "当前电量65%，还能用3小时左右。", "电量只剩15%，建议充电。",
        "电量充足，还有80%。", "电量40%，可以继续使用。",
        "电量快用完了，剩余10%。", "电量92%，不用担心。"
    ]
    
    volume_responses = [
        "音量已调为30%。", "好的，音量调大了。", "已静音。",
        "音量已设置为50%。", "音量调小了。", "当前音量是60%。"
    ]
    
    clarifications = [
        "您想了解电量还是音量？", "需要我调整什么？",
        "要查看电量还是调节音量？"
    ]
    
    data = []
    for i in range(num_samples):
        scenario = random.random()
        turns = []
        
        # 决定是电量查询还是音量查询
        query_type = random.choice(["battery", "volume", "general"])

        if scenario < 0.3:
            # 简单的单轮查询
            if query_type == "battery":
                turns.append({"speaker": "user", "text": random.choice(battery_queries)})
                turns.append({"speaker": "system", "text": random.choice(battery_responses)})
            elif query_type == "volume":
                turns.append({"speaker": "user", "text": random.choice(volume_queries)})
                turns.append({"speaker": "system", "text": random.choice(volume_responses)})
            else:
                turns.append({"speaker": "user", "text": random.choice(general_queries)})
                turns.append({"speaker": "system", "text": random.choice([
                    "电量70%，音量适中，一切正常。",
                    "运行正常，电量充足。",
                    "状态良好，随时为您服务。"
                ])})
                
        elif scenario < 0.55:
            # 多轮对话，先询问后明确
            turns.append({"speaker": "user", "text": random.choice(general_queries)})
            turns.append({"speaker": "system", "text": random.choice(clarifications)})
            
            if query_type == "battery" or random.random() < 0.5:
                turns.append({"speaker": "user", "text": random.choice(battery_queries)})
                turns.append({"speaker": "system", "text": random.choice(battery_responses)})
            else:
                turns.append({"speaker": "user", "text": random.choice(volume_queries)})
                turns.append({"speaker": "system", "text": random.choice(volume_responses)})
                
        elif scenario < 0.75:
            # 查询后进行操作
            if query_type == "battery" or random.random() < 0.5:
                turns.append({"speaker": "user", "text": random.choice(battery_queries)})
                turns.append({"speaker": "system", "text": random.choice(battery_responses)})
                if random.random() < 0.6:
                    turns.append({"speaker": "user", "text": random.choice(battery_actions)})
                    turns.append({"speaker": "system", "text": random.choice([
                        "好的，已开启省电模式。", "我这就去充电。",
                        "收到，电量低时会提醒您。"
                    ])})
            else:
                turns.append({"speaker": "user", "text": random.choice(volume_queries)})
                turns.append({"speaker": "system", "text": random.choice(volume_responses)})
                if random.random() < 0.5:
                    turns.append({"speaker": "user", "text": random.choice(volume_actions)})
                    turns.append({"speaker": "system", "text": random.choice([
                        "音量已调整。", "好的，已经调好了。", "音量设置完成。"
                    ])})
                    
        else:
            # 直接请求操作
            if query_type == "battery" or random.random() < 0.5:
                turns.append({"speaker": "user", "text": random.choice(battery_actions)})
                turns.append({"speaker": "system", "text": random.choice([
                    "好的，已开启省电模式。", "我马上去充电。",
                    "收到，我会提醒您的。"
                ])})
            else:
                turns.append({"speaker": "user", "text": random.choice(volume_actions)})
                turns.append({"speaker": "system", "text": random.choice(volume_responses)})

        data.append(create_dialogue(f"d_6_{i+1:04d}", turns, 6))
    return data


# ============================================================================
# Label 7: 通话视频（真实联系人）
# ============================================================================

def generate_label_7_data(num_samples=500):
    """生成通话视频对话，使用真实联系人名"""
    # 真实的联系人
    family = ["妈妈", "爸爸", "老婆", "老公", "儿子", "女儿"]
    friends = ["小王", "小李", "晓晓", "阿明"]
    work = ["老板", "同事", "客户"]
    all_contacts = family + friends + work
    
    call_requests = [
        "打电话", "打给{contact}", "给{contact}打", "呼叫{contact}", "打",
        "联系{contact}", "视频给{contact}", "拨{contact}", "打个电话给{contact}"
    ]
    call_contexts = [
        "开免提", "来个视频通话", "用耳机接", "别开免提",
        "走家庭号", "用公司的座机", "用微信打"
    ]
    system_prompts = [
        "请问要打给谁？", "是语音还是视频？", "要使用哪个设备拨号？",
        "需要同步通讯录吗？", "要我帮你记录通话内容吗？"
    ]
    system_confirms = [
        "正在为您呼叫", "马上为您接通", "好的，正在拨号", "我帮你接通",
        "正在连接，请稍等"
    ]
    follow_commands = [
        "免提", "挂断", "转接到手机", "改成视频", "等一下再拨", "稍后提醒我回拨",
        "加入会议", "同步给家人"
    ]
    
    data = []
    for i in range(num_samples):
        num_user_turns = random.randint(1, 4)
        contact = random.choice(all_contacts)
        turns = []
        
        if num_user_turns == 1:
            request = random.choice(call_requests).replace("{contact}", contact)
            turns.append({"speaker": "user", "text": request})
        elif num_user_turns == 2:
            turns.append({"speaker": "user", "text": f"打给{contact}"})
            turns.append({"speaker": "system", "text": f"正在拨打{contact}..."})
            turns.append({"speaker": "user", "text": random.choice(["免提", "挂断"])})
        else:
            turns.append({"speaker": "user", "text": random.choice([
                "打电话", "帮我联系一下", "拨个电话"
            ])})
            turns.append({"speaker": "system", "text": random.choice(system_prompts)})
            turns.append({"speaker": "user", "text": random.choice([
                contact,
                f"打给{contact}",
                f"用视频打{contact}",
                f"打给{contact}开免提"
            ])})
            turns.append({"speaker": "system", "text": random.choice(system_confirms)})
            if random.random() < 0.5:
                turns.append({"speaker": "user", "text": random.choice(call_contexts)})
                turns.append({"speaker": "system", "text": random.choice([
                    "好的，切换完成。", "已经设置为免提。", "我改成视频了。"
                ])})
            if random.random() < 0.4:
                turns.append({"speaker": "user", "text": random.choice(follow_commands)})
                turns.append({"speaker": "system", "text": random.choice([
                    "已经操作。", "好的，稍后提醒您。", "通话已挂断。"
                ])})
        data.append(create_dialogue(f"d_7_{i+1:04d}", turns, 7))
    return data


# ============================================================================
# Label 8: 跌倒监测确认（真实反应）
# ============================================================================

def generate_label_8_data(num_samples=500):
    """生成跌倒检测确认对话，真实的用户反应"""
    fall_responses = {
        "no_fall": ["没事", "我没事", "没有", "我很好", "别担心", "误报了", "我站得稳"] ,
        "fell_but_ok": ["摔了一下，没事", "跌倒了但我没事", "已经起来了", "擦破皮而已"] ,
        "need_help": ["跌倒了", "我摔了", "疼", "起不来", "帮我", "需要帮助", "腿抬不起来"],
        "confused": ["什么", "怎么了", "啊", "嗯", "你说啥", "发生什么事"]
    }
    system_prompts = [
        "检测到您可能跌倒了，还好吗？", "刚刚我看起来像是检测到了跌倒，情况怎么样？",
        "您的姿态有异常，需要派人过去吗？", "请回应一下，确认是否需要帮助。",
        "我收到跌倒警报，请告诉我您是否安全。"
    ]
    follow_checks = [
        "需不需要我打电话给紧急联系人？", "要不要我通知家里人？",
        "是否需要呼叫救护车？", "我可以帮您记录这次情况。",
        "要不要我继续跟您保持通话？"
    ]
    reassurance = [
        "好的，我会继续观察。", "明白了，我会注意。", "那我再过几分钟确认一次。",
        "如果不舒服记得说。", "我帮您记录下来了。"
    ]
    help_actions = [
        "我马上呼叫救护车。", "我给家属打电话。", "我持续跟您保持通话。",
        "我帮您开启室内广播。"
    ]
    
    data = []
    for i in range(num_samples):
        num_user_turns = random.randint(1, 4)
        response_type = random.choice(list(fall_responses.keys()))
        turns = []
        
        if num_user_turns == 1:
            response = random.choice(fall_responses[response_type])
            turns.append({"speaker": "user", "text": response})
        elif num_user_turns == 2:
            turns.append({"speaker": "system", "text": random.choice(system_prompts)})
            turns.append({"speaker": "user", "text": random.choice(fall_responses[response_type])})
            if random.random() < 0.5:
                turns.append({"speaker": "system", "text": random.choice(reassurance)})
        elif num_user_turns == 3:
            turns.append({"speaker": "system", "text": "检测到可能跌倒"})
            turns.append({"speaker": "user", "text": random.choice(fall_responses["confused"])})
            turns.append({"speaker": "system", "text": random.choice(system_prompts)})
            turns.append({"speaker": "user", "text": random.choice(fall_responses[response_type])})
            if random.random() < 0.4:
                turns.append({"speaker": "system", "text": random.choice(follow_checks)})
                turns.append({"speaker": "user", "text": random.choice([
                    "不用不用", "帮我通知家里", "叫救护车", "我试试自己站起来"
                ])})
        else:
            turns.append({"speaker": "system", "text": random.choice(system_prompts)})
            turns.append({"speaker": "user", "text": random.choice([
                random.choice(fall_responses["confused"]),
                "我有点晕", "等一下"
            ])})
            turns.append({"speaker": "system", "text": random.choice(system_prompts)})
            turns.append({"speaker": "user", "text": random.choice(fall_responses[response_type])})
            turns.append({"speaker": "system", "text": random.choice([
                random.choice(follow_checks),
                random.choice(reassurance)
            ])})
            if response_type == "need_help" and random.random() < 0.6:
                turns.append({"speaker": "user", "text": random.choice([
                    "帮我叫救护车", "帮我叫家里人", "别挂电话",
                    "我打不开门", "我站不起来"
                ])})
                turns.append({"speaker": "system", "text": random.choice(help_actions)})

        data.append(create_dialogue(f"d_8_{i+1:04d}", turns, 8))
    return data


# ============================================================================
# Label 9: 空间操作
# ============================================================================

def generate_label_9_data(num_samples=500):
    """生成空间操作对话 - 强调必须通过移动才能完成的任务"""
    locations = ["厨房", "卧室", "客厅", "浴室", "阳台", "书房", "储物间", "玄关"]
    
    # 老年人常寻找或需要拿的物品
    objects = [
        "遥控器", "杯子", "手机", "书", "钥匙", "眼镜", "药盒", "垃圾袋", "充电器",
        "老花镜", "拐杖", "血压计", "药瓶", "水杯", "毛巾", "报纸", "收音机"
    ]
    
    # 强调"移动"才能完成的动作
    actions = [
        "去拿", "去看看", "去检查", "过去看", "巡视", "检查",
        "开灯", "关窗", "打开空调", "关空调", "带回来", "看看那里情况",
        "看看火关了没", "看看窗户关了没", "看看门锁了没"
    ]
    constraints = [
        "小心别碰到花瓶", "注意地上有水", "顺便把门带上",
        "如果看见猫就喂点粮", "有人在睡觉别吵", "动作轻一点"
    ]
    system_prompts = [
        "我现在在客厅，需要去哪个位置？", "要不要顺便带点什么回来？",
        "有具体目标物品吗？", "需要我记录过程吗？", "要按照什么路线去？"
    ]
    confirmations = [
        "好的，正在前往{location}。", "马上去{location}看看。",
        "收到，去{location}{action}。", "我从{location}开始执行。"
    ]

    data = []
    for i in range(num_samples):
        num_user_turns = random.randint(1, 4)
        location = random.choice(locations)
        obj = random.choice(objects)
        action = random.choice(actions)
        turns = []

        if num_user_turns == 1:
            request = random.choice([
                # 强调"去"或"移动到"的表达
                f"去{location}帮我拿{obj}", f"到{location}拿一下{obj}",
                f"去{location}{action}", f"去{location}看看{obj}在不在",
                f"帮我去{location}拿{obj}", f"去{location}帮我取{obj}",
                # 隐含移动的表达
                f"{location}的{obj}帮我取一下", f"帮我拿{location}的{obj}",
                f"{location}那边{action}", f"到{location}{action}",
                # 强调检查某地状态（必须移动）
                f"去{location}看看火关了没", f"去{location}检查一下窗户",
                f"帮我看看{location}的灯关了吗", f"去{location}看看门锁了没"
            ])
            turns.append({"speaker": "user", "text": request})
        elif num_user_turns == 2:
            turns.append({"speaker": "user", "text": random.choice([
                f"去{location}帮我拿{obj}", f"帮我拿{location}的{obj}",
                f"到{location}取{obj}", f"{location}{action}"
            ])})
            turns.append({"speaker": "system", "text": confirmations[0].format(location=location, action=action)})
            turns.append({"speaker": "user", "text": random.choice([
                "顺便看看窗户开了吗", "动作轻一点", "注意下楼梯", "路上别撞到桌子"
            ])})
        elif num_user_turns == 3:
            turns.append({"speaker": "user", "text": random.choice([
                "帮我拿东西", "帮我看下", "过去处理一下", "机器人帮忙",
                "需要你帮个忙", "能帮我做点事吗"
            ])})
            turns.append({"speaker": "system", "text": random.choice(system_prompts)})
            turns.append({"speaker": "user", "text": random.choice([
                f"{location}{action}{obj if random.random() < 0.6 else ''}",
                f"帮我{action}{location}",
                f"{location}的{obj}拿一下",
                f"去{location}{action}"
            ]).strip()})
            turns.append({"speaker": "system", "text": confirmations[1].format(location=location, action=action)})
        else:
            turns.append({"speaker": "user", "text": random.choice([
                f"去{location}帮我拿{obj}",
                f"帮我拿{location}的{obj}",
                f"到{location}{action}",
                f"{location}那边检查下有没有{obj}",
                f"帮我{action}{location}",
                "帮我巡一圈",
                f"机器人{action}{location}"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "要我带回来吗？", "有具体注意事项吗？",
                "我从哪个入口去？"
            ])})
            turns.append({"speaker": "user", "text": random.choice([
                f"从客厅过去", f"顺便把{obj}带回来", f"到那边时{random.choice(constraints)}",
                f"到了给我发个通知"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                confirmations[2].format(location=location, action=action),
                confirmations[3].format(location=location, action=action)
            ])})
            if random.random() < 0.4:
                turns.append({"speaker": "user", "text": random.choice([
                    "快点", "慢点别碰倒东西", "到了叫我", "顺便看看垃圾需要倒吗"
                ])})
                turns.append({"speaker": "system", "text": random.choice([
                    "收到，马上执行。", "好的，我会小心。", "执行中，稍后汇报。"
                ])})

        data.append(create_dialogue(f"d_9_{i+1:04d}", turns, 9))
    return data


# ============================================================================
# Label 10: 视觉识别（明确区分用法）
# ============================================================================

def generate_label_10_data(num_samples=500):
    """生成视觉识别对话，明确区分"看"的图像识别用法"""
    vision_requests = [
        # 明确的识别请求
        "帮我识别这个物体", "这是什么", "识别一下", "看看这是啥",
        "照片里是什么", "帮我看看这个图片", "这个图案是什么",
        
        # OCR相关
        "帮我读一下这段文字", "这上面写的什么", "识别这个文字",
        
        # 明确"看"指向图像
        "看看我拍的这张照片", "帮我分析这张图", "这个画面里有什么",
        
        # 历史记忆
        "你还记得昨天给你看的药吗", "之前那个瓶子是什么",
    ]
    
    # 老年人特有的视觉识别场景
    elderly_vision_requests = [
        # 药品相关（老年人常见）
        "帮我看看这个药盒上写的什么", "这个药说明书字太小看不清",
        "帮我读一下药品用法", "这个药是饭前吃还是饭后吃",
        "这个药瓶上的保质期是多少", "帮我看看这个药过期了没",
        
        # 食品保质期
        "这个保质期到什么时候", "帮我看看这个食品还能吃吗",
        "这个牛奶是不是过期了", "看看这个生产日期",
        
        # 日常物品识别
        "这个是什么东西", "帮我看看这是啥玩意儿",
        "这个按钮是干什么的", "字太小我看不清",
        
        # 票据、账单
        "帮我看看这个账单", "这个收据上写的什么",
        "帮我读一下这个通知", "这个单子上的字我看不清"
    ]
    
    extra_contexts = [
        "我刚拍的账单", "监控截了张图", "帮我看看孩子的作业",
        "识别一下包装说明", "看看这是不是坏了", "看看这张票是不是假的",
        "帮我确认一下食材有没有坏", "看看这张衣服尺码",
        # 老年人相关
        "这个药盒我看不清", "字太小了", "这个说明书密密麻麻的",
        "眼镜忘戴了", "老花眼看不清", "这个标签上写什么"
    ]
    follow_questions = [
        "需要放大吗？", "我需要聚焦哪里？", "要不要读取文字？",
        "需要我比对库存吗？", "要不要记录识别结果？"
    ]
    system_answers = [
        "这看起来像是一瓶洗洁精。", "图里有一只橘猫。", "文字内容是...",
        "检测到包装有破损。", "识别到的是牛奶保质期。",
        "画面中有一个老人摔倒。", "应该是一张发票"
    ]
    user_followups = [
        "能读一下上面的字吗", "帮我框出重点", "有没异物",
        "换个角度再看看", "识别一下二维码", "帮我记录下来"
    ]
    history_refs = [
        "你还记得我昨天给你的那个药吗", "还是之前那张票据",
        "上次识别过的那个零件", "同款鞋子我还拍了张另一角度"
    ]
    
    data = []
    for i in range(num_samples):
        scenario = random.random()
        turns = []
        
        # 30%使用老年人特有的视觉识别场景
        use_elderly_scenario = random.random() < 0.3

        if scenario < 0.2:
            if use_elderly_scenario:
                turns.append({"speaker": "user", "text": random.choice(elderly_vision_requests)})
            else:
                turns.append({"speaker": "user", "text": random.choice(vision_requests)})
        elif scenario < 0.45:
            turns.append({"speaker": "user", "text": random.choice([
                "在不在", "在吗", "小助手这边"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "在的", "我在", "在这，怎么啦？"
            ])})
            turns.append({"speaker": "user", "text": random.choice([
                random.choice(vision_requests), random.choice(extra_contexts)
            ])})
        elif scenario < 0.7:
            turns.append({"speaker": "user", "text": random.choice([
                "帮我识别一下这个", "给你看张照片", "识别"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "好的，请出示图片", "需要我聚焦哪部分？", "我准备好了，你发照片。"
            ])})
            turns.append({"speaker": "user", "text": random.choice(extra_contexts)})
            if random.random() < 0.5:
                turns.append({"speaker": "system", "text": random.choice(follow_questions)})
                turns.append({"speaker": "user", "text": random.choice([
                    "主要看右下角", "帮我读下中间那行字", "检查一下有没有裂开"
                ])})
            turns.append({"speaker": "system", "text": random.choice(system_answers)})
        elif scenario < 0.9:
            turns.append({"speaker": "user", "text": random.choice(history_refs)})
            turns.append({"speaker": "system", "text": random.choice([
                "有印象，还是同一个吗？", "记得，那是个蓝色包装。",
                "要对比一下吗？"
            ])})
            turns.append({"speaker": "user", "text": random.choice([
                "帮我再确认一次", "这次换个角度看看", "看看是不是同一个瑕疵"
            ])})
            turns.append({"speaker": "system", "text": random.choice(system_answers)})
            if random.random() < 0.5:
                turns.append({"speaker": "user", "text": random.choice(user_followups)})
                turns.append({"speaker": "system", "text": random.choice([
                    "已经帮你记录了。", "我再帮你读取一下文字。",
                    "好了，我标注出来了。"
                ])})
        else:
            turns.append({"speaker": "user", "text": random.choice([
                "识别一下这个二维码", "帮我看看这张图有没有异物",
                "看下这个破损严重吗"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "我先处理图像。", "稍等，我分析一下。", "我放大看看。"
            ])})
            turns.append({"speaker": "system", "text": random.choice(system_answers)})
            turns.append({"speaker": "user", "text": random.choice([
                "那能处理一下吗", "发我识别结果", "帮我记录下来"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "结果已保存。", "我帮你导出成文本。", "好的，记录完成。"
            ])})

        data.append(create_dialogue(f"d_10_{i+1:04d}", turns, 10))
    return data


# ============================================================================
# Label 11: 餐厅推荐
# ============================================================================

def generate_label_11_data(num_samples=500):
    """生成餐厅推荐对话"""
    food_types = [
        "川菜", "粤菜", "火锅", "烧烤", "日料", "西餐", "素食", "本帮菜", "湘菜", "面馆"
    ]
    constraints = [
        "要干净卫生", "预算人均五十", "情侣约会", "适合带孩子", "靠地铁", "能包间",
        "支持外卖", "有停车位", "安静一点", "要营业到很晚"
    ]
    followups = [
        "要评分高的", "想吃辣的", "最好有优惠", "能线上排队吗", "有没有团购",
        "顺便推荐个甜品店", "要不要预约", "附近还有啥玩的吗"
    ]
    system_prompts = [
        "想吃什么类型的？", "有没有具体要求？", "距离多远可以接受？",
        "要坐多少人？", "需要我帮你下单吗？"
    ]
    system_replies = [
        "附近有家评分4.8的川菜馆。", "三公里内有两家火锅正好有折扣。",
        "有家新开的日料店口碑不错。", "附近商场有家烧烤很热门。",
        "我找到一家适合家庭聚餐的餐厅。"
    ]

    data = []
    for i in range(num_samples):
        scenario = random.random()
        turns = []
        food_type = random.choice(food_types)

        if scenario < 0.25:
            turns.append({"speaker": "user", "text": random.choice([
                "附近有什么好吃的", "推荐个餐厅", f"想吃{food_type}",
                "晚上吃啥好", "想找个聚餐的地方"
            ])})
            turns.append({"speaker": "system", "text": random.choice(system_prompts)})
            turns.append({"speaker": "user", "text": random.choice([
                f"最好{random.choice(constraints)}", f"{random.choice(food_types)}有没有好的",
                random.choice(followups)
            ])})
        elif scenario < 0.5:
            turns.append({"speaker": "user", "text": random.choice([
                "我们三个人想吃饭", "约会想找个餐厅", "家里来客人，找地方吃"
            ])})
            turns.append({"speaker": "system", "text": random.choice(system_prompts)})
            turns.append({"speaker": "user", "text": random.choice([
                f"想吃{food_type}", f"人均别超过一百", f"最好{random.choice(constraints)}"
            ])})
            turns.append({"speaker": "system", "text": random.choice(system_replies)})
            if random.random() < 0.5:
                turns.append({"speaker": "user", "text": random.choice([
                    "能帮我订位吗", "要排队吗", "有优惠券吗", "附近停车方便吗"
                ])})
                turns.append({"speaker": "system", "text": random.choice([
                    "我帮你预约一下。", "目前排队约20分钟。", "我给你查下优惠。"
                ])})
        elif scenario < 0.75:
            turns.append({"speaker": "user", "text": random.choice([
                "我在商场这边", "附近有啥好吃的", "想吃点热的",
                "想吃海鲜", "弄点清淡的"
            ])})
            turns.append({"speaker": "system", "text": "正在为您查询..."})
            turns.append({"speaker": "system", "text": random.choice(system_replies)})
            turns.append({"speaker": "user", "text": random.choice([
                random.choice(followups), f"{random.choice(constraints)}行不行"
            ])})
            turns.append({"speaker": "system", "text": random.choice([
                "满足这些要求。", "可能要稍微远一点。", "该店支持线上排队。"
            ])})
        else:
            turns.append({"speaker": "user", "text": random.choice([
                "订个餐厅", "周末聚餐帮我安排", "附近有什么家庭聚餐的地方"
            ])})
            turns.append({"speaker": "system", "text": random.choice(system_prompts)})
            turns.append({"speaker": "user", "text": random.choice([
                f"要能坐{random.randint(4, 10)}人", f"最好靠近地铁", f"想要{food_type}"
            ])})
            turns.append({"speaker": "system", "text": random.choice(system_replies)})
            if random.random() < 0.4:
                turns.append({"speaker": "user", "text": random.choice([
                    "把详细地址发我", "帮我发给朋友", "下次也记得推荐",
                    "帮我加到收藏"
                ])})
                turns.append({"speaker": "system", "text": random.choice([
                    "已经发送到你的手机。", "我帮你收藏好了。", "分享给朋友了。"
                ])})

        data.append(create_dialogue(f"d_11_{i+1:04d}", turns, 11))
    return data


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='生成智能助手对话意图识别数据集')
    parser.add_argument('--samples_per_class', type=int, default=3000,
                       help='每个类别生成的样本数量（默认3000，最少500）')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='输出目录（默认data）')
    parser.add_argument('--split', action='store_true',
                       help='自动划分train/dev/test')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例（默认0.8）')
    parser.add_argument('--dev_ratio', type=float, default=0.1,
                       help='验证集比例（默认0.1）')
    
    args = parser.parse_args()
    samples_per_class = max(500, args.samples_per_class)
    
    print("="*70)
    print("智能助手多轮对话意图识别数据集生成器")
    print("="*70)
    print(f"\n配置:")
    print(f"  每类样本数: {samples_per_class}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  总计: {samples_per_class * 12} 条对话")
    print("\n特点:")
    print("  ✓ 包含基础模板和真实日常表达")
    print("  ✓ 口语化、简短、情绪驱动")
    print("  ✓ 边界案例和容易混淆的样本")
    print("  ✓ 真实联系人、场景化表达\n")
    
    # 生成器映射
    generators = {
        0: generate_label_0_data,
        1: generate_label_1_data,
        2: generate_label_2_data,
        3: generate_label_3_data,
        4: generate_label_4_data,
        5: generate_label_5_data,
        6: generate_label_6_data,
        7: generate_label_7_data,
        8: generate_label_8_data,
        9: generate_label_9_data,
        10: generate_label_10_data,
        11: generate_label_11_data,
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    all_data = []
    
    # 生成各类别数据
    for label in range(12):
        print(f"生成 Label {label} ({INTENT_LABELS[label]})...")
        data = generators[label](samples_per_class)
        
        # 保存单独文件
        filepath = f'{args.output_dir}/label_{label}.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {len(data)} 条 → {filepath}")
        
        all_data.extend(data)
    
    # 如果需要自动划分
    if args.split:
        print(f"\n正在划分数据集 ({args.train_ratio}/{args.dev_ratio}/{1-args.train_ratio-args.dev_ratio})...")
        random.shuffle(all_data)
        
        total = len(all_data)
        train_size = int(total * args.train_ratio)
        dev_size = int(total * args.dev_ratio)
        
        train_data = all_data[:train_size]
        dev_data = all_data[train_size:train_size + dev_size]
        test_data = all_data[train_size + dev_size:]
        
        with open(f'{args.output_dir}/train.json', 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        with open(f'{args.output_dir}/dev.json', 'w', encoding='utf-8') as f:
            json.dump(dev_data, f, ensure_ascii=False, indent=2)
        with open(f'{args.output_dir}/test.json', 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 训练集: {len(train_data)} 条")
        print(f"  ✓ 验证集: {len(dev_data)} 条")
        print(f"  ✓ 测试集: {len(test_data)} 条")
    
    print("\n" + "="*70)
    print("✓ 数据生成完成！")
    print("="*70)
    print(f"\n总计: {len(all_data)} 条对话数据 (12类 × {samples_per_class}条)")
    print(f"文件: {args.output_dir}/label_0.json ~ label_11.json")
    
    if args.split:
        print(f"\n已自动划分:")
        print(f"  {args.output_dir}/train.json")
        print(f"  {args.output_dir}/dev.json")
        print(f"  {args.output_dir}/test.json")
    else:
        print(f"\n提示: 使用 --split 参数可自动划分train/dev/test")
    
    print("="*70)


if __name__ == "__main__":
    main()

