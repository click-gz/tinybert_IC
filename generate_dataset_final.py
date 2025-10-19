#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集生成脚本 - 最终版本
确保所有对话都以用户发言结束
"""

import json
import random
import os

# 设置随机种子
random.seed(42)

# 类别定义
INTENT_LABELS = {
    0: "问题求助/技术支持",
    1: "闲聊",
    2: "天气查询",
    3: "新闻资讯",
    4: "音乐播放",
    5: "提醒设置",
    6: "设备状态查询",
    7: "通话视频",
    8: "安全监测",
    9: "空间操作",
    10: "视觉识别",
    11: "地点推荐"
}

# ============================================================================
# Label 0: 问题求助/技术支持
# ============================================================================

def generate_label_0_data(num_samples=500):
    """生成问题求助类对话，最后一轮必须是求助意图"""
    
    problems = [
        ("电脑", ["开不了机", "死机了", "蓝屏", "运行很慢", "黑屏", "自动重启"]),
        ("手机", ["卡顿", "发热严重", "电池耗电快", "屏幕失灵", "无法充电", "黑屏"]),
        ("WiFi", ["连不上", "信号弱", "频繁断线", "速度很慢", "搜不到网络"]),
        ("软件", ["打不开", "闪退", "安装失败", "卸载不掉", "更新出错", "运行报错"]),
        ("数学题", ["不会做", "解不出来", "看不懂题意", "不知道用什么公式"]),
        ("编程", ["代码报错", "逻辑不对", "不知道怎么实现", "调试不出来"]),
        ("文件", ["删除了怎么恢复", "找不到了", "打不开", "格式转换"]),
        ("设置", ["怎么修改", "在哪里找", "如何重置", "忘记密码了"]),
    ]
    
    data = []
    
    for i in range(num_samples):
        # 生成1-4轮对话，但确保以用户结束
        num_user_turns = random.randint(1, 3)  # 用户发言次数
        category, issues = random.choice(problems)
        issue = random.choice(issues)
        
        turns = []
        
        if num_user_turns == 1:
            # 单轮：只有用户发言
            turns.append({"speaker": "user", "text": f"我的{category}{issue}怎么办"})
        elif num_user_turns == 2:
            # 两轮：用户-系统-用户
            turns.append({"speaker": "user", "text": f"我的{category}{issue}"})
            turns.append({"speaker": "system", "text": "请详细描述一下具体情况。"})
            turns.append({"speaker": "user", "text": "就是突然这样的，怎么办"})
        else:  # num_user_turns == 3
            # 三轮：用户-系统-用户-系统-用户
            turns.append({"speaker": "user", "text": f"我的{category}{issue}"})
            turns.append({"speaker": "system", "text": "我来帮您。能说说具体情况吗？"})
            turns.append({"speaker": "user", "text": f"就是{random.choice(['昨天', '今天早上', '刚才'])}开始的"})
            turns.append({"speaker": "system", "text": "我明白了。建议您先尝试重启。"})
            turns.append({"speaker": "user", "text": "还有其他方法吗"})
        
        data.append({
            "dialogue_id": f"d_0_{i+1:04d}",
            "turns": turns,
            "label": 0
        })
    
    return data


# ============================================================================
# Label 1: 闲聊
# ============================================================================

def generate_label_1_data(num_samples=500):
    """生成闲聊类对话，最后一轮必须是闲聊意图"""
    
    topics = {
        "问候": ["你好", "早上好", "晚上好", "最近怎么样", "今天过得好吗"],
        "兴趣": ["你喜欢什么", "你的爱好是什么", "你喜欢什么颜色", "你喜欢音乐吗"],
        "哲学": ["什么是幸福", "人生的意义", "你对未来的看法", "你会思考吗"],
        "情感": ["你有情感吗", "你会孤独吗", "你开心吗", "你会难过吗"],
        "AI话题": ["你是真正的智能吗", "AI会取代人类吗", "你的梦想是什么"]
    }
    
    data = []
    
    for i in range(num_samples):
        num_user_turns = random.randint(1, 3)
        topic_type = random.choice(list(topics.keys()))
        
        turns = []
        
        if num_user_turns == 1:
            # 单轮：只有用户发言
            turns.append({"speaker": "user", "text": random.choice(topics[topic_type]) + "？"})
        elif num_user_turns == 2:
            # 两轮：用户-系统-用户
            turns.append({"speaker": "user", "text": "你好！"})
            turns.append({"speaker": "system", "text": "您好！很高兴见到您。"})
            turns.append({"speaker": "user", "text": random.choice(topics[topic_type]) + "？"})
        else:  # num_user_turns == 3
            # 三轮：用户-系统-用户-系统-用户
            turns.append({"speaker": "user", "text": "你好！"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": "随便聊聊吧"})
            turns.append({"speaker": "system", "text": "好的，我很乐意和您聊天。"})
            turns.append({"speaker": "user", "text": random.choice(topics[topic_type]) + "？"})
        
        data.append({
            "dialogue_id": f"d_1_{i+1:04d}",
            "turns": turns,
            "label": 1
        })
    
    return data


# ============================================================================
# Label 2: 天气查询
# ============================================================================

def generate_label_2_data(num_samples=500):
    """生成天气查询类对话，最后一轮必须是天气查询意图"""
    
    cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "重庆"]
    times = ["今天", "明天", "后天", "周末", "下周", "这周"]
    weathers = ["晴", "多云", "阴", "小雨", "大雨", "雷阵雨", "雪"]
    
    data = []
    
    for i in range(num_samples):
        num_user_turns = random.randint(1, 3)
        city = random.choice(cities) if random.random() < 0.6 else ""
        time = random.choice(times)
        weather = random.choice(weathers)
        temp = random.randint(15, 35)
        
        turns = []
        
        if num_user_turns == 1:
            # 单轮：只有用户发言
            queries = [
                f"{time}{city}天气怎么样",
                f"{time}{city}会下雨吗",
                f"{time}{city}温度多少",
                f"查一下{time}{city}的天气"
            ]
            turns.append({"speaker": "user", "text": random.choice(queries)})
        elif num_user_turns == 2:
            # 两轮：用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": f"我想查一下{time}{city}的天气"})
        else:  # num_user_turns == 3
            # 三轮：用户-系统-用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": f"我想查一下{time}{city}的天气"})
            turns.append({"speaker": "system", "text": f"正在查询...{time}{city}天气{weather}，温度{temp}度。"})
            follow_ups = [
                "那后天呢",
                "适合穿什么衣服",
                "需要带伞吗",
                "空气质量怎么样"
            ]
            turns.append({"speaker": "user", "text": random.choice(follow_ups)})
        
        data.append({
            "dialogue_id": f"d_2_{i+1:04d}",
            "turns": turns,
            "label": 2
        })
    
    return data


# ============================================================================
# Label 3: 新闻资讯
# ============================================================================

def generate_label_3_data(num_samples=500):
    """生成新闻资讯查询对话，最后一轮必须是新闻查询意图"""
    
    categories = ["科技", "体育", "娱乐", "财经", "国际", "社会"]
    news_items = {
        "科技": ["AI技术突破", "新手机发布", "芯片研发进展", "互联网新政策"],
        "体育": ["世界杯预选赛", "篮球联赛", "网球大满贯", "奥运会准备"],
        "娱乐": ["新电影上映", "明星新专辑", "综艺节目", "颁奖典礼"],
        "财经": ["股市行情", "经济政策", "企业财报", "汇率变化"],
        "国际": ["外交会晤", "国际合作", "地区局势", "贸易协定"],
        "社会": ["民生政策", "教育改革", "医疗进展", "环保行动"]
    }
    
    data = []
    
    for i in range(num_samples):
        num_user_turns = random.randint(1, 3)
        category = random.choice(categories)
        items = news_items[category]
        
        turns = []
        
        if num_user_turns == 1:
            # 单轮：只有用户发言
            queries = [
                f"最近有什么{category}新闻",
                f"{category}方面有什么动态",
                f"播报一下{category}新闻",
                f"{category}界有什么新消息"
            ]
            turns.append({"speaker": "user", "text": random.choice(queries)})
        elif num_user_turns == 2:
            # 两轮：用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": f"我想了解一下{category}新闻"})
        else:  # num_user_turns == 3
            # 三轮：用户-系统-用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": f"我想了解一下{category}新闻"})
            selected_items = random.sample(items, min(2, len(items)))
            turns.append({"speaker": "system", "text": f"正在检索{category}新闻...最新动态：{selected_items[0]}、{selected_items[1]}。"})
            follow_ups = [
                "第一条详细说说",
                "能展开讲讲吗",
                f"{selected_items[0]}具体什么情况",
                "还有其他的吗"
            ]
            turns.append({"speaker": "user", "text": random.choice(follow_ups)})
        
        data.append({
            "dialogue_id": f"d_3_{i+1:04d}",
            "turns": turns,
            "label": 3
        })
    
    return data


# ============================================================================
# Label 4: 音乐播放
# ============================================================================

def generate_label_4_data(num_samples=500):
    """生成音乐播放类对话，最后一轮必须是音乐播放意图"""
    
    singers = ["周杰伦", "邓紫棋", "薛之谦", "林俊杰", "陈奕迅", "王菲", "李荣浩"]
    songs = ["青花瓷", "告白气球", "夜曲", "稻香", "七里香", "演员", "修炼爱情"]
    music_types = ["流行", "摇滚", "古典", "轻音乐", "电子", "民谣", "爵士"]
    operations = ["暂停", "下一首", "上一首", "音量大一点", "音量小一点", "切歌"]
    
    data = []
    
    for i in range(num_samples):
        num_user_turns = random.randint(1, 3)
        
        turns = []
        
        if num_user_turns == 1:
            # 单轮：只有用户发言
            choice = random.random()
            if choice < 0.3:
                singer = random.choice(singers)
                turns.append({"speaker": "user", "text": f"播放{singer}的歌"})
            elif choice < 0.6:
                song = random.choice(songs)
                turns.append({"speaker": "user", "text": f"我想听{song}"})
            else:
                music_type = random.choice(music_types)
                turns.append({"speaker": "user", "text": f"来点{music_type}音乐"})
        elif num_user_turns == 2:
            # 两轮：用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            choice = random.random()
            if choice < 0.3:
                singer = random.choice(singers)
                turns.append({"speaker": "user", "text": f"播放{singer}的歌"})
            else:
                music_type = random.choice(music_types)
                turns.append({"speaker": "user", "text": f"来点{music_type}音乐"})
        else:  # num_user_turns == 3
            # 三轮：用户-系统-用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            choice = random.random()
            if choice < 0.3:
                singer = random.choice(singers)
                turns.append({"speaker": "user", "text": f"播放{singer}的歌"})
            else:
                music_type = random.choice(music_types)
                turns.append({"speaker": "user", "text": f"来点{music_type}音乐"})
            song = random.choice(songs)
            turns.append({"speaker": "system", "text": f"正在为您播放《{song}》。"})
            turns.append({"speaker": "user", "text": random.choice(operations)})
        
        data.append({
            "dialogue_id": f"d_4_{i+1:04d}",
            "turns": turns,
            "label": 4
        })
    
    return data


# ============================================================================
# Label 5: 提醒设置
# ============================================================================

def generate_label_5_data(num_samples=500):
    """生成提醒设置类对话，最后一轮必须是提醒相关意图"""
    
    times = ["早上7点", "上午9点", "中午12点", "下午3点", "下午6点", "晚上8点", "明天", "后天", "每天"]
    events = ["开会", "喝水", "吃药", "运动", "休息", "起床", "去银行", "取快递", "打电话"]
    
    data = []
    
    for i in range(num_samples):
        num_user_turns = random.randint(1, 3)
        time = random.choice(times)
        event = random.choice(events)
        
        turns = []
        
        if num_user_turns == 1:
            # 单轮：只有用户发言
            requests = [
                f"{time}提醒我{event}",
                f"设置{time}的{event}提醒",
                f"{time}别忘了提醒我{event}",
                f"帮我设个{time}{event}的提醒"
            ]
            turns.append({"speaker": "user", "text": random.choice(requests)})
        elif num_user_turns == 2:
            # 两轮：用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": f"我想设置{time}的{event}提醒"})
        else:  # num_user_turns == 3
            # 三轮：用户-系统-用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": f"我想设置{time}的{event}提醒"})
            turns.append({"speaker": "system", "text": f"已为您设置{time}的{event}提醒。"})
            follow_ops = [
                "查看我的所有提醒",
                "再加个备注",
                "修改一下时间",
                f"取消{time}的提醒",
                "删除这个提醒"
            ]
            turns.append({"speaker": "user", "text": random.choice(follow_ops)})
        
        data.append({
            "dialogue_id": f"d_5_{i+1:04d}",
            "turns": turns,
            "label": 5
        })
    
    return data


# ============================================================================
# Label 6: 设备状态查询
# ============================================================================

def generate_label_6_data(num_samples=500):
    """生成设备状态查询类对话，最后一轮必须是设备查询意图"""
    
    queries = [
        ("电量", ["还有多少电量", "电池还剩多少", "能用多久", "电量情况"]),
        ("音量", ["音量多少", "音量是多大", "声音大小"]),
        ("存储", ["存储空间还有多少", "内存使用情况", "还能存多少"]),
        ("网络", ["WiFi连接状态", "网络怎么样", "信号强度"]),
        ("蓝牙", ["蓝牙开了吗", "蓝牙状态"])
    ]
    
    data = []
    
    for i in range(num_samples):
        num_user_turns = random.randint(1, 3)
        query_type, query_list = random.choice(queries)
        
        turns = []
        
        if num_user_turns == 1:
            # 单轮：只有用户发言
            turns.append({"speaker": "user", "text": random.choice(query_list)})
        elif num_user_turns == 2:
            # 两轮：用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": f"我想查看一下{query_type}状态"})
        else:  # num_user_turns == 3
            # 三轮：用户-系统-用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": f"我想查看一下{query_type}状态"})
            
            # 系统回复状态
            if query_type == "电量":
                value = random.randint(30, 95)
                hours = random.randint(3, 10)
                response = f"当前电量{value}%，预计可使用{hours}小时。"
            elif query_type == "音量":
                value = random.randint(40, 90)
                response = f"当前音量{value}%。"
            elif query_type == "存储":
                used = random.randint(50, 100)
                total = 128
                response = f"总容量{total}GB，已使用{used}GB，剩余{total-used}GB。"
            elif query_type == "网络":
                response = "当前连接到HomeWiFi，信号强度良好。"
            else:
                response = "蓝牙已开启，已连接耳机。"
            
            turns.append({"speaker": "system", "text": response})
            
            # 最后一轮：用户设备相关操作
            if query_type == "电量":
                adjustments = ["开启省电模式", "关闭省电模式", "查看耗电排行"]
            elif query_type == "音量":
                adjustments = [f"调到{random.randint(50,80)}%", "调大一点", "调小一点"]
            elif query_type == "存储":
                adjustments = ["清理垃圾文件", "查看详细占用", "卸载不用的应用"]
            elif query_type == "网络":
                adjustments = ["切换到5G网络", "断开WiFi", "连接其他网络"]
            else:
                adjustments = ["关闭蓝牙", "断开耳机连接", "搜索设备"]
            
            turns.append({"speaker": "user", "text": random.choice(adjustments)})
        
        data.append({
            "dialogue_id": f"d_6_{i+1:04d}",
            "turns": turns,
            "label": 6
        })
    
    return data


# ============================================================================
# Label 7: 通话视频
# ============================================================================

def generate_label_7_data(num_samples=500):
    """生成通话视频类对话，最后一轮必须是通话相关意图"""
    
    contacts = ["张三", "李四", "王五", "赵六", "小明", "小红", "老王", "小李", "妈妈", "爸爸"]
    call_types = ["打电话", "视频通话", "语音通话"]
    
    data = []
    
    for i in range(num_samples):
        num_user_turns = random.randint(1, 3)
        contact = random.choice(contacts)
        call_type = random.choice(call_types)
        
        turns = []
        
        if num_user_turns == 1:
            # 单轮：只有用户发言
            requests = [
                f"给{contact}{call_type}",
                f"拨打{contact}",
                f"呼叫{contact}",
                f"和{contact}{call_type}"
            ]
            turns.append({"speaker": "user", "text": random.choice(requests)})
        elif num_user_turns == 2:
            # 两轮：用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": f"我想给{contact}{call_type}"})
        else:  # num_user_turns == 3
            # 三轮：用户-系统-用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": f"我想给{contact}{call_type}"})
            turns.append({"speaker": "system", "text": f"正在拨打{contact}的电话...已接通。"})
            adjustments = ["开启免提", "关闭免提", "静音", "取消静音", "开启视频", "挂断"]
            turns.append({"speaker": "user", "text": random.choice(adjustments)})
        
        data.append({
            "dialogue_id": f"d_7_{i+1:04d}",
            "turns": turns,
            "label": 7
        })
    
    return data


# ============================================================================
# Label 8: 安全监测
# ============================================================================

def generate_label_8_data(num_samples=500):
    """生成安全监测类对话，最后一轮必须是安全监测意图"""
    
    situations = ["摔倒", "心率异常", "血压偏高", "体温异常", "长时间未活动"]
    
    data = []
    
    for i in range(num_samples):
        num_user_turns = random.randint(1, 3)
        situation = random.choice(situations)
        
        turns = []
        
        if num_user_turns == 1:
            # 单轮：只有用户发言
            requests = [
                f"我刚才{situation}了",
                "帮我检测一下健康状况",
                "监测我的心率",
                "我感觉不太舒服",
                "检查一下血压"
            ]
            turns.append({"speaker": "user", "text": random.choice(requests)})
        elif num_user_turns == 2:
            # 两轮：用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": "我想检查一下健康状况"})
        else:  # num_user_turns == 3
            # 三轮：用户-系统-用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": "我想检查一下健康状况"})
            turns.append({"speaker": "system", "text": "正在检测...请问您需要帮助吗？"})
            responses = [
                "我没事",
                "有点不舒服",
                "帮我联系家人",
                "继续监测",
                "详细数据是什么"
            ]
            turns.append({"speaker": "user", "text": random.choice(responses)})
        
        data.append({
            "dialogue_id": f"d_8_{i+1:04d}",
            "turns": turns,
            "label": 8
        })
    
    return data


# ============================================================================
# Label 9: 空间操作
# ============================================================================

def generate_label_9_data(num_samples=500):
    """生成空间操作类对话，最后一轮必须是空间操作意图"""
    
    rooms = ["客厅", "卧室", "厨房", "书房", "阳台"]
    objects = ["灯", "空调", "电视", "窗帘", "门", "窗户"]
    items = ["遥控器", "手机", "钥匙", "书", "杯子", "眼镜"]
    directions = ["向前", "向后", "向左", "向右"]
    
    data = []
    
    for i in range(num_samples):
        num_user_turns = random.randint(1, 3)
        room = random.choice(rooms)
        
        turns = []
        
        if num_user_turns == 1:
            # 单轮：只有用户发言
            choice = random.random()
            if choice < 0.4:
                obj = random.choice(objects)
                actions = [f"去{room}关闭{obj}", f"到{room}打开{obj}", f"去{room}看看{obj}开着吗"]
                turns.append({"speaker": "user", "text": random.choice(actions)})
            else:
                item = random.choice(items)
                turns.append({"speaker": "user", "text": f"去{room}帮我拿{item}"})
        elif num_user_turns == 2:
            # 两轮：用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            choice = random.random()
            if choice < 0.4:
                obj = random.choice(objects)
                turns.append({"speaker": "user", "text": f"去{room}关闭{obj}"})
            else:
                item = random.choice(items)
                turns.append({"speaker": "user", "text": f"去{room}帮我拿{item}"})
        else:  # num_user_turns == 3
            # 三轮：用户-系统-用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            choice = random.random()
            if choice < 0.4:
                obj = random.choice(objects)
                turns.append({"speaker": "user", "text": f"去{room}关闭{obj}"})
            else:
                item = random.choice(items)
                turns.append({"speaker": "user", "text": f"去{room}帮我拿{item}"})
            turns.append({"speaker": "system", "text": f"正在前往{room}...已到达。"})
            continues = [
                f"{random.choice(directions)}一点",
                "找到了吗",
                "帮我检查一下",
                "停"
            ]
            turns.append({"speaker": "user", "text": random.choice(continues)})
        
        data.append({
            "dialogue_id": f"d_9_{i+1:04d}",
            "turns": turns,
            "label": 9
        })
    
    return data


# ============================================================================
# Label 10: 视觉识别
# ============================================================================

def generate_label_10_data(num_samples=500):
    """生成视觉识别类对话，最后一轮必须是视觉识别意图"""
    
    objects = ["苹果", "香蕉", "杯子", "手机", "书", "椅子", "桌子", "电脑", "笔"]
    colors = ["红色", "蓝色", "黄色", "绿色", "白色", "黑色"]
    
    data = []
    
    for i in range(num_samples):
        num_user_turns = random.randint(1, 3)
        obj = random.choice(objects)
        color = random.choice(colors)
        
        turns = []
        
        if num_user_turns == 1:
            # 单轮：只有用户发言
            requests = [
                "这是什么",
                "面前是什么东西",
                "帮我看看这个",
                "识别一下",
                "这个叫什么",
                "照片里有什么"
            ]
            turns.append({"speaker": "user", "text": random.choice(requests)})
        elif num_user_turns == 2:
            # 两轮：用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": "我想识别一下这个东西"})
        else:  # num_user_turns == 3
            # 三轮：用户-系统-用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            turns.append({"speaker": "user", "text": "我想识别一下这个东西"})
            choice = random.random()
            if choice < 0.8:
                turns.append({"speaker": "system", "text": f"正在识别...这是一个{color}的{obj}。"})
            else:
                turns.append({"speaker": "system", "text": "正在通过OCR识别...检测到文字：'欢迎使用'。"})
            follow_ups = [
                "能详细描述吗",
                "还有别的吗",
                "在哪里",
                "是什么材质",
                "识别一下文字"
            ]
            turns.append({"speaker": "user", "text": random.choice(follow_ups)})
        
        data.append({
            "dialogue_id": f"d_10_{i+1:04d}",
            "turns": turns,
            "label": 10
        })
    
    return data


# ============================================================================
# Label 11: 地点推荐
# ============================================================================

def generate_label_11_data(num_samples=500):
    """生成地点推荐类对话，最后一轮必须是地点推荐意图"""
    
    place_types = ["餐厅", "咖啡厅", "电影院", "商场", "公园", "健身房", "书店"]
    cuisines = ["川菜", "粤菜", "日料", "韩餐", "西餐", "火锅"]
    
    data = []
    
    for i in range(num_samples):
        num_user_turns = random.randint(1, 3)
        place_type = random.choice(place_types)
        
        turns = []
        
        if num_user_turns == 1:
            # 单轮：只有用户发言
            if place_type == "餐厅":
                cuisine = random.choice(cuisines)
                requests = [
                    f"附近有什么好的{cuisine}餐厅",
                    f"推荐一家{cuisine}馆",
                    f"哪里有好吃的{cuisine}"
                ]
            else:
                requests = [
                    f"附近有{place_type}吗",
                    f"推荐一家{place_type}",
                    f"找一个{place_type}"
                ]
            turns.append({"speaker": "user", "text": random.choice(requests)})
        elif num_user_turns == 2:
            # 两轮：用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            if place_type == "餐厅":
                cuisine = random.choice(cuisines)
                turns.append({"speaker": "user", "text": f"我想找{cuisine}餐厅"})
            else:
                turns.append({"speaker": "user", "text": f"我想找{place_type}"})
        else:  # num_user_turns == 3
            # 三轮：用户-系统-用户-系统-用户
            turns.append({"speaker": "user", "text": "你好"})
            turns.append({"speaker": "system", "text": "您好！有什么可以帮您的吗？"})
            if place_type == "餐厅":
                cuisine = random.choice(cuisines)
                turns.append({"speaker": "user", "text": f"我想找{cuisine}餐厅"})
            else:
                turns.append({"speaker": "user", "text": f"我想找{place_type}"})
            rating = round(random.uniform(4.3, 4.9), 1)
            distance = random.randint(200, 2000)
            turns.append({"speaker": "system", "text": f"为您找到3家{place_type}：A店(评分{rating})、B店、C店，距离您{distance}米。"})
            follow_ups = [
                "第一家怎么样",
                "营业时间",
                "人均消费多少",
                "停车方便吗",
                "需要预订吗"
            ]
            turns.append({"speaker": "user", "text": random.choice(follow_ups)})
        
        data.append({
            "dialogue_id": f"d_11_{i+1:04d}",
            "turns": turns,
            "label": 11
        })
    
    return data


# ============================================================================
# 主函数
# ============================================================================

def main():
    """生成所有类别的数据"""
    print("="*70)
    print(" TinyBERT 多轮对话意图识别 - 数据集生成 最终版本")
    print("="*70)
    print("\n特点：")
    print("  - 所有对话都以用户发言结束（包括单轮对话）")
    print("  - 每个对话的标签基于用户最后一轮发言的意图")
    print("  - 对话逻辑连贯，系统回复与用户请求匹配")
    print("  - 每个类别精确500条数据\n")
    
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
    
    total_samples = 0
    
    for label in range(12):
        print(f"生成 Label {label} ({INTENT_LABELS[label]})...")
        data = generators[label](500)
        
        # 保存数据
        filepath = f'data/label_{label}.json'
        os.makedirs('data', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"  OK {len(data)} 条数据已保存到 {filepath}")
        total_samples += len(data)
    
    print(f"\n{'='*70}")
    print(f" OK 数据生成完成！")
    print(f"{'='*70}")
    print(f"\n总计: {total_samples} 条对话数据")
    print(f"文件: data/label_0.json ~ data/label_11.json")
    print(f"\n数据质量保证：")
    print(f"  OK 所有对话都以用户发言结束")
    print(f"  OK 每个对话的标签基于最后一轮用户意图")
    print(f"  OK 对话逻辑连贯")
    print(f"  OK 每个类别精确500条")


if __name__ == "__main__":
    main()
