#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web Demo for Dialogue Intent Classification using Gradio

Usage:
    python demo_web.py --checkpoint checkpoints/teacher/best_model.pt
    
Then open http://localhost:7860 in your browser
"""

import argparse
import gradio as gr
import torch
from transformers import BertTokenizer
import json
from typing import List, Dict
import pandas as pd

from src import (
    MultiTurnDialogueClassifier,
    load_config,
    get_device
)


# Intent label mapping (智能助手意图识别)
INTENT_LABELS = {
    0: "问题求助/技术支持",
    1: "闲聊",
    2: "天气查询",
    3: "新闻资讯",
    4: "音乐播放",
    5: "提醒设置",
    6: "设备状态查询",
    7: "通话视频",
    8: "跌倒监测确认",
    9: "空间操作",
    10: "视觉识别",
    11: "餐厅推荐"
}


class WebPredictor:
    """Web界面预测器"""
    
    def __init__(self, checkpoint_path: str, config_path: str = 'config/teacher_config.yaml'):
        # 加载配置
        self.config = load_config(config_path)
        self.device = get_device(self.config.get('device', 'cuda'))
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.config['model']['encoder_name'])
        
        # 创建并加载模型
        self.model = MultiTurnDialogueClassifier(
            encoder_name=self.config['model']['encoder_name'],
            num_labels=self.config['model']['num_labels'],
            max_seq_length=self.config['data']['max_seq_length'],
            dropout=self.config['model']['dropout']
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ 模型加载成功！设备: {self.device}")
    
    def predict_from_turns(self, turns_text: str) -> tuple:
        """
        从多轮对话文本预测意图（与训练时格式完全一致）
        
        Args:
            turns_text: 对话文本，每行一轮，格式: [说话人] 话语内容
            
        Returns:
            (预测结果文本, Top-K预测表格, 概率分布图)
        """
        try:
            # 解析对话
            turns = []
            for line in turns_text.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # 解析 [说话人] 话语
                if line.startswith('[') and ']' in line:
                    speaker_end = line.index(']')
                    speaker = line[1:speaker_end].strip().lower()
                    text = line[speaker_end+1:].strip()
                else:
                    # 默认为客户
                    speaker = 'user'
                    text = line
                
                # 统一转换为 user/system
                if speaker in ['顾客', '客户', '用户', 'user', 'customer']:
                    speaker = 'user'
                else:
                    speaker = 'system'
                
                turns.append({
                    'speaker': speaker,
                    'text': text
                })
            
            if not turns:
                return "❌ 请输入有效的对话内容！", None, None
            
            # 拼接对话 - 与训练时的dataset.py格式完全一致
            dialogue_parts = []
            for turn in turns:
                speaker = turn['speaker'].upper()  # USER or SYSTEM
                text = turn['text']
                dialogue_parts.append(f"[{speaker}] {text}")
            
            full_dialogue = " ".join(dialogue_parts)
            
            # Tokenize - 与训练时完全一致
            encoded = self.tokenizer(
                full_dialogue,
                max_length=self.config['data']['max_seq_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            for key in encoded:
                encoded[key] = encoded[key].to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=-1)
                
                predicted_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, predicted_id].item()
                all_probs = probs[0].cpu().numpy()
            
            # 格式化结果
            result_text = f"""
## 🎯 预测结果

**意图**: {INTENT_LABELS.get(predicted_id, '未知')}  
**置信度**: {confidence:.4f} ({confidence*100:.2f}%)  

### 📝 输入对话:
"""
            for i, turn in enumerate(turns, 1):
                speaker_cn = "用户" if turn['speaker'] == 'user' else "客服"
                result_text += f"{i}. **[{speaker_cn}]**: {turn['text']}\n"
            
            # Top-K预测表格
            top_k = min(5, len(all_probs))
            top_indices = torch.topk(torch.tensor(all_probs), k=top_k).indices.numpy()
            
            top_k_data = []
            for rank, idx in enumerate(top_indices, 1):
                top_k_data.append({
                    '排名': rank,
                    '意图': INTENT_LABELS.get(int(idx), '未知'),
                    '置信度': f"{all_probs[idx]:.4f}",
                    '百分比': f"{all_probs[idx]*100:.2f}%"
                })
            
            top_k_df = pd.DataFrame(top_k_data)
            
            # 概率分布图
            prob_data = {
                '意图': [INTENT_LABELS.get(i, f'Intent_{i}') for i in range(len(all_probs))],
                '概率': all_probs.tolist()
            }
            prob_df = pd.DataFrame(prob_data)
            prob_df = prob_df.sort_values('概率', ascending=False)
            
            return result_text, top_k_df, prob_df
            
        except Exception as e:
            return f"❌ 预测出错: {str(e)}", None, None


def create_demo(predictor: WebPredictor):
    """创建Gradio界面"""
    
    # 示例对话（智能助手场景）
    examples = [
        "[用户] 你好\n[助手] 您好！有什么可以帮您的吗？\n[用户] 今天天气怎么样",
        "[用户] 播放一首歌\n[助手] 好的，请问想听什么歌？\n[用户] 周杰伦的歌",
        "[用户] 我的WiFi连不上了\n[助手] 我来帮您排查。请问其他设备能连接吗？\n[用户] 其他设备可以",
        "[用户] 还有多少电量\n[助手] 当前电量65%，预计可使用5小时。\n[用户] 开启省电模式",
        "[用户] 明天早上7点提醒我开会\n[助手] 好的，已为您设置明天7:00的提醒。\n[用户] 谢谢",
        "[用户] 这是什么东西\n[助手] 我来识别一下...这是一个遥控器。\n[用户] 能详细介绍一下吗",
        "[用户] 去厨房帮我拿遥控器\n[助手] 好的，正在前往厨房。\n[用户] 快点",
        "[用户] 附近有什么好吃的\n[助手] 正在为您查询附近餐厅...\n[用户] 要评分高的",
    ]
    
    with gr.Blocks(title="对话意图分类系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 智能助手多轮对话意图分类系统
        
        ## 使用说明
        1. 在左侧文本框中输入多轮对话，每行一轮
        2. 格式: `[说话人] 话语内容`，例如 `[用户] 你好` 或 `[助手] 您好`
        3. 说话人可以是: **user/用户** 或 **system/助手/机器人**（中英文均可）
        4. 点击"预测"按钮查看结果
        5. 或点击下方示例快速体验
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="输入多轮对话",
                    placeholder="[用户] 你好\n[助手] 您好！有什么可以帮您的吗？\n[用户] 今天天气怎么样",
                    lines=10,
                    max_lines=20
                )
                
                predict_btn = gr.Button("🎯 预测意图", variant="primary", size="lg")
                
                gr.Markdown("### 📋 快速示例")
                gr.Examples(
                    examples=examples,
                    inputs=input_text,
                    label="点击使用示例"
                )
            
            with gr.Column(scale=1):
                result_text = gr.Markdown(label="预测结果")
                
                with gr.Accordion("📊 Top-5 预测结果", open=True):
                    top_k_table = gr.Dataframe(
                        headers=['排名', '意图', '置信度', '百分比'],
                        label="Top-5 预测"
                    )
                
                with gr.Accordion("📈 所有类别概率分布", open=False):
                    prob_plot = gr.BarPlot(
                        x="意图",
                        y="概率",
                        title="意图分类概率分布",
                        x_title="意图类别",
                        y_title="概率",
                        height=400,
                        width=600,
                    )
        
        # 绑定预测函数
        predict_btn.click(
            fn=predictor.predict_from_turns,
            inputs=input_text,
            outputs=[result_text, top_k_table, prob_plot]
        )
        
        gr.Markdown("""
        ---
        ### 💡 意图类别说明
        
        本系统支持以下12种智能助手对话意图分类：
        - **问题求助/技术支持**: 用户遇到问题寻求帮助（如WiFi连不上、设备故障等）
        - **闲聊**: 非任务导向的随意聊天（问候、兴趣话题等）
        - **天气查询**: 查询天气信息、温度、降雨概率等
        - **新闻资讯**: 查询各类新闻动态、热点事件
        - **音乐播放**: 播放音乐、切换歌曲、调整音量等
        - **提醒设置**: 设置提醒、闹钟、查看或管理提醒
        - **设备状态查询**: 查询或调整设备状态（电量、音量、存储、网络等）
        - **通话视频**: 发起通话、视频、调整通话设置
        - **跌倒监测确认**: 用户跌倒检测识别后进行确认
        - **空间操作**: 控制机器人在空间中移动、操作物体
        - **视觉识别**: 识别物体、文字OCR、图像分析
        - **餐厅推荐**: 推荐餐厅场所
        """)
    
    return demo


def main():
    parser = argparse.ArgumentParser(description='对话意图分类Web Demo')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/teacher/best_model.pt',
        help='模型检查点路径'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/teacher_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Web服务端口'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='生成公网分享链接'
    )
    
    args = parser.parse_args()
    
    # 创建预测器
    print("\n正在加载模型...")
    predictor = WebPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )
    
    # 创建并启动Demo
    print("\n正在启动Web界面...")
    demo = create_demo(predictor)
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )


if __name__ == '__main__':
    main()

