#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teacher Model Inference Script
Load trained model and make predictions on custom dialogues

Usage:
    # Single prediction
    python inference.py --checkpoint checkpoints/teacher/best_model.pt --dialogue "你好|顾客|我想查询余额|顾客"
    
    # Interactive mode
    python inference.py --checkpoint checkpoints/teacher/best_model.pt --interactive
"""

import argparse
import torch
from transformers import BertTokenizer
import json
from typing import List, Dict
import logging

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


class DialogueClassifierPredictor:
    """对话意图分类预测器"""
    
    def __init__(self, checkpoint_path: str, config_path: str = 'config/teacher_config.yaml', device: str = None):
        """
        初始化预测器
        
        Args:
            checkpoint_path: 模型检查点路径
            config_path: 配置文件路径
            device: 设备 (cuda/cpu)，None则自动选择
        """
        # 加载配置
        self.config = load_config(config_path)
        
        # 设置设备
        if device is None:
            device = self.config.get('device', 'cuda')
        self.device = get_device(device)
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.config['model']['encoder_name'])
        
        # 创建模型
        self.model = MultiTurnDialogueClassifier(
            encoder_name=self.config['model']['encoder_name'],
            num_labels=self.config['model']['num_labels'],
            max_seq_length=self.config['data']['max_seq_length'],
            dropout=self.config['model']['dropout']
        )
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"模型加载成功！")
        logging.info(f"设备: {self.device}")
        logging.info(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def parse_dialogue_string(self, dialogue_str: str) -> List[Dict[str, str]]:
        """
        解析对话字符串
        
        格式: "text1|speaker1|text2|speaker2|..." (speaker可以是user/system或中文)
        或: JSON格式的turns列表 [{"speaker": "user", "text": "..."}]
        
        Args:
            dialogue_str: 对话字符串
            
        Returns:
            turns列表，格式: [{"speaker": "user"/"system", "text": "..."}]
        """
        # 尝试解析JSON格式
        try:
            turns = json.loads(dialogue_str)
            if isinstance(turns, list):
                # 确保格式正确
                formatted_turns = []
                for turn in turns:
                    speaker = turn.get('speaker', 'user').lower()
                    text = turn.get('text', turn.get('utterance', ''))
                    # 统一转换为 user/system
                    if speaker in ['顾客', '客户', '用户', 'user', 'customer']:
                        speaker = 'user'
                    else:
                        speaker = 'system'
                    formatted_turns.append({'speaker': speaker, 'text': text})
                return formatted_turns
        except:
            pass
        
        # 解析分隔符格式
        parts = dialogue_str.split('|')
        if len(parts) % 2 != 0:
            raise ValueError("对话格式错误！应为: text1|speaker1|text2|speaker2|...")
        
        turns = []
        for i in range(0, len(parts), 2):
            text = parts[i].strip()
            speaker = parts[i+1].strip().lower()
            # 统一转换为 user/system
            if speaker in ['顾客', '客户', '用户', 'user', 'customer']:
                speaker = 'user'
            else:
                speaker = 'system'
            turns.append({
                'speaker': speaker,
                'text': text
            })
        
        return turns
    
    def preprocess_dialogue(self, turns: List[Dict[str, str]]) -> Dict:
        """
        预处理对话数据（与训练时格式完全一致）
        
        Args:
            turns: 对话轮次列表，格式: [{"speaker": "user"/"system", "text": "..."}]
            
        Returns:
            编码后的输入字典
        """
        # 拼接对话 - 与训练时的dataset.py格式完全一致
        dialogue_parts = []
        for turn in turns:
            speaker = turn.get('speaker', 'user').upper()  # USER or SYSTEM
            text = turn.get('text', turn.get('utterance', ''))
            # 添加说话人标记，格式: [USER] text 或 [SYSTEM] text
            dialogue_parts.append(f"[{speaker}] {text}")
        
        # Join all turns with space
        full_dialogue = " ".join(dialogue_parts)
        
        # Tokenize - 与训练时完全一致
        encoded = self.tokenizer(
            full_dialogue,
            max_length=self.config['data']['max_seq_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        for key in encoded:
            encoded[key] = encoded[key].to(self.device)
        
        return encoded
    
    def predict(self, dialogue_input) -> Dict:
        """
        预测对话意图
        
        Args:
            dialogue_input: 对话输入，可以是:
                - 字符串格式: "utterance1|speaker1|utterance2|speaker2|..."
                - 列表格式: [{"utterance": "...", "speaker": "..."}]
                
        Returns:
            预测结果字典，包含:
                - intent_id: 预测的意图ID
                - intent_name: 预测的意图名称
                - confidence: 置信度
                - probabilities: 所有类别的概率分布
        """
        # 解析输入
        if isinstance(dialogue_input, str):
            turns = self.parse_dialogue_string(dialogue_input)
        elif isinstance(dialogue_input, list):
            turns = dialogue_input
        else:
            raise ValueError("输入格式错误！")
        
        # 预处理
        encoded = self.preprocess_dialogue(turns)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            
            # 获取预测结果
            predicted_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, predicted_id].item()
            
            # 获取所有类别概率
            all_probs = probs[0].cpu().numpy()
        
        result = {
            'intent_id': predicted_id,
            'intent_name': INTENT_LABELS.get(predicted_id, f"Unknown_{predicted_id}"),
            'confidence': confidence,
            'probabilities': {
                INTENT_LABELS.get(i, f"Unknown_{i}"): float(prob) 
                for i, prob in enumerate(all_probs)
            },
            'top_k_predictions': []
        }
        
        # 获取Top-3预测
        top_k_indices = torch.topk(probs[0], k=min(3, len(all_probs))).indices.cpu().numpy()
        for idx in top_k_indices:
            result['top_k_predictions'].append({
                'intent_id': int(idx),
                'intent_name': INTENT_LABELS.get(int(idx), f"Unknown_{idx}"),
                'confidence': float(all_probs[idx])
            })
        
        return result
    
    def format_result(self, dialogue_input, result: Dict) -> str:
        """格式化输出结果"""
        output = "\n" + "="*70 + "\n"
        output += "对话意图分类结果\n"
        output += "="*70 + "\n"
        
        # 显示输入对话
        if isinstance(dialogue_input, str):
            turns = self.parse_dialogue_string(dialogue_input)
        else:
            turns = dialogue_input
        
        output += "\n输入对话:\n"
        for i, turn in enumerate(turns, 1):
            speaker = turn.get('speaker', 'user').upper()
            text = turn.get('text', turn.get('utterance', ''))
            speaker_cn = "用户" if speaker == "USER" else "客服"
            output += f"  轮次{i} [{speaker_cn}]: {text}\n"
        
        # 显示预测结果
        output += f"\n预测意图: {result['intent_name']}\n"
        output += f"置信度: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)\n"
        
        # 显示Top-3预测
        output += f"\nTop-3 预测:\n"
        for i, pred in enumerate(result['top_k_predictions'], 1):
            output += f"  {i}. {pred['intent_name']:<15} - {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)\n"
        
        output += "="*70 + "\n"
        
        return output


def interactive_mode(predictor: DialogueClassifierPredictor):
    """交互式预测模式"""
    print("\n" + "="*70)
    print("智能助手多轮对话意图分类")
    print("="*70)
    print("\n输入格式: text1|speaker1|text2|speaker2|...")
    print("说话人可以是: user/用户 或 system/助手/机器人")
    print("输入 'quit' 或 'exit' 退出\n")
    print("示例:")
    print("  今天天气怎么样|user|正在查询...今天晴，温度25度|system|需要带伞吗|user")
    print("  或使用中文: 播放音乐|用户|好的，想听什么歌|助手|周杰伦的歌|用户")
    print("\n支持的意图: 天气查询、音乐播放、问题求助、设备查询、提醒设置等12种")
    print("="*70 + "\n")
    
    while True:
        try:
            # 获取用户输入
            dialogue_input = input("请输入对话 (或输入示例编号1-5): ").strip()
            
            if dialogue_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！")
                break
            
            if not dialogue_input:
                continue
            
            # 处理示例对话
            if dialogue_input == '1':
                dialogue_input = "你好|user|您好！有什么可以帮您的吗？|system|今天天气怎么样|user"
                print(f"使用示例1 (天气查询): {dialogue_input}")
            elif dialogue_input == '2':
                dialogue_input = "播放一首歌|user|好的，请问想听什么歌？|system|周杰伦的歌|user"
                print(f"使用示例2 (音乐播放): {dialogue_input}")
            elif dialogue_input == '3':
                dialogue_input = "我的WiFi连不上了|user|我来帮您排查。请问其他设备能连接吗？|system|其他设备可以|user"
                print(f"使用示例3 (问题求助): {dialogue_input}")
            elif dialogue_input == '4':
                dialogue_input = "还有多少电量|user|当前电量65%，预计可使用5小时。|system|开启省电模式|user"
                print(f"使用示例4 (设备状态): {dialogue_input}")
            elif dialogue_input == '5':
                dialogue_input = "明天早上7点提醒我开会|user|好的，已为您设置明天7:00的提醒。|system|谢谢|user"
                print(f"使用示例5 (提醒设置): {dialogue_input}")
            elif dialogue_input == '6':
                dialogue_input = "这是什么东西|user|我来识别一下...这是一个遥控器。|system|能详细介绍一下吗|user"
                print(f"使用示例6 (视觉识别): {dialogue_input}")
            elif dialogue_input == '7':
                dialogue_input = "去厨房帮我拿遥控器|user|好的，正在前往厨房。|system|快点|user"
                print(f"使用示例7 (空间操作): {dialogue_input}")
            
            # 预测
            result = predictor.predict(dialogue_input)
            
            # 显示结果
            print(predictor.format_result(dialogue_input, result))
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            print("请检查输入格式是否正确\n")


def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 解析参数
    parser = argparse.ArgumentParser(description='对话意图分类推理')
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
        '--dialogue',
        type=str,
        default=None,
        help='对话输入 (格式: utterance1|speaker1|utterance2|speaker2|...)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='进入交互式模式'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='设备 (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # 创建预测器
    print("\n正在加载模型...")
    predictor = DialogueClassifierPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # 交互式模式
    if args.interactive:
        interactive_mode(predictor)
    # 单次预测
    elif args.dialogue:
        result = predictor.predict(args.dialogue)
        print(predictor.format_result(args.dialogue, result))
    # 默认进入交互模式
    else:
        interactive_mode(predictor)


if __name__ == '__main__':
    main()

