#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX API 性能测试脚本 - 包含性能测试和准确率评估
"""

import requests
import time
import json
import numpy as np
from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os

API_URL = "http://localhost:8030"

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


def test_single_request(turns: List[Dict[str, str]], verbose: bool = True):
    """测试单次请求"""
    start_time = time.time()
    
    response = requests.post(
        f"{API_URL}/predict",
        json={"turns": turns, "return_all_probs": False}
    )
    
    total_time = (time.time() - start_time) * 1000  # 毫秒
    
    if response.status_code == 200:
        result = response.json()
        if verbose:
            print(f"\n预测结果:")
            print(f"  意图: {result['intent_name']} (ID: {result['intent_id']})")
            print(f"  置信度: {result['confidence']:.4f}")
            print(f"  推理耗时: {result['inference_time_ms']:.2f}ms")
            print(f"  总耗时(含网络): {total_time:.2f}ms")
        return {
            "success": True,
            "inference_time": result['inference_time_ms'],
            "total_time": total_time,
            "result": result
        }
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)
        return {"success": False}


def test_batch_requests(test_cases: List[List[Dict[str, str]]], num_runs: int = 10):
    """批量测试多个用例"""
    print(f"\n{'='*70}")
    print(f"批量性能测试 - {len(test_cases)} 个测试用例，每个运行 {num_runs} 次")
    print(f"{'='*70}\n")
    
    all_inference_times = []
    all_total_times = []
    
    for idx, turns in enumerate(test_cases, 1):
        inference_times = []
        total_times = []
        
        print(f"测试用例 {idx}: {turns[0]['text'][:40]}...")
        
        # 预热
        test_single_request(turns, verbose=False)
        
        # 正式测试
        for _ in range(num_runs):
            result = test_single_request(turns, verbose=False)
            if result["success"]:
                inference_times.append(result["inference_time"])
                total_times.append(result["total_time"])
        
        if inference_times:
            print(f"  推理耗时: {np.mean(inference_times):.2f}ms (±{np.std(inference_times):.2f})")
            print(f"  总耗时:   {np.mean(total_times):.2f}ms (±{np.std(total_times):.2f})")
            print(f"  预测结果: {result['result']['intent_name']} ({result['result']['confidence']:.3f})")
            
            all_inference_times.extend(inference_times)
            all_total_times.extend(total_times)
    
    # 汇总统计
    if all_inference_times:
        print(f"\n{'='*70}")
        print("总体性能统计")
        print(f"{'='*70}")
        print(f"推理耗时:")
        print(f"  平均: {np.mean(all_inference_times):.2f}ms")
        print(f"  中位数: {np.median(all_inference_times):.2f}ms")
        print(f"  最小: {np.min(all_inference_times):.2f}ms")
        print(f"  最大: {np.max(all_inference_times):.2f}ms")
        print(f"  标准差: {np.std(all_inference_times):.2f}ms")
        
        print(f"\n总耗时(含网络):")
        print(f"  平均: {np.mean(all_total_times):.2f}ms")
        print(f"  中位数: {np.median(all_total_times):.2f}ms")
        print(f"  QPS估算: {1000 / np.mean(all_total_times):.2f} req/s")
        print(f"{'='*70}\n")


def test_concurrent_requests(turns: List[Dict[str, str]], num_concurrent: int = 10):
    """并发测试"""
    import concurrent.futures
    
    print(f"\n{'='*70}")
    print(f"并发测试 - {num_concurrent} 个并发请求")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [
            executor.submit(test_single_request, turns, False)
            for _ in range(num_concurrent)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    total_time = time.time() - start_time
    
    successful = [r for r in results if r.get("success")]
    if successful:
        inference_times = [r["inference_time"] for r in successful]
        print(f"成功请求: {len(successful)}/{num_concurrent}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均推理时间: {np.mean(inference_times):.2f}ms")
        print(f"并发QPS: {len(successful) / total_time:.2f} req/s")
    print(f"{'='*70}\n")


def test_accuracy_on_dataset(test_data_path: str = "data/test.json", max_samples: int = None):
    """在测试集上评估准确率"""
    print(f"\n{'='*70}")
    print("准确率评估 - 在测试集上评估模型性能")
    print(f"{'='*70}\n")
    
    # 加载测试数据
    if not os.path.exists(test_data_path):
        print(f"测试集文件不存在: {test_data_path}")
        return
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    print(f"测试集样本数: {len(test_data)}")
    
    # 批量预测
    y_true = []
    y_pred = []
    inference_times = []
    failed_count = 0
    
    print("正在预测...")
    for i, item in enumerate(test_data):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(test_data)}")
        
        turns = [{"speaker": t["speaker"], "text": t["text"]} for t in item["turns"]]
        true_label = item["label"]
        
        result = test_single_request(turns, verbose=False)
        
        if result["success"]:
            y_true.append(true_label)
            y_pred.append(result["result"]["intent_id"])
            inference_times.append(result["inference_time"])
        else:
            failed_count += 1
    
    if failed_count > 0:
        print(f"\n警告: {failed_count} 个样本预测失败")
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # 每类指标
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 输出结果
    print(f"\n{'='*70}")
    print("评估结果")
    print(f"{'='*70}")
    print(f"准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"宏平均精确率 (Precision): {precision:.4f}")
    print(f"宏平均召回率 (Recall): {recall:.4f}")
    print(f"宏平均 F1: {f1:.4f}")
    print(f"平均推理时间: {np.mean(inference_times):.2f}ms")
    
    # 每类详细结果
    print(f"\n{'='*70}")
    print("各类别详细指标")
    print(f"{'='*70}")
    print(f"{'类别':<20} {'支持数':>8} {'精确率':>8} {'召回率':>8} {'F1':>8}")
    print("-" * 70)
    for i in range(12):
        label_name = INTENT_LABELS.get(i, f"Label_{i}")
        print(f"{label_name:<20} {int(support[i]):>8} {precision_per_class[i]:>8.4f} "
              f"{recall_per_class[i]:>8.4f} {f1_per_class[i]:>8.4f}")
    
    # 找出误分类样本
    misclassified = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            misclassified.append({
                "index": i,
                "true_label": INTENT_LABELS[true],
                "pred_label": INTENT_LABELS[pred],
                "dialogue": test_data[i]["turns"]
            })
    
    if misclassified:
        print(f"\n{'='*70}")
        print(f"误分类样本 (共 {len(misclassified)} 个)")
        print(f"{'='*70}")
        for item in misclassified[:10]:  # 只显示前10个
            user_turns = [t["text"] for t in item["dialogue"] if t["speaker"] == "user"]
            print(f"\n样本 {item['index']}:")
            print(f"  对话: {' | '.join(user_turns)[:80]}")
            print(f"  真实标签: {item['true_label']}")
            print(f"  预测标签: {item['pred_label']}")
        
        if len(misclassified) > 10:
            print(f"\n... 还有 {len(misclassified) - 10} 个误分类样本未显示")
    else:
        print(f"\n🎉 所有样本预测正确！")
    
    # 混淆矩阵
    print(f"\n{'='*70}")
    print("混淆矩阵")
    print(f"{'='*70}")
    print("行: 真实标签, 列: 预测标签")
    print(cm)
    
    print(f"{'='*70}\n")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "misclassified_count": len(misclassified)
    }


def main():
    # 检查服务是否可用
    try:
        response = requests.get(f"{API_URL}/")
        print(f"服务状态: {response.json()}")
    except Exception as e:
        print(f"无法连接到服务: {e}")
        print(f"请先启动服务: python tinybert_IC/onnx_api_server.py")
        return
    
    # 测试用例
    test_cases = [
        # 闲聊 - 社交邀约
        [{"speaker": "user", "text": "我们有空一起去看风景好不好"}],
        
        # 天气查询
        [
            {"speaker": "user", "text": "今天天气怎么样"},
            {"speaker": "system", "text": "今天晴，最高30度"},
            {"speaker": "user", "text": "需要带伞吗"}
        ],
        
        # 音乐播放
        [{"speaker": "user", "text": "播放周杰伦的歌"}],
        
        # 空间操作
        [{"speaker": "user", "text": "去厨房帮我拿遥控器"}],
        
        # 问题求助
        [{"speaker": "user", "text": "我的WiFi连不上了怎么办"}],
        
        # 提醒设置
        [{"speaker": "user", "text": "明天早上7点提醒我开会"}],
    ]
    
    # 单次测试
    print("\n" + "="*70)
    print("单次推理测试")
    print("="*70)
    test_single_request(test_cases[0])
    
    # 批量测试
    test_batch_requests(test_cases, num_runs=20)
    
    # 并发测试
    test_concurrent_requests(test_cases[0], num_concurrent=10)
    
    # 准确率评估
    print("\n" + "="*70)
    print("开始准确率评估（使用完整测试集）")
    print("="*70)
    test_accuracy_on_dataset("data/test.json")


if __name__ == "__main__":
    main()

