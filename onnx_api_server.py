#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX 推理 API 服务
使用 FastAPI 封装，模型启动时加载一次，支持批量推理和耗时统计
"""

import time
import numpy as np
import onnxruntime as ort
from transformers import BertTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

from src import load_config

# 意图标签
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


class Turn(BaseModel):
    """单轮对话"""
    speaker: str  # "user" or "system"
    text: str


class PredictRequest(BaseModel):
    """预测请求"""
    turns: List[Turn]
    return_all_probs: bool = False  # 是否返回所有类别概率


class PredictResponse(BaseModel):
    """预测响应"""
    intent_id: int
    intent_name: str
    confidence: float
    inference_time_ms: float
    probabilities: Optional[Dict[str, float]] = None


class ONNXPredictor:
    """ONNX 推理器（单例）"""
    
    def __init__(self, model_path: str, config_path: str, use_gpu: bool = False):
        print(f"正在加载模型: {model_path}")
        start_time = time.time()
        
        # 加载配置
        self.config = load_config(config_path)
        self.max_seq_length = self.config["data"]["max_seq_length"]
        
        # 加载 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            self.config["model"]["encoder_name"]
        )
        
        # 加载 ONNX 模型
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # 预热（避免首次推理慢）
        self._warmup()
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}秒")
        print(f"执行提供者: {self.session.get_providers()}")
        print(f"最大序列长度: {self.max_seq_length}")
    
    def _warmup(self, num_warmup: int = 3):
        """预热模型"""
        dummy_turns = [{"speaker": "user", "text": "测试"}]
        for _ in range(num_warmup):
            self._predict(dummy_turns)
    
    def _encode_dialogue(self, turns: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
        """编码对话"""
        # 拼接对话
        parts = []
        for turn in turns:
            speaker = turn.get("speaker", "user").upper()
            text = turn.get("text", "")
            parts.append(f"[{speaker}] {text}")
        dialogue = " ".join(parts)
        
        # Tokenize
        encoded = self.tokenizer(
            dialogue,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )
        
        # 转换数据类型
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        token_type_ids = encoded.get("token_type_ids")
        
        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
        else:
            token_type_ids = token_type_ids.astype(np.int64)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax 函数"""
        e = np.exp(x - np.max(x))
        return e / e.sum()
    
    def _predict(self, turns: List[Dict[str, str]]) -> Dict:
        """执行预测"""
        # 编码
        inputs = self._encode_dialogue(turns)
        
        # 推理
        start_time = time.time()
        logits = self.session.run(["logits"], inputs)[0][0]
        inference_time = (time.time() - start_time) * 1000  # 转为毫秒
        
        # 计算概率
        probs = self._softmax(logits)
        pred_id = int(np.argmax(probs))
        
        return {
            "intent_id": pred_id,
            "intent_name": INTENT_LABELS.get(pred_id, f"Unknown_{pred_id}"),
            "confidence": float(probs[pred_id]),
            "inference_time_ms": inference_time,
            "probabilities": {
                INTENT_LABELS[i]: float(p) for i, p in enumerate(probs)
            }
        }
    
    def predict(self, turns: List[Turn], return_all_probs: bool = False) -> Dict:
        """对外预测接口"""
        turns_dict = [{"speaker": t.speaker, "text": t.text} for t in turns]
        result = self._predict(turns_dict)
        
        if not return_all_probs:
            result.pop("probabilities", None)
        
        return result


# 全局预测器实例
predictor: Optional[ONNXPredictor] = None


# 创建 FastAPI 应用
app = FastAPI(
    title="意图识别 ONNX 推理服务",
    description="基于 ONNX Runtime 的多轮对话意图分类 API",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    global predictor
    predictor = ONNXPredictor(
        model_path="checkpoints/teacher/best_model.onnx",
        config_path="config/teacher_config.yaml",
        use_gpu=False  # 改为 True 启用 GPU
    )


@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "ok",
        "service": "意图识别 ONNX 推理服务",
        "model_loaded": predictor is not None
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    预测对话意图
    
    - **turns**: 对话轮次列表，每轮包含 speaker 和 text
    - **return_all_probs**: 是否返回所有类别的概率分布
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    if not request.turns:
        raise HTTPException(status_code=400, detail="对话不能为空")
    
    try:
        result = predictor.predict(request.turns, request.return_all_probs)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@app.get("/intents")
async def get_intents():
    """获取所有意图类别"""
    return {"intents": INTENT_LABELS}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8030,
        log_level="info"
    )

