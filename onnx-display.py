import onnxruntime as ort
from transformers import BertTokenizer
import numpy as np
import json

from src import load_config

config = load_config("config/teacher_config.yaml")
tokenizer = BertTokenizer.from_pretrained(config["model"]["encoder_name"])
max_len = config["data"]["max_seq_length"]

session = ort.InferenceSession(
    "checkpoints/teacher/best_model.onnx",
    providers=["CPUExecutionProvider"]
)

def encode_dialogue(turns):
    parts = []
    for turn in turns:
        speaker = turn.get("speaker", "user").upper()
        text = turn.get("text", "")
        parts.append(f"[{speaker}] {text}")
    dialogue = " ".join(parts)

    encoded = tokenizer(
        dialogue,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np"  # transformers>=4.29 支持
    )

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

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

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

def predict_intent(turns):
    inputs = encode_dialogue(turns)
    logits = session.run(["logits"], inputs)[0][0]
    probs = softmax(logits)
    pred_id = int(np.argmax(probs))
    return {
        "intent_id": pred_id,
        "intent_name": INTENT_LABELS.get(pred_id, f"Unknown_{pred_id}"),
        "confidence": float(probs[pred_id]),
        "probabilities": {INTENT_LABELS[i]: float(p) for i, p in enumerate(probs)}
    }

# 示例对话
dialogue = [
    {"speaker": "user", "text": "我们有空一起去看风景好不好"}
]

result = predict_intent(dialogue)
print(json.dumps(result, ensure_ascii=False, indent=2))