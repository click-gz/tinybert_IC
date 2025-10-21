# 教师模型 ONNX 部署指南

本文档说明如何将已经训练完成的教师意图分类模型导出为 ONNX，并使用 ONNX Runtime 在生产环境中进行推理部署。

---

## 1. 环境准备

- Python 3.8 及以上
- 与训练阶段一致的依赖（`torch`、`transformers`、`pyyaml` 等）
- `onnxruntime` 或 `onnxruntime-gpu`

推荐在新的虚拟环境中安装：

```bash
pip install onnxruntime transformers torch pyyaml
```

如需 GPU 推理，请安装 `onnxruntime-gpu` 对应版本，并确保 CUDA 驱动匹配。

---

## 2. 导出 ONNX 模型

项目根目录下提供了脚本 `tinybert_IC/export_to_onnx.py`。

### 2.1 脚本参数

| 参数         | 默认值                                   | 说明                         |
| ------------ | ---------------------------------------- | ---------------------------- |
| `--checkpoint` | `checkpoints/teacher/best_model.pt`       | 训练好的 PyTorch 模型路径    |
| `--config`     | `tinybert_IC/config/teacher_config.yaml`  | 配置文件路径                 |
| `--output`     | `checkpoints/teacher/best_model.onnx`     | ONNX 模型输出路径            |
| `--opset`      | `13`                                      | 导出使用的 ONNX opset 版本   |
| `--device`     | `None`（配置文件中设置）                   | 导出时使用的设备（cuda/cpu） |
| `--use-fp16`   | `False`                                   | 是否转换成 FP16（需 GPU）    |

### 2.2 导出示例

```bash
cd /home/guzhen/tinybert
python tinybert_IC/export_to_onnx.py \
  --checkpoint checkpoints/teacher/best_model.pt \
  --config tinybert_IC/config/teacher_config.yaml \
  --output checkpoints/teacher/best_model.onnx \
  --opset 13 \
  --device cpu
```

执行完成后，将在 `checkpoints/teacher/` 目录下生成 `best_model.onnx` 文件。

> **注意**：如果使用 `--use-fp16`，需在 GPU 上执行，并安装 GPU 版 ONNX Runtime。

---

## 3. ONNX Runtime 推理示例

以下示例展示如何加载导出的 ONNX 模型并完成单条对话的意图预测。

```python
import onnxruntime as ort
from transformers import BertTokenizer
import numpy as np
import json

from tinybert_IC.src import load_config

config = load_config("tinybert_IC/config/teacher_config.yaml")
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
    {"speaker": "user", "text": "今天天气怎么样"},
    {"speaker": "system", "text": "今天晴，最高30度"},
    {"speaker": "user", "text": "需要带伞吗"}
]

result = predict_intent(dialogue)
print(json.dumps(result, ensure_ascii=False, indent=2))
```

---

## 4. 服务化部署建议

1. **API 封装**：
   - 使用 FastAPI/Flask，将 `encode_dialogue` 和 `predict_intent` 封装成 HTTP 接口。
   - 在应用启动时加载一次 `InferenceSession` 和 tokenizer。

2. **容器 & 调度**：
   - 如需容器化，编写 `Dockerfile` 并在启动脚本中运行 Web 服务（例如 `uvicorn`）。
   - 根据业务需求配置副本数、负载均衡、健康检查等。

3. **性能优化**：
   - 使用批量预测（一次传入多条对话）提升吞吐。
   - 如有 GPU，可尝试 `onnxruntime-gpu` 并启用 `CUDAExecutionProvider`。
   - 若模型较大，可考虑蒸馏、量化等进一步压缩。

---

## 5. 版本管理与验证

- 导出后建议运行一组验证样例，确保 ONNX 推理结果与原 PyTorch 模型一致。
- 记录模型版本、导出时间、使用的 `opset`、配置文件等元信息，便于上线回溯。

---

如需进一步示例（例如批量预测、FastAPI 服务模板、Dockerfile）请告知。我们可以在现有基础上继续完善部署方案。


