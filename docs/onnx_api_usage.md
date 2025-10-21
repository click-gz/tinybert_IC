# ONNX API 推理服务使用指南

## 概述

将 ONNX 模型封装成 REST API 服务，避免每次推理都重新加载模型，提升推理效率并支持性能测试。

## 架构

- **FastAPI**: Web 框架
- **ONNX Runtime**: 推理引擎
- **Uvicorn**: ASGI 服务器

## 安装依赖

```bash
pip install fastapi uvicorn requests
```

## 启动服务

### 方式一：直接运行

```bash
cd /home/guzhen/tinybert
python tinybert_IC/onnx_api_server.py
```

服务将在 `http://localhost:8000` 启动。

### 方式二：使用 Uvicorn 命令

```bash
cd /home/guzhen/tinybert/tinybert_IC
uvicorn onnx_api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 配置 GPU 推理

编辑 `onnx_api_server.py` 第 138 行：

```python
use_gpu=True  # 启用 GPU
```

确保已安装 `onnxruntime-gpu`。

## API 接口

### 1. 健康检查

```bash
curl http://localhost:8000/
```

响应：
```json
{
  "status": "ok",
  "service": "意图识别 ONNX 推理服务",
  "model_loaded": true
}
```

### 2. 意图预测

**端点**: `POST /predict`

**请求体**:
```json
{
  "turns": [
    {"speaker": "user", "text": "我们有空一起去看风景好不好"}
  ],
  "return_all_probs": false
}
```

**响应**:
```json
{
  "intent_id": 1,
  "intent_name": "闲聊",
  "confidence": 0.9523,
  "inference_time_ms": 12.34
}
```

如果 `return_all_probs=true`，还会返回：
```json
{
  "probabilities": {
    "问题求助/技术支持": 0.0012,
    "闲聊": 0.9523,
    "天气查询": 0.0231,
    ...
  }
}
```

### 3. 获取意图列表

```bash
curl http://localhost:8000/intents
```

## 性能测试

运行性能测试脚本：

```bash
cd /home/guzhen/tinybert
python tinybert_IC/test_onnx_api.py
```

测试内容：
1. **单次推理测试**: 验证基本功能和单次耗时
2. **批量测试**: 多个测试用例各运行 20 次，统计平均耗时、标准差
3. **并发测试**: 10 个并发请求，测试 QPS

## 典型性能指标

### CPU 推理 (示例)
- 平均推理时间: 15-30ms (取决于 `max_seq_length`)
- 总耗时(含网络): 20-40ms
- QPS: 约 25-50 req/s (单实例)

### GPU 推理 (示例)
- 平均推理时间: 3-8ms
- 总耗时(含网络): 8-15ms
- QPS: 约 60-120 req/s (单实例)

> 注：实际性能受硬件、模型大小、序列长度影响。

## 优化建议

### 1. 减小 `max_seq_length`
如果对话通常较短，可在 `teacher_config.yaml` 中将 `max_seq_length` 从 512 降至 256 或更小，可显著加速。

### 2. 使用 GPU
安装 `onnxruntime-gpu` 并在代码中启用 GPU。

### 3. 批量推理
如果有多条对话需要预测，可修改代码支持批量输入。

### 4. 模型量化
导出 ONNX 时使用 FP16（需 GPU）或 INT8 量化（需额外工具）。

### 5. 多进程/多实例
使用 Gunicorn 等多进程服务器：
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker tinybert_IC.onnx_api_server:app
```

## 部署到生产环境

### Docker 部署

创建 `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY tinybert_IC/ /app/tinybert_IC/
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "tinybert_IC/onnx_api_server.py"]
```

构建和运行：
```bash
docker build -t intent-classifier .
docker run -p 8000:8000 intent-classifier
```

### Kubernetes 部署

可配置 Deployment、Service 和 HPA（水平扩展）。

## Python 客户端示例

```python
import requests

def predict_intent(turns):
    response = requests.post(
        "http://localhost:8000/predict",
        json={"turns": turns}
    )
    return response.json()

# 使用
result = predict_intent([
    {"speaker": "user", "text": "播放周杰伦的歌"}
])
print(f"意图: {result['intent_name']}, 置信度: {result['confidence']:.3f}")
```

## 监控与日志

- 访问 `http://localhost:8000/docs` 查看自动生成的 API 文档（Swagger UI）
- 使用 Prometheus + Grafana 监控 QPS、延迟等指标
- 配置日志记录推理结果和异常

## 常见问题

**Q: 首次请求很慢？**  
A: 模型会在启动时预热 3 次，首次可能稍慢，后续正常。

**Q: 如何支持更高并发？**  
A: 使用多进程部署（Gunicorn）或部署多实例 + 负载均衡。

**Q: 推理时间为什么比 PyTorch 快？**  
A: ONNX Runtime 对模型做了优化，且避免了 PyTorch 的动态图开销。

**Q: 能否同时支持多个模型？**  
A: 可以在启动时加载多个 `ONNXPredictor` 实例，为不同端点服务。

