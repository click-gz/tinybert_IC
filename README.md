# TinyBERT 多轮对话意图识别

基于TinyBERT的多轮对话意图分类系统，支持2-4轮对话，12个意图类别。

## 项目结构

```
TinyBert_IC/
├── src/                      # 源代码模块
│   ├── __init__.py          # 模块初始化
│   ├── model.py             # 模型架构定义
│   ├── dataset.py           # 数据集类
│   ├── trainer.py           # 训练器类
│   └── utils.py             # 工具函数
├── config/                   # 配置文件
│   └── teacher_config.yaml  # 教师模型配置
├── data/                     # 数据目录
│   ├── train.json           # 训练集 (86,278样本)
│   ├── dev.json             # 验证集 (10,974样本)
│   └── test.json            # 测试集 (11,034样本)
├── train_teacher.py         # 教师模型训练脚本
├── requirements.txt         # 项目依赖
├── 方案.md                  # 详细技术方案文档
└── README.md               # 本文件
```

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
conda create -n tinybert python=3.9
conda activate tinybert

# 安装依赖
pip install -r requirements.txt
```

### 2. 训练教师模型 (BERT-base)

```bash
# 使用默认配置训练
python train_teacher.py

# 或指定配置文件
python train_teacher.py --config config/teacher_config.yaml

# 训练完成后，模型保存在: checkpoints/teacher/best_model.pt
```

### 3. 训练学生模型 (TinyBERT-4L)

**方式一：知识蒸馏训练**（推荐）
```bash
# 从Teacher模型蒸馏到Student
python train_distill.py \
    --teacher_checkpoint checkpoints/teacher/best_model.pt \
    --config config/student_config.yaml
```

**方式二：直接训练**（无蒸馏）
```bash
# 使用真实标签直接训练Student
python train_teacher.py \
    --config config/student_config.yaml
```

### 3. 查看训练日志

```bash
# 启动TensorBoard
tensorboard --logdir runs/teacher

# 在浏览器中打开 http://localhost:6006
```

## 配置说明

编辑 `config/teacher_config.yaml` 来修改训练参数：

```yaml
data:
  train_path: data/train.json       # 训练数据路径
  dev_path: data/dev.json          # 验证数据路径
  test_path: data/test.json        # 测试数据路径
  max_turns: 4                     # 最大轮次数
  max_seq_length: 80               # 每轮最大token数

model:
  encoder_name: bert-base-chinese  # 预训练模型
  num_labels: 12                   # 意图类别数
  dropout: 0.1                     # Dropout概率

training:
  encoder_lr: 2.0e-5              # 编码器学习率
  task_lr: 4.0e-5                 # 任务层学习率
  batch_size: 32                  # 批次大小
  num_epochs: 5                   # 训练轮数
  warmup_ratio: 0.1               # Warmup比例
  weight_decay: 0.01              # 权重衰减
  gradient_clip: 1.0              # 梯度裁剪
  patience: 3                     # 早停patience

logging:
  save_dir: checkpoints/teacher   # 模型保存路径
  tensorboard_dir: runs/teacher   # TensorBoard日志路径
  log_every: 100                  # 每N步记录一次
  eval_every: 1                   # 每N个epoch评估一次

device: cuda                      # 训练设备
seed: 42                         # 随机种子
```

## 数据格式

输入数据为JSON格式，每个样本包含：

```json
{
  "dialogue_id": "d3052",
  "turns": [
    {"speaker": "user", "text": "现在手机电量多少？"},
    {"speaker": "system", "text": "剩余电量还有多少？"},
    {"speaker": "user", "text": "现在手机电量多少？"},
    {"speaker": "system", "text": "剩余电量还有多少？"}
  ],
  "label": 6
}
```

## 模型架构

### Teacher模型 (BERT-base-chinese, ~102M参数)

**序列拼接方式**：
```
输入对话 (变长1-4轮)
    ↓
添加说话人标记并拼接
"[USER] 订机票 [SYSTEM] 好的 [USER] 北京到上海"
    ↓
BERT Encoder (12层, 768维)
    一次性编码整段对话，全局注意力
    ↓
取 [CLS] token 表示
    ↓
Classification Head
    ↓
意图预测 (12类)
```

### Student模型 (TinyBERT-4L, ~14.5M参数)

**与Teacher相同的架构，但使用更小的编码器**：
```
TinyBERT Encoder (4层, 312维)
    ↓
Classification Head
    ↓
意图预测 (12类)
```

**关键特性**：
- **序列拼接**：所有轮次拼接成单一序列
- **Speaker区分**：user/system embedding
- **位置编码**：轮次位置信息
- **多轮融合**：Transformer聚合上下文

## 训练流程

### 阶段1：教师模型训练（当前阶段）

```bash
python train_teacher.py
```

**预期结果**：
- 训练时间：2-3小时（RTX 3090）
- Dev Macro-F1：~0.88-0.92
- 模型大小：~110M参数

**输出文件**：
- `checkpoints/teacher/best_model.pt` - 最佳模型
- `checkpoints/teacher/last_model.pt` - 最后一个epoch模型
- `runs/teacher/` - TensorBoard日志

### 阶段2：知识蒸馏（待实现）

训练TinyBERT学生模型，从教师模型学习知识。

### 阶段3：学生微调（待实现）

在任务上进一步优化学生模型性能。

## 硬件要求

### 最低配置
- GPU: 8GB显存（如GTX 1080）
- RAM: 16GB
- 存储: 5GB

### 推荐配置
- GPU: 10GB+显存（如RTX 3080/3090）
- RAM: 32GB
- 存储: 10GB

### 调整策略（显存不足）
```yaml
# 在 config/teacher_config.yaml 中修改
training:
  batch_size: 16  # 减小batch size
data:
  max_seq_length: 64  # 减少序列长度
  num_workers: 0  # 减少数据加载进程
```

## 监控指标

训练过程中监控：
- **train/loss**: 训练损失
- **train/f1**: 训练集Macro-F1
- **dev/loss**: 验证集损失
- **dev/f1**: 验证集Macro-F1（最重要）
- **train/lr**: 学习率变化

## 常见问题

### Q1: CUDA Out of Memory
**解决方案**：
- 减小 `batch_size`
- 减小 `max_seq_length`
- 设置 `num_workers=0`

### Q2: 训练速度慢
**解决方案**：
- 增加 `num_workers`（数据加载并行）
- 使用更快的GPU
- 减少 `log_every` 频率

### Q3: Dev F1不提升
**检查**：
- 学习率是否过大/过小
- 是否过拟合（train F1 >> dev F1）
- 数据质量是否有问题

## 下一步

1. **当前任务**：训练教师模型
   ```bash
   python train_teacher.py
   ```

2. **知识蒸馏**：训练TinyBERT学生模型（代码待实现）

3. **模型部署**：导出ONNX，部署为API服务

## 参考文献

- TinyBERT论文：[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)
- BERT论文：[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

## License

MIT License

## 联系方式

如有问题，请查阅 `方案.md` 获取详细技术文档。

