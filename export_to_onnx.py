#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export the trained teacher model to ONNX."""

import argparse
import os
import torch
from transformers import BertTokenizer

from src import (
    MultiTurnDialogueClassifier,
    load_config,
    get_device,
)


class OnnxDialogueClassifier(torch.nn.Module):
    """ONNX-friendly wrapper that exposes logits only."""

    def __init__(self, model: MultiTurnDialogueClassifier):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs["logits"]


def parse_args():
    parser = argparse.ArgumentParser(description="导出对话意图分类模型为 ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/teacher/checkpoint_epoch_1.pt",
        help="训练完成的模型检查点路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/teacher_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/teacher/best_model.onnx",
        help="ONNX 模型输出路径",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset 版本 (建议 >=13)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="导出时使用的设备 (cuda/cpu)",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="是否将权重转换为 FP16 后再导出 (需 GPU 支持)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    device_id = args.device or config.get("device", "cpu")
    device = get_device(device_id)

    tokenizer = BertTokenizer.from_pretrained(config["model"]["encoder_name"])

    model = MultiTurnDialogueClassifier(
        encoder_name=config["model"]["encoder_name"],
        num_labels=config["model"]["num_labels"],
        max_seq_length=config["data"]["max_seq_length"],
        dropout=config["model"].get("dropout", 0.1),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dummy_dialogue = "[USER] 测试 [SYSTEM] 测试"
    encoded = tokenizer(
        dummy_dialogue,
        max_length=config["data"]["max_seq_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    if "token_type_ids" in encoded:
        token_type_ids = encoded["token_type_ids"].to(device)
    else:
        token_type_ids = torch.zeros_like(input_ids, device=device)

    # FP16 导出（注意：input_ids 等必须保持 int64）
    if args.use_fp16:
        if device.type != "cuda":
            raise RuntimeError("FP16 导出需要在 GPU 上进行")
        # 注意：不要转换模型，而是使用动态量化或者导出后量化
        print("警告：FP16 导出功能暂不完全支持，建议使用 FP32 导出")

    wrapper = OnnxDialogueClassifier(model)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "token_type_ids": {0: "batch", 1: "seq"},
        "logits": {0: "batch"},
    }

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, token_type_ids),
        args.output,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    print(f"ONNX 模型已导出到: {args.output}")


if __name__ == "__main__":
    main()


