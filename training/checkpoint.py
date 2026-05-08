"""Checkpoint 读写模块。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from sentiment_scale import NUM_SENTIMENT_CLASSES

from .model import SentimentLSTMModel


def save_checkpoint(
    checkpoint_path: str,
    model,
    max_len: int,
    vocab_size: int,
    best_val_f1: float,
    best_epoch: int,
    metrics=None,
) -> None:
    """保存模型参数和关键元信息。"""
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = round(best_val_f1, 6)

    metric_payload = {}
    if metrics is not None:
        metric_payload = {
            "best_val_weighted_f1": round(float(metrics.weighted_f1), 6),
            "best_val_mae": round(float(metrics.mae), 6),
            "best_val_qwk": round(float(metrics.quadratic_weighted_kappa), 6),
            "confusion_matrix": metrics.confusion_matrix,
            "support": metrics.support,
        }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "max_len": max_len,
            "vocab_size": vocab_size,
            "num_classes": NUM_SENTIMENT_CLASSES,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            **metric_payload,
        },
        checkpoint_file,
    )

    # 同时保存轻量元信息文件，供模型管理 API 快速读取
    meta_path = checkpoint_file.parent / "training_meta.json"
    meta_path.write_text(
        json.dumps({
            "max_len": max_len,
            "vocab_size": vocab_size,
            "num_classes": NUM_SENTIMENT_CLASSES,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            **metric_payload,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_checkpoint(checkpoint_path: str, device: torch.device, default_vocab_size: int):
    """加载 checkpoint；不存在时返回 None。"""
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        return None

    checkpoint = torch.load(checkpoint_file, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    num_classes = int(checkpoint.get("num_classes") or _infer_num_classes(state_dict))
    model = SentimentLSTMModel(
        checkpoint.get("vocab_size", default_vocab_size),
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {checkpoint_file}")
    return model


def _infer_num_classes(state_dict: dict) -> int:
    weight = state_dict.get("fc.weight")
    if weight is not None and hasattr(weight, "shape"):
        return int(weight.shape[0])
    return NUM_SENTIMENT_CLASSES
