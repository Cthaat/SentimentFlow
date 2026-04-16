"""Checkpoint 读写模块。"""

from __future__ import annotations

import os

import torch

from .model import SentimentLSTMModel


def save_checkpoint(
    checkpoint_path: str,
    model,
    max_len: int,
    vocab_size: int,
    best_val_f1: float,
    best_epoch: int,
) -> None:
    """保存模型参数和关键元信息。"""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "max_len": max_len,
            "vocab_size": vocab_size,
            "best_val_f1": round(best_val_f1, 6),
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )


def load_checkpoint(checkpoint_path: str, device: torch.device, default_vocab_size: int):
    """加载 checkpoint；不存在时返回 None。"""
    if not os.path.exists(checkpoint_path):
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SentimentLSTMModel(checkpoint.get("vocab_size", default_vocab_size)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model
