"""Checkpoint 读写模块。"""

from __future__ import annotations

import json
import os
from pathlib import Path

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
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = round(best_val_f1, 6)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "max_len": max_len,
            "vocab_size": vocab_size,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
        },
        checkpoint_file,
    )

    # 同时保存轻量元信息文件，供模型管理 API 快速读取
    meta_path = checkpoint_file.parent / "training_meta.json"
    meta_path.write_text(
        json.dumps({
            "max_len": max_len,
            "vocab_size": vocab_size,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_checkpoint(checkpoint_path: str, device: torch.device, default_vocab_size: int):
    """加载 checkpoint；不存在时返回 None。"""
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        return None

    checkpoint = torch.load(checkpoint_file, map_location=device)
    model = SentimentLSTMModel(checkpoint.get("vocab_size", default_vocab_size)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {checkpoint_file}")
    return model
