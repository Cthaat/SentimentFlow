"""训练/加载调度模块。"""

from __future__ import annotations

import os

import torch

from .checkpoint import load_checkpoint as _load_checkpoint
from .config import CHECKPOINT_PATH, VOCAB_SIZE
from .trainer import train_model


def load_checkpoint(device: torch.device):
    """按默认 checkpoint 路径加载模型。"""
    return _load_checkpoint(CHECKPOINT_PATH, device=device, default_vocab_size=VOCAB_SIZE)


def load_or_train():
    """优先加载已有模型；加载失败时自动训练。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FORCE_RETRAIN 环境变量示例：1（表示强制重新训练模型并覆盖现有 checkpoint）
    force_retrain = os.getenv("FORCE_RETRAIN", "0") == "1"

    if not force_retrain:
        model = load_checkpoint(device)
        if model is not None:
            return model, device

    return train_model()
