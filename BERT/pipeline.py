"""训练/加载调度模块。"""

from __future__ import annotations

import os

import torch

from .checkpoint import load_checkpoint as _load_checkpoint
from .config import CHECKPOINT_PATH
from .trainer import train_model


def load_checkpoint(device: torch.device):
    """按默认 checkpoint 路径加载模型。"""
    return _load_checkpoint(CHECKPOINT_PATH, device=device)


def load_or_train():
    """优先加载已有模型；加载失败时自动训练。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    force_retrain = os.getenv("BERT_FORCE_RETRAIN", os.getenv("FORCE_RETRAIN", "0")) == "1"

    if not force_retrain:
        model = load_checkpoint(device)
        if model is not None:
            return model, device

    return train_model()
