"""训练/加载调度模块。"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .checkpoint import load_checkpoint as _load_checkpoint
from .config import get_checkpoint_path
from .trainer import train_model


def load_checkpoint(device: torch.device):
    """按默认 checkpoint 路径加载模型。"""
    return _load_checkpoint(get_checkpoint_path(), device=device)


def load_or_train():
    """优先加载已有模型；加载失败时自动训练。

    若 checkpoint 路径存在但包含无效 checkpoint（缺少 training_meta.json），
    则抛出 FileNotFoundError 而非自动训练，以避免用未微调的基础模型做预测。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    force_retrain = os.getenv("BERT_FORCE_RETRAIN", os.getenv("FORCE_RETRAIN", "0")) == "1"

    if not force_retrain:
        model = load_checkpoint(device)
        if model is not None:
            return model, device

    checkpoint_path = get_checkpoint_path()
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    if ckpt_path and ckpt_path.exists():
        raise FileNotFoundError(
            f"BERT checkpoint at {ckpt_path} is not a valid fine-tuned model"
            f" (missing training_meta.json)."
            f" Please delete this directory and train a new BERT model."
        )

    return train_model()
