"""Checkpoint 读写模块。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .config import BERT_MODEL_NAME, MAX_LEN
from .model import SentimentBertModel
from .text_processing import get_tokenizer


def save_checkpoint(
    checkpoint_path: str,
    model,
    max_len: int,
    model_name: str,
    best_val_f1: float,
    best_epoch: int,
) -> None:
    """保存模型、tokenizer 和元信息。"""
    save_dir = Path(checkpoint_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.backbone.save_pretrained(save_dir)
    tokenizer = get_tokenizer(model_name)
    tokenizer.save_pretrained(save_dir)

    metadata = {
        "max_len": max_len,
        "model_name": model_name,
        "best_val_f1": round(best_val_f1, 6),
        "best_epoch": best_epoch,
    }
    (save_dir / "training_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """加载 checkpoint；不存在时返回 None。"""
    save_dir = Path(checkpoint_path)
    if not save_dir.exists() or not (save_dir / "config.json").exists():
        return None

    meta_path = save_dir / "training_meta.json"
    model_name = BERT_MODEL_NAME
    loaded_max_len = MAX_LEN
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model_name = str(meta.get("model_name", BERT_MODEL_NAME))
            loaded_max_len = int(meta.get("max_len", MAX_LEN))
        except Exception:
            pass

    model = SentimentBertModel(model_name=model_name).to(device)
    model.backbone = model.backbone.from_pretrained(str(save_dir)).to(device)
    model.eval()

    os.environ["BERT_MODEL_MAX_LEN"] = str(loaded_max_len)
    print(f"Loaded BERT model from {save_dir}")
    return model
