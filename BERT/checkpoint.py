"""Checkpoint 读写模块。"""

from __future__ import annotations

import json
import os
from pathlib import Path

# 抑制 HF Hub 后台线程请求和 safetensors 自动转换
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

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
    """加载 checkpoint；不存在或无效时返回 None。

    仅接受包含 training_meta.json 的微调 checkpoint，
    以避免加载仅含基础预训练权重的目录（无分类器头）。
    """
    save_dir = Path(checkpoint_path)
    if not save_dir.exists() or not (save_dir / "config.json").exists():
        return None

    meta_path = save_dir / "training_meta.json"
    if not meta_path.exists():
        print(
            f"Skipping {save_dir}: not a fine-tuned checkpoint"
            f" (missing training_meta.json —"
            f" classifier weights have not been trained)"
        )
        return None

    model_name = BERT_MODEL_NAME
    loaded_max_len = MAX_LEN
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
