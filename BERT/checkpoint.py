"""Checkpoint 读写模块。"""

from __future__ import annotations

import json
import os
from pathlib import Path

# 抑制 HF Hub 后台线程请求和 safetensors 自动转换；不要禁用 HF_TOKEN。
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch

from sentiment_scale import NUM_SENTIMENT_CLASSES

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
    metrics=None,
) -> None:
    """保存模型、tokenizer 和元信息。"""
    save_dir = Path(checkpoint_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = getattr(model, "module", model)
    model_to_save = getattr(model_to_save, "_orig_mod", model_to_save)
    model_to_save.save_pretrained(save_dir)
    tokenizer = get_tokenizer(model_name)
    tokenizer.save_pretrained(save_dir)

    metric_payload = {}
    if metrics is not None:
        metric_payload = {
            "best_val_weighted_f1": round(float(metrics.weighted_f1), 6),
            "best_val_mae": round(float(metrics.mae), 6),
            "best_val_rmse": round(float(metrics.rmse), 6),
            "best_val_qwk": round(float(metrics.quadratic_weighted_kappa), 6),
            "best_val_spearman": round(float(metrics.spearman), 6),
            "confusion_matrix": metrics.confusion_matrix,
            "support": metrics.support,
        }

    metadata = {
        "max_len": max_len,
        "model_name": model_name,
        "architecture": getattr(model_to_save, "architecture", "sequence"),
        "num_labels": NUM_SENTIMENT_CLASSES,
        "best_val_f1": round(best_val_f1, 6),
        "best_epoch": best_epoch,
        **metric_payload,
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

    model_name = str(save_dir)
    architecture = "sequence"
    loaded_max_len = MAX_LEN
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_name = str(meta.get("model_name") or save_dir)
        architecture = str(meta.get("architecture", "sequence"))
        loaded_max_len = int(meta.get("max_len", MAX_LEN))
    except Exception:
        pass

    state_path = save_dir / "sentiment_model_state.pt"
    if architecture in {"hybrid", "multiclass"} and state_path.exists():
        model = SentimentBertModel(
            model_name=str(save_dir),
            num_labels=NUM_SENTIMENT_CLASSES,
            architecture="multiclass",
        ).to(device)
        state_dict = torch.load(state_path, map_location=device)
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                "Loaded multiclass checkpoint with compatible subset: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
    else:
        model = SentimentBertModel(
            model_name=str(save_dir),
            num_labels=NUM_SENTIMENT_CLASSES,
            architecture="sequence",
        ).to(device)
    model.eval()

    os.environ["BERT_MODEL_MAX_LEN"] = str(loaded_max_len)
    os.environ["BERT_MODEL_NAME"] = str(save_dir)
    print(f"Loaded BERT model from {save_dir}")
    return model
