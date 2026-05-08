"""验证评估模块。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sentiment_scale import NUM_SENTIMENT_CLASSES, compute_classification_metrics

from .config import MAX_LEN
from .config import get_model_name
from .dataset import CsvStreamDataset
from .inference import predicted_classes_from_outputs
from .text_processing import get_tokenizer


@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    mae: float
    rmse: float
    quadratic_weighted_kappa: float
    spearman: float
    confusion_matrix: list[list[int]]
    support: list[int]
    per_class_f1: list[float]

    def __iter__(self):
        """Backward-compatible unpacking as (accuracy, macro_f1)."""
        yield self.accuracy
        yield self.macro_f1


def bert_collate_fn(batch, max_len: int = MAX_LEN):
    """批量 tokenize 的 collate 函数 - 模块级定义以支持多进程序列化。
    
    Args:
        batch: List of (text, label) tuples from Dataset
        max_len: Maximum token sequence length
    
    Returns:
        Dict with input_ids, attention_mask, labels as PyTorch tensors
    """
    tokenizer = get_tokenizer(get_model_name())
    records = [_normalize_batch_record(item) for item in batch]
    texts = [record["text"] for record in records]
    labels = [int(record["label"]) for record in records]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _normalize_batch_record(item) -> dict:
    if isinstance(item, dict):
        return item
    text, label = item
    return {"text": str(text), "label": int(label)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    split,
    device: torch.device,
    batch_size: int,
    max_len: int,
    label_map: dict | None = None,
) -> EvaluationMetrics:
    """在验证集上计算多分类和序数评分指标。"""
    eval_dataset = CsvStreamDataset(
        split,
        chunk_size=batch_size * 8,
        label_map=label_map,
        labels_are_normalized=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=lambda batch: bert_collate_fn(batch, max_len=max_len),
    )

    model.eval()
    true_labels: list[int] = []
    pred_labels: list[int] = []
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pred = predicted_classes_from_outputs(outputs)

        true_labels.extend(labels.detach().cpu().tolist())
        pred_labels.extend(pred.detach().cpu().tolist())

    metrics = compute_classification_metrics(
        true_labels,
        pred_labels,
        num_classes=NUM_SENTIMENT_CLASSES,
    )
    return EvaluationMetrics(
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        quadratic_weighted_kappa=metrics["quadratic_weighted_kappa"],
        spearman=metrics["spearman"],
        confusion_matrix=metrics["confusion_matrix"],
        support=metrics["support"],
        per_class_f1=metrics["per_class_f1"],
    )
