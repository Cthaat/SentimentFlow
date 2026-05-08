"""验证评估模块。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sentiment_scale import NUM_SENTIMENT_CLASSES, compute_classification_metrics

from .dataset import CsvStreamDataset


@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    mae: float
    quadratic_weighted_kappa: float
    confusion_matrix: list[list[int]]
    support: list[int]
    per_class_f1: list[float]

    def __iter__(self):
        """Backward-compatible unpacking as (accuracy, macro_f1)."""
        yield self.accuracy
        yield self.macro_f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    split,
    device: torch.device,
    batch_size: int,
    max_len: int,
    vocab_size: int,
    label_map: dict | None = None,
) -> EvaluationMetrics:
    """在验证集上计算多分类和序数评分指标。"""
    eval_loader = DataLoader(
        CsvStreamDataset(
            split,
            chunk_size=batch_size * 8,
            max_len=max_len,
            vocab_size=vocab_size,
            label_map=label_map,
            labels_are_normalized=True,
        ),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model.eval()
    true_labels: list[int] = []
    pred_labels: list[int] = []
    for batch_x, batch_y in eval_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        logits = model(batch_x)
        pred = torch.argmax(logits, dim=1)

        true_labels.extend(batch_y.detach().cpu().tolist())
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
        quadratic_weighted_kappa=metrics["quadratic_weighted_kappa"],
        confusion_matrix=metrics["confusion_matrix"],
        support=metrics["support"],
        per_class_f1=metrics["per_class_f1"],
    )
