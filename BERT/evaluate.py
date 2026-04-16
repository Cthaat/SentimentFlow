"""验证评估模块。"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import CsvStreamDataset


@torch.no_grad()
def evaluate(
    model: nn.Module,
    split,
    device: torch.device,
    batch_size: int,
    max_len: int,
    label_map: dict | None = None,
) -> Tuple[float, float]:
    """在验证集上计算 Accuracy 和 Macro-F1。"""
    eval_loader = DataLoader(
        CsvStreamDataset(split, chunk_size=batch_size * 8, max_len=max_len, label_map=label_map),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model.eval()
    tp = fp = fn = tn = 0
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(logits, dim=1)

        tp += int(((pred == 1) & (labels == 1)).sum().item())
        tn += int(((pred == 0) & (labels == 0)).sum().item())
        fp += int(((pred == 1) & (labels == 0)).sum().item())
        fn += int(((pred == 0) & (labels == 1)).sum().item())

    total = max(1, tp + tn + fp + fn)
    accuracy = (tp + tn) / total

    precision_pos = tp / max(1, tp + fp)
    recall_pos = tp / max(1, tp + fn)
    f1_pos = 2 * precision_pos * recall_pos / max(1e-12, precision_pos + recall_pos)

    precision_neg = tn / max(1, tn + fn)
    recall_neg = tn / max(1, tn + fp)
    f1_neg = 2 * precision_neg * recall_neg / max(1e-12, precision_neg + recall_neg)

    macro_f1 = 0.5 * (f1_pos + f1_neg)
    return accuracy, macro_f1
