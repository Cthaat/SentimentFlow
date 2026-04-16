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
    vocab_size: int,
    label_map: dict | None = None,
) -> Tuple[float, float]:
    """在验证集上计算 Accuracy 和 F1。"""
    eval_loader = DataLoader(
        CsvStreamDataset(split, chunk_size=batch_size * 8, max_len=max_len, vocab_size=vocab_size, label_map=None),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model.eval()
    tp = fp = fn = tn = 0
    for batch_x, batch_y in eval_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        logits = model(batch_x)
        pred = torch.argmax(logits, dim=1)

        tp += int(((pred == 1) & (batch_y == 1)).sum().item())
        tn += int(((pred == 0) & (batch_y == 0)).sum().item())
        fp += int(((pred == 1) & (batch_y == 0)).sum().item())
        fn += int(((pred == 0) & (batch_y == 1)).sum().item())

    total = max(1, tp + tn + fp + fn)
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return accuracy, f1
