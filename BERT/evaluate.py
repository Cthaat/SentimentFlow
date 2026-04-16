"""验证评估模块。"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import MAX_LEN
from .dataset import CsvStreamDataset
from .text_processing import get_tokenizer


def bert_collate_fn(batch, max_len: int = MAX_LEN):
    """批量 tokenize 的 collate 函数 - 模块级定义以支持多进程序列化。
    
    Args:
        batch: List of (text, label) tuples from Dataset
        max_len: Maximum token sequence length
    
    Returns:
        Dict with input_ids, attention_mask, labels as PyTorch tensors
    """
    tokenizer = get_tokenizer()
    texts, labels = zip(*batch)

    encoded = tokenizer(
        list(texts),
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
    eval_dataset = CsvStreamDataset(split, chunk_size=batch_size * 8, label_map=label_map)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=bert_collate_fn,
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
