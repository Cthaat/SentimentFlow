"""训练主流程模块。"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import (
    BERT_MODEL_NAME,
    CHECKPOINT_PATH,
    EPOCHS,
    MAX_LEN,
    get_runtime_settings,
)
from .data_sources import build_train_split_and_val_split, get_label_distribution
from .dataset import CsvStreamDataset
from .evaluate import evaluate
from .model import SentimentBertModel
from .text_processing import get_tokenizer


def bert_collate_fn(batch, max_len: int = MAX_LEN):
    """批量 tokenize 的 collate 函数 - 模块级定义以支持多进程序列化。
    
    关键优化：在 DataLoader batch 级别进行批量 tokenize，
    而不是在 Dataset 中逐条 encode。这样 tokenizer 一次性处理
    整个 batch，性能提升 100 倍以上。
    
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


def _build_train_loader(train_split, settings, device: torch.device, label_map: dict | None = None):
    dataset = CsvStreamDataset(
        train_split,
        chunk_size=settings.chunk_size,
        label_map=label_map,
    )
    loader_kwargs = {
        "batch_size": settings.batch_size,
        "num_workers": max(0, settings.num_workers),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "collate_fn": bert_collate_fn,
    }
    if settings.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1
        loader_kwargs["persistent_workers"] = True

    return DataLoader(dataset, **loader_kwargs)


def _build_loss_fn(neg_count: int, pos_count: int, total_count: int, settings, device: torch.device):
    if settings.use_weighted_loss and neg_count > 0 and pos_count > 0:
        class_weights = torch.tensor(
            [total_count / (2 * neg_count), total_count / (2 * pos_count)],
            dtype=torch.float32,
            device=device,
        )
        print(
            f"Using weighted loss: neg_w={class_weights[0].item():.4f}, "
            f"pos_w={class_weights[1].item():.4f}"
        )
        return nn.CrossEntropyLoss(weight=class_weights)
    return nn.CrossEntropyLoss()


def train_model():
    """执行完整训练，返回最优模型和设备。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using CUDA device for BERT training.")
    else:
        print("CUDA is not available in this environment. BERT training will run on CPU.")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    settings = get_runtime_settings(device.type)
    dataset_names, train_split, val_split, label_map = build_train_split_and_val_split()
    train_size = len(train_split)
    val_size = len(val_split)

    neg_count, pos_count = get_label_distribution(train_split, label_map)
    total_count = max(1, neg_count + pos_count)

    print(f"Datasets: {', '.join(dataset_names)}")
    print("DATA SUMMARY:")
    print(f"  Total train samples (after merge): {train_size:,}")
    print(f"  Total val samples: {val_size:,}")
    print(
        f"Train samples: {total_count}, neg={neg_count}, pos={pos_count}, "
        f"pos_ratio={pos_count / total_count:.4f}"
    )
    print(
        f"Training config: batch_size={settings.batch_size}, "
        f"eval_batch_size={settings.eval_batch_size}, "
        f"grad_accum_steps={settings.grad_accum_steps}, num_workers={settings.num_workers}, "
        f"early_stop_patience={settings.early_stop_patience}, "
        f"early_stop_min_delta={settings.early_stop_min_delta}, model={BERT_MODEL_NAME}"
    )

    loader = _build_train_loader(train_split, settings, device, label_map)

    model = SentimentBertModel(model_name=BERT_MODEL_NAME).to(device)
    loss_fn = _build_loss_fn(neg_count, pos_count, total_count, settings, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
    no_improve_epochs = 0

    model.train()
    for epoch in range(EPOCHS):
        total_loss = torch.zeros((), device=device)
        batch_count = 0
        step = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, start=1):
            batch_count += 1
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels) / settings.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % settings.grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.detach()

        if batch_count > 0 and step % settings.grad_accum_steps != 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        val_acc, val_f1 = evaluate(
            model,
            val_split,
            device,
            batch_size=settings.eval_batch_size,
            max_len=MAX_LEN,
            label_map=label_map,
        )
        avg_loss = (total_loss / max(1, batch_count)).item()
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, ValAcc: {val_acc:.4f}, ValMacroF1: {val_f1:.4f}")

        model.train()

        if val_f1 > best_f1 + settings.early_stop_min_delta:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_state_dict = deepcopy(model.state_dict())
            no_improve_epochs = 0
            save_checkpoint(
                checkpoint_path=CHECKPOINT_PATH,
                model=model,
                max_len=MAX_LEN,
                model_name=BERT_MODEL_NAME,
                best_val_f1=best_f1,
                best_epoch=best_epoch,
            )
            print(f"Best model updated at epoch {epoch + 1}, ValMacroF1={best_f1:.4f}")
        else:
            no_improve_epochs += 1
            print(
                f"No significant improvement for {no_improve_epochs} epoch(s) "
                f"(best={best_f1:.4f}, current={val_f1:.4f}, "
                f"min_delta={settings.early_stop_min_delta})."
            )
            if settings.early_stop_patience > 0 and no_improve_epochs >= settings.early_stop_patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"No improvement for {no_improve_epochs} consecutive epoch(s)."
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best in-memory checkpoint from epoch {best_epoch}, ValMacroF1={best_f1:.4f}")

    print(f"Training finished. Best model saved to {CHECKPOINT_PATH}")
    return model, device
