"""训练主流程模块。

职责边界：
- 设备与性能设置
- DataLoader 构建
- 损失函数与优化器初始化
- epoch 训练循环
- 验证评估与最优模型保存
"""

from __future__ import annotations

import csv
import os
import threading
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import MAX_LEN, VOCAB_SIZE, get_checkpoint_path, get_epochs, get_runtime_settings
from .data_sources import build_train_split_and_val_split, get_label_distribution
from .dataset import CsvStreamDataset
from .evaluate import evaluate
from .model import SentimentLSTMModel


def _materialize_train_split_for_multiprocess(train_split, settings) -> str | Path | object:
    """在 Windows 多进程场景下，把内存数据集落盘为 CSV 以避免 spawn pickle 失败。"""
    if os.name != "nt" or settings.num_workers <= 0:
        return train_split

    if isinstance(train_split, (str, Path)):
        return train_split

    cache_dir = Path(os.getenv("TRAIN_CACHE_DIR", str(Path(__file__).resolve().parent.parent / ".cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = getattr(train_split, "_fingerprint", None) or f"len{len(train_split)}"
    safe_fingerprint = str(fingerprint).replace("/", "_").replace("\\", "_")
    cache_path = cache_dir / f"train_stream_{safe_fingerprint}_{len(train_split)}.csv"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        print(f"Reusing materialized training CSV for multi-worker loading: {cache_path}")
        return cache_path

    print(
        "Materializing in-memory train split to CSV for Windows multi-worker DataLoader: "
        f"{cache_path}"
    )
    with cache_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for row in train_split:
            text = row.get("text") or row.get("review") or row.get("content") or ""
            label = int(row.get("label", 0))
            writer.writerow([str(text), label])

    return cache_path


def _build_train_loader(train_split, settings, device: torch.device, label_map: dict | None = None):
    train_source = _materialize_train_split_for_multiprocess(train_split, settings)
    dataset = CsvStreamDataset(
        train_source,
        chunk_size=settings.chunk_size,
        max_len=MAX_LEN,
        vocab_size=VOCAB_SIZE,
        label_map=label_map,
    )
    loader_kwargs = {
        "batch_size": settings.batch_size,
        "num_workers": max(0, settings.num_workers),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
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


def train_model(cancel_event: threading.Event | None = None):
    """执行完整训练，返回最优模型和设备。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using CUDA device for training.")
    else:
        print("CUDA is not available in this environment. Training will run on CPU.")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    settings = get_runtime_settings(device.type)
    epochs = get_epochs()
    checkpoint_path = get_checkpoint_path()
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
        f"grad_accum_steps={settings.grad_accum_steps}, num_workers={settings.num_workers}, "
        f"epochs={epochs}, "
        f"early_stop_patience={settings.early_stop_patience}, "
        f"early_stop_min_delta={settings.early_stop_min_delta}"
    )

    loader = _build_train_loader(train_split, settings, device, label_map)

    model = SentimentLSTMModel(VOCAB_SIZE).to(device)
    loss_fn = _build_loss_fn(neg_count, pos_count, total_count, settings, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    # 使用 PyTorch 的自动混合精度（AMP）功能来加速训练，特别是在 GPU 上。GradScaler 用于动态调整梯度缩放以避免数值不稳定。
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
    no_improve_epochs = 0

    # 训练循环：每个 epoch 迭代训练数据，计算损失，进行反向传播和优化器更新。每个 epoch 结束后在验证集上评估模型性能，并根据 Macro-F1 指标保存最优模型。
    model.train()
    for epoch in range(epochs):
        if cancel_event is not None and cancel_event.is_set():
            print("Training cancelled before next epoch.")
            break

        total_loss = torch.zeros((), device=device)
        batch_count = 0
        step = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (batch_x, batch_y) in enumerate(loader, start=1):
            if cancel_event is not None and cancel_event.is_set():
                print("Training cancellation requested; stopping current epoch.")
                break

            batch_count += 1
            # 将输入数据移动到设备（GPU 或 CPU），并启用 AMP 上下文以使用混合精度计算。计算模型输出和损失，并根据 grad_accum_steps 进行梯度累积和优化器更新。
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                # 前向传播：输入数据经过模型得到输出 logits，计算损失函数值。损失值除以 grad_accum_steps 以实现梯度累积。
                output = model(batch_x)
                loss = loss_fn(output, batch_y) / settings.grad_accum_steps

            # 反向传播和优化器更新
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积和优化器更新：每当 step 达到 grad_accum_steps 的倍数时，执行优化器更新并清零梯度。这样可以在显存有限的情况下实现更大的有效 batch size。
            if step % settings.grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # 累积损失以计算平均损失值，便于在日志中输出。每个 epoch 结束后，在验证集上评估模型性能，并根据 Macro-F1 指标保存最优模型。
            total_loss += loss.detach()

        if batch_count > 0 and step % settings.grad_accum_steps != 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if cancel_event is not None and cancel_event.is_set():
            break

        # 在验证集上评估模型性能，计算 Accuracy 和 Macro-F1 指标。根据 Macro-F1 指标判断是否更新最优模型，并实现早停逻辑以避免过拟合。
        val_acc, val_f1 = evaluate(
            model,
            val_split,
            device,
            batch_size=512,
            max_len=MAX_LEN,
            vocab_size=VOCAB_SIZE,
            label_map=label_map,
        )
        avg_loss = (total_loss / max(1, batch_count)).item()
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, ValAcc: {val_acc:.4f}, ValMacroF1: {val_f1:.4f}")

        model.train()

        # 根据验证 Macro-F1 指标判断是否更新最优模型，并实现早停逻辑以避免过拟合。如果当前 epoch 的 Macro-F1 指标比之前的最佳值提升超过 early_stop_min_delta，则更新最佳模型并重置无提升计数器；否则增加无提升计数器，并在达到 early_stop_patience 时触发早停。
        if val_f1 > best_f1 + settings.early_stop_min_delta:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_state_dict = deepcopy(model.state_dict())
            no_improve_epochs = 0
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                max_len=MAX_LEN,
                vocab_size=VOCAB_SIZE,
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
    
    # 训练结束后，如果内存中保存了最优模型的状态字典，则加载该状态字典到模型中，并输出加载信息。最后打印训练完成信息和最优模型保存路径，并返回最优模型和设备。
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best in-memory checkpoint from epoch {best_epoch}, ValMacroF1={best_f1:.4f}")

    if cancel_event is not None and cancel_event.is_set():
        print("Training cancelled.")
    print(f"Training finished. Best model saved to {checkpoint_path}")
    return model, device
