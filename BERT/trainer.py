"""训练主流程模块。"""

from __future__ import annotations

import math
import os
import threading
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from sentiment_scale import NUM_SENTIMENT_CLASSES
from ordinal_loss import (
    DistanceAwareOrdinalLoss,
    OrdinalLossConfig,
    effective_number_class_weights,
)

from .checkpoint import save_checkpoint
from .config import (
    MAX_LEN,
    get_checkpoint_path,
    get_epochs,
    get_model_name,
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
        "soft_labels": _collate_soft_labels(records),
        "sample_weights": torch.tensor(
            [float(record.get("sample_weight", 1.0)) for record in records],
            dtype=torch.float32,
        ),
        "label_sources": [str(record.get("label_source", "real")) for record in records],
    }


def _normalize_batch_record(item) -> dict:
    if isinstance(item, dict):
        return item
    text, label = item
    return {
        "text": str(text),
        "label": int(label),
        "soft_labels": None,
        "sample_weight": 1.0,
        "label_source": "legacy_tuple",
    }


def _collate_soft_labels(records: list[dict]) -> torch.Tensor | None:
    values = [record.get("soft_labels") for record in records]
    if not any(value is not None for value in values):
        return None

    rows: list[list[float]] = []
    for record, value in zip(records, values):
        if value is None:
            label = int(record["label"])
            rows.append([1.0 if index == label else 0.0 for index in range(NUM_SENTIMENT_CLASSES)])
        else:
            row = [float(item) for item in value]
            if len(row) != NUM_SENTIMENT_CLASSES:
                raise ValueError(f"Expected {NUM_SENTIMENT_CLASSES} soft-label probabilities, got {len(row)}.")
            total = sum(row)
            if total <= 0:
                label = int(record["label"])
                row = [1.0 if index == label else 0.0 for index in range(NUM_SENTIMENT_CLASSES)]
            else:
                row = [item / total for item in row]
            rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def _build_train_loader(train_split, settings, device: torch.device, label_map: dict | None = None):
    dataset = CsvStreamDataset(
        train_split,
        chunk_size=settings.chunk_size,
        label_map=label_map,
        labels_are_normalized=True,
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


def _build_loss_fn(class_counts: list[int], settings, device: torch.device):
    total_count = sum(class_counts)
    present_classes = sum(1 for count in class_counts if count > 0)
    loss_fn = DistanceAwareOrdinalLoss(config=_ordinal_loss_config())
    if settings.use_weighted_loss and present_classes > 1:
        weights = effective_number_class_weights(
            class_counts,
            beta=float(os.getenv("CLASS_BALANCED_BETA", "0.9999")),
            min_weight=float(os.getenv("CLASS_WEIGHT_MIN", "0.05")),
        )
        class_weights = weights.to(device)
        loss_fn = DistanceAwareOrdinalLoss(
            class_weights=class_weights,
            config=_ordinal_loss_config(),
        )
        print(
            "Using effective-number class-balanced loss: "
            + ", ".join(f"score_{idx}_w={float(weight):.4f}" for idx, weight in enumerate(weights.tolist()))
        )

    if total_count > 0:
        priors = torch.tensor(
            [max(1, count) / max(1, total_count) for count in class_counts],
            dtype=torch.float32,
            device=device,
        )
        loss_fn.set_logit_adjustment(priors)
    return loss_fn


def _ordinal_loss_config() -> OrdinalLossConfig:
    return OrdinalLossConfig(
        ce_weight=float(os.getenv("ORDINAL_CE_WEIGHT", "1.0")),
        distance_weight=float(os.getenv("ORDINAL_DISTANCE_WEIGHT", "0.35")),
        ordinal_weight=float(os.getenv("ORDINAL_BCE_WEIGHT", "0.5")),
        regression_weight=float(os.getenv("ORDINAL_REGRESSION_WEIGHT", "0.2")),
        label_smoothing=float(os.getenv("ORDINAL_LABEL_SMOOTHING", "0.05")),
        pseudo_label_smoothing=float(os.getenv("PSEUDO_LABEL_SMOOTHING", "0.02")),
        focal_gamma=float(os.getenv("FOCAL_GAMMA", "1.5")),
        logit_adjustment_weight=float(os.getenv("LOGIT_ADJUSTMENT_WEIGHT", "0.3")),
    )


def _apply_curriculum_sample_weights(
    sample_weights: torch.Tensor,
    label_sources: list[str] | None,
    epoch: int,
) -> torch.Tensor:
    warmup_epochs = max(0, int(os.getenv("BERT_PSEUDO_CURRICULUM_EPOCHS", "2")))
    if warmup_epochs <= 0 or not label_sources:
        return sample_weights

    start_scale = max(0.0, min(1.0, float(os.getenv("BERT_PSEUDO_CURRICULUM_START_SCALE", "0.3"))))
    progress = min(1.0, float(epoch + 1) / max(1, warmup_epochs))
    pseudo_scale = start_scale + (1.0 - start_scale) * progress
    if pseudo_scale >= 0.999:
        return sample_weights

    scaled = sample_weights.clone()
    for index, source in enumerate(label_sources):
        if source in {"pseudo", "interpolated"}:
            scaled[index] = scaled[index] * pseudo_scale
    return scaled


def _amp_dtype() -> torch.dtype:
    precision = os.getenv("BERT_MIXED_PRECISION", os.getenv("MIXED_PRECISION", "fp16")).strip().lower()
    return torch.bfloat16 if precision == "bf16" else torch.float16


def _init_experiment_loggers():
    tensorboard_writer = None
    wandb_run = None
    tensorboard_dir = os.getenv("TENSORBOARD_LOG_DIR", os.getenv("BERT_TENSORBOARD_LOG_DIR", "")).strip()
    if tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tensorboard_writer = SummaryWriter(tensorboard_dir)
        except Exception as exc:
            print(f"Warning: TensorBoard logger disabled: {type(exc).__name__}: {exc}")

    wandb_project = os.getenv("WANDB_PROJECT", "").strip()
    if wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                name=os.getenv("WANDB_RUN_NAME") or None,
                config={
                    "training_stage": os.getenv("BERT_TRAINING_STAGE", os.getenv("TRAINING_STAGE", "auto")),
                    "model_name": get_model_name(),
                },
            )
        except Exception as exc:
            print(f"Warning: wandb logger disabled: {type(exc).__name__}: {exc}")
    return tensorboard_writer, wandb_run


def _log_metrics(tensorboard_writer, wandb_run, epoch: int, loss: float, metrics) -> None:
    payload = {
        "train/loss": loss,
        "val/accuracy": metrics.accuracy,
        "val/macro_f1": metrics.macro_f1,
        "val/weighted_f1": metrics.weighted_f1,
        "val/mae": metrics.mae,
        "val/rmse": metrics.rmse,
        "val/qwk": metrics.quadratic_weighted_kappa,
        "val/spearman": metrics.spearman,
    }
    if tensorboard_writer is not None:
        for key, value in payload.items():
            tensorboard_writer.add_scalar(key, value, epoch)
    if wandb_run is not None:
        wandb_run.log({**payload, "epoch": epoch})


def _trainer_state_path(checkpoint_path: str) -> Path:
    return Path(checkpoint_path) / "trainer_state.pt"


def _resume_state_candidates(checkpoint_path: str, resume_path: str) -> list[Path]:
    candidates = [_trainer_state_path(checkpoint_path)]
    if resume_path:
        resume_state = _trainer_state_path(resume_path)
        if resume_state not in candidates:
            candidates.append(resume_state)
    return candidates


def _unwrap_model(model, accelerator=None):
    if accelerator is not None:
        try:
            model = accelerator.unwrap_model(model)
        except Exception:
            pass
    model = getattr(model, "module", model)
    return getattr(model, "_orig_mod", model)


def _init_accelerator():
    use_accelerate = os.getenv("BERT_USE_ACCELERATE", os.getenv("USE_ACCELERATE", "0")).strip() == "1"
    if not use_accelerate:
        return None
    try:
        from accelerate import Accelerator
    except Exception as exc:
        print(f"Warning: Accelerate requested but unavailable: {type(exc).__name__}: {exc}")
        return None

    precision = os.getenv("BERT_MIXED_PRECISION", os.getenv("MIXED_PRECISION", "fp16")).strip().lower()
    mixed_precision = "bf16" if precision == "bf16" else "fp16" if precision == "fp16" else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    print(f"Accelerate enabled: mixed_precision={mixed_precision}, device={accelerator.device}")
    return accelerator


def _is_main_process(accelerator) -> bool:
    return accelerator is None or bool(getattr(accelerator, "is_main_process", True))


def _metric_for_selection(metrics) -> float:
    metric_name = os.getenv("BERT_SELECTION_METRIC", "qwk").strip().lower()
    if metric_name == "macro_f1":
        return float(metrics.macro_f1)
    if metric_name == "weighted_f1":
        return float(metrics.weighted_f1)
    if metric_name == "mae":
        return -float(metrics.mae)
    return float(metrics.quadratic_weighted_kappa)


def _build_optimizer(model, settings, device: torch.device):
    base_lr = settings.learning_rate
    weight_decay = float(os.getenv("BERT_WEIGHT_DECAY", "0.01"))
    layer_decay = float(os.getenv("BERT_LAYERWISE_LR_DECAY", "0.9"))
    head_lr_multiplier = float(os.getenv("BERT_HEAD_LR_MULTIPLIER", "2.0"))
    no_decay_terms = ("bias", "LayerNorm.weight", "layer_norm.weight")
    backbone_config = getattr(getattr(model, "backbone", None), "config", None)
    num_layers = int(getattr(backbone_config, "num_hidden_layers", 12) or 12)

    grouped: dict[tuple[float, float], list[torch.nn.Parameter]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and os.getenv("BERT_INCLUDE_FROZEN_PARAMS_IN_OPTIMIZER", "1") != "1":
            continue
        weight_decay_value = 0.0 if any(term in name for term in no_decay_terms) else weight_decay
        lr_scale = _layer_lr_scale(name, num_layers, layer_decay, head_lr_multiplier)
        grouped[(base_lr * lr_scale, weight_decay_value)].append(parameter)

    parameter_groups = [
        {"params": params, "lr": lr, "weight_decay": decay}
        for (lr, decay), params in grouped.items()
        if params
    ]
    use_fused = device.type == "cuda" and os.getenv("BERT_FUSED_ADAMW", "1") == "1"
    try:
        optimizer = torch.optim.AdamW(parameter_groups, lr=base_lr, fused=use_fused)
    except TypeError:
        optimizer = torch.optim.AdamW(parameter_groups, lr=base_lr)
    print(
        f"Optimizer: AdamW groups={len(parameter_groups)}, base_lr={base_lr}, "
        f"weight_decay={weight_decay}, layer_decay={layer_decay}, fused={use_fused}"
    )
    return optimizer


def _layer_lr_scale(name: str, num_layers: int, layer_decay: float, head_lr_multiplier: float) -> float:
    if not name.startswith("backbone."):
        return head_lr_multiplier
    marker = ".encoder.layer."
    if marker not in name:
        return layer_decay ** num_layers
    try:
        layer_id = int(name.split(marker, 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        return 1.0
    return layer_decay ** max(0, num_layers - layer_id - 1)


def _build_scheduler(optimizer, *, estimated_batches: int, epochs: int, start_epoch: int, settings):
    update_steps_per_epoch = max(1, math.ceil(estimated_batches / max(1, settings.grad_accum_steps)))
    total_steps = max(1, update_steps_per_epoch * max(1, epochs - start_epoch))
    warmup_ratio = float(os.getenv("BERT_WARMUP_RATIO", "0.06"))
    warmup_steps = int(os.getenv("BERT_WARMUP_STEPS", "0")) or int(total_steps * warmup_ratio)
    scheduler_name = os.getenv("BERT_SCHEDULER", "cosine").strip().lower()
    if scheduler_name == "none":
        return None
    try:
        from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

        if scheduler_name == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    except Exception:
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / max(1, warmup_steps)
            progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"Scheduler: {scheduler_name}, total_steps={total_steps}, warmup_steps={warmup_steps}")
    return scheduler


def _apply_gradual_unfreezing(model, epoch: int) -> None:
    total_epochs = int(os.getenv("BERT_GRADUAL_UNFREEZE_EPOCHS", "0"))
    if total_epochs <= 0:
        return
    base_model = _unwrap_model(model)
    backbone = getattr(base_model, "backbone", None)
    if backbone is None:
        return
    num_layers = int(getattr(getattr(backbone, "config", None), "num_hidden_layers", 12))
    trainable_layers = min(num_layers, max(1, math.ceil(num_layers * (epoch + 1) / total_epochs)))
    for name, parameter in backbone.named_parameters():
        marker = "encoder.layer."
        if marker in name:
            try:
                layer_id = int(name.split(marker, 1)[1].split(".", 1)[0])
            except (IndexError, ValueError):
                layer_id = num_layers
            parameter.requires_grad = layer_id >= num_layers - trainable_layers
        else:
            parameter.requires_grad = epoch + 1 >= total_epochs
    print(f"Gradual unfreezing: epoch={epoch + 1}, trainable_last_layers={trainable_layers}/{num_layers}")


def _maybe_compile_model(model):
    if os.getenv("BERT_TORCH_COMPILE", "0") != "1" or not hasattr(torch, "compile"):
        return model
    try:
        compiled = torch.compile(model, mode=os.getenv("BERT_TORCH_COMPILE_MODE", "reduce-overhead"))
        print("torch.compile enabled for BERT model.")
        return compiled
    except Exception as exc:
        print(f"Warning: torch.compile disabled: {type(exc).__name__}: {exc}")
        return model


def _save_trainer_state(
    state_path: Path,
    *,
    epoch: int,
    optimizer,
    scheduler,
    scaler,
    best_metric: float,
    best_epoch: int,
    no_improve_epochs: int,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "best_f1": best_metric,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "no_improve_epochs": no_improve_epochs,
        },
        state_path,
    )


def train_model(cancel_event: threading.Event | None = None):
    """执行完整训练，返回最优模型和设备。"""
    accelerator = _init_accelerator()
    device = accelerator.device if accelerator is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using CUDA device for BERT training.")
    else:
        print("CUDA is not available in this environment. BERT training will run on CPU.")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    settings = get_runtime_settings(device.type)
    epochs = get_epochs()
    checkpoint_path = get_checkpoint_path()
    model_name = get_model_name()
    log_interval = max(1, int(os.getenv("BERT_TRAIN_LOG_INTERVAL", "10")))
    dataset_names, train_split, val_split, label_map = build_train_split_and_val_split()
    train_size = len(train_split)
    val_size = len(val_split)

    class_counts = get_label_distribution(train_split, label_map)
    total_count = max(1, sum(class_counts))

    print(f"Datasets: {', '.join(dataset_names)}")
    print("DATA SUMMARY:")
    print(f"  Total train samples (after merge): {train_size:,}")
    print(f"  Total val samples: {val_size:,}")
    print(
        f"Train samples: {total_count}, score_counts={class_counts}"
    )
    print(
        f"Training config: batch_size={settings.batch_size}, "
        f"eval_batch_size={settings.eval_batch_size}, "
        f"grad_accum_steps={settings.grad_accum_steps}, num_workers={settings.num_workers}, "
        f"epochs={epochs}, "
        f"log_interval={log_interval}, "
        f"early_stop_patience={settings.early_stop_patience}, "
        f"early_stop_min_delta={settings.early_stop_min_delta}, "
        f"pseudo_curriculum_epochs={os.getenv('BERT_PSEUDO_CURRICULUM_EPOCHS', '2')}, "
        f"model={model_name}"
    )

    loader = _build_train_loader(train_split, settings, device, label_map)
    estimated_batches = max(1, math.ceil(train_size / settings.batch_size))

    resume_path = os.getenv("BERT_RESUME_FROM_CHECKPOINT", "").strip()
    if resume_path:
        from .checkpoint import load_checkpoint

        resumed_model = load_checkpoint(resume_path, device=device)
        if resumed_model is None:
            raise FileNotFoundError(f"BERT_RESUME_FROM_CHECKPOINT is not loadable: {resume_path}")
        model = resumed_model
        print(f"Resumed BERT model weights from {resume_path}")
    else:
        model = SentimentBertModel(model_name=model_name, num_labels=NUM_SENTIMENT_CLASSES).to(device)

    if os.getenv("BERT_GRADIENT_CHECKPOINTING", "1") == "1":
        _unwrap_model(model, accelerator).enable_gradient_checkpointing()
        print("Gradient checkpointing enabled for BERT backbone.")

    loss_fn = _build_loss_fn(class_counts, settings, device)
    optimizer = _build_optimizer(model, settings, device)
    amp_dtype = _amp_dtype()
    scaler = None if accelerator is not None else torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and amp_dtype == torch.float16,
    )
    tensorboard_writer, wandb_run = _init_experiment_loggers() if _is_main_process(accelerator) else (None, None)

    scheduler = _build_scheduler(
        optimizer,
        estimated_batches=estimated_batches,
        epochs=epochs,
        start_epoch=0,
        settings=settings,
    )

    model = _maybe_compile_model(model)
    if (
        accelerator is None
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
        and os.getenv("BERT_USE_DATA_PARALLEL", "1") == "1"
    ):
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs.")

    best_metric = -float("inf")
    best_epoch = -1
    best_state_dict = None
    no_improve_epochs = 0
    start_epoch = 0
    state_path = _trainer_state_path(checkpoint_path)
    loaded_state_path = next((path for path in _resume_state_candidates(checkpoint_path, resume_path) if path.exists()), None)
    if resume_path and loaded_state_path is not None:
        trainer_state = torch.load(loaded_state_path, map_location="cpu")
        optimizer_state = trainer_state.get("optimizer_state_dict")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        if scaler is not None and scaler.is_enabled() and trainer_state.get("scaler_state_dict"):
            scaler.load_state_dict(trainer_state["scaler_state_dict"])
        if scheduler is not None and trainer_state.get("scheduler_state_dict"):
            scheduler.load_state_dict(trainer_state["scheduler_state_dict"])
        best_metric = float(trainer_state.get("best_metric", trainer_state.get("best_f1", best_metric)))
        best_epoch = int(trainer_state.get("best_epoch", best_epoch))
        no_improve_epochs = int(trainer_state.get("no_improve_epochs", no_improve_epochs))
        start_epoch = int(trainer_state.get("epoch", 0))
        print(f"Resumed trainer state from {loaded_state_path} at epoch {start_epoch}.")

    if accelerator is not None:
        if scheduler is None:
            model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
        else:
            model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    model.train()
    for epoch in range(start_epoch, epochs):
        if cancel_event is not None and cancel_event.is_set():
            print("Training cancelled before next epoch.")
            break
        _apply_gradual_unfreezing(model, epoch)

        total_loss = torch.zeros((), device=device)
        batch_count = 0
        step = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, start=1):
            if cancel_event is not None and cancel_event.is_set():
                print("Training cancellation requested; stopping current epoch.")
                break

            batch_count += 1
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            soft_labels = batch["soft_labels"]
            if soft_labels is not None:
                soft_labels = soft_labels.to(device, non_blocking=True)
            sample_weights = _apply_curriculum_sample_weights(
                batch["sample_weights"],
                batch.get("label_sources"),
                epoch,
            ).to(device, non_blocking=True)

            autocast_context = (
                accelerator.autocast()
                if accelerator is not None
                else torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda")
            )
            with autocast_context:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                loss = loss_fn(
                    outputs,
                    labels,
                    soft_labels=soft_labels,
                    sample_weights=sample_weights,
                ) / settings.grad_accum_steps

            if accelerator is not None:
                accelerator.backward(loss)
            elif scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % settings.grad_accum_steps == 0:
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.detach() * settings.grad_accum_steps

            if step == 1 or step % log_interval == 0:
                step_loss = (loss.detach() * settings.grad_accum_steps).item()
                print(
                    f"Epoch {epoch + 1}/{epochs}, Step {step}/{estimated_batches}, "
                    f"Loss: {step_loss:.4f}"
                )

        if batch_count > 0 and step % settings.grad_accum_steps != 0:
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if cancel_event is not None and cancel_event.is_set():
            break

        eval_model = _unwrap_model(model, accelerator)
        metrics = evaluate(
            eval_model,
            val_split,
            device,
            batch_size=settings.eval_batch_size,
            max_len=MAX_LEN,
            label_map=label_map,
        )
        avg_loss = (total_loss / max(1, batch_count)).item()
        print(
            f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, "
            f"ValAcc: {metrics.accuracy:.4f}, "
            f"ValMacroF1: {metrics.macro_f1:.4f}, "
            f"ValWeightedF1: {metrics.weighted_f1:.4f}, "
            f"ValMAE: {metrics.mae:.4f}, "
            f"ValRMSE: {metrics.rmse:.4f}, "
            f"ValQWK: {metrics.quadratic_weighted_kappa:.4f}, "
            f"ValSpearman: {metrics.spearman:.4f}"
        )
        print(f"ValConfusionMatrix: {metrics.confusion_matrix}")
        print(f"ValPerClassF1: {[round(value, 4) for value in metrics.per_class_f1]}")
        _log_metrics(tensorboard_writer, wandb_run, epoch + 1, avg_loss, metrics)

        model.train()
        should_stop = False
        current_metric = _metric_for_selection(metrics)

        if current_metric > best_metric + settings.early_stop_min_delta:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_state_dict = deepcopy(_unwrap_model(model, accelerator).state_dict())
            no_improve_epochs = 0
            if _is_main_process(accelerator):
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=_unwrap_model(model, accelerator),
                    max_len=MAX_LEN,
                    model_name=model_name,
                    best_val_f1=metrics.macro_f1,
                    best_epoch=best_epoch,
                    metrics=metrics,
                )
            print(
                f"Best model updated at epoch {epoch + 1}, "
                f"SelectionMetric={current_metric:.4f}, ValMacroF1={metrics.macro_f1:.4f}, "
                f"ValQWK={metrics.quadratic_weighted_kappa:.4f}"
            )
        else:
            no_improve_epochs += 1
            print(
                f"No significant improvement for {no_improve_epochs} epoch(s) "
                f"(best={best_metric:.4f}, current={current_metric:.4f}, "
                f"min_delta={settings.early_stop_min_delta})."
            )
            if settings.early_stop_patience > 0 and no_improve_epochs >= settings.early_stop_patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"No improvement for {no_improve_epochs} consecutive epoch(s)."
                )
                should_stop = True

        if _is_main_process(accelerator):
            _save_trainer_state(
                state_path,
                epoch=epoch + 1,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_metric=best_metric,
                best_epoch=best_epoch,
                no_improve_epochs=no_improve_epochs,
            )
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if should_stop:
            break

    if best_state_dict is not None:
        _unwrap_model(model, accelerator).load_state_dict(best_state_dict)
        print(f"Loaded best in-memory checkpoint from epoch {best_epoch}, SelectionMetric={best_metric:.4f}")

    if tensorboard_writer is not None:
        tensorboard_writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    if cancel_event is not None and cancel_event.is_set():
        print("Training cancelled.")
    print(f"Training finished. Best model saved to {checkpoint_path}")
    return model, device
