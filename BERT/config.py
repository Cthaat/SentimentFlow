"""BERT 训练配置模块。"""

from __future__ import annotations

import os
from dataclasses import dataclass

# 抑制 HF Hub 后台线程请求和 safetensors 自动转换
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")


MAX_LEN = int(os.getenv("BERT_MODEL_MAX_LEN", "128"))
EPOCHS = int(os.getenv("BERT_EPOCHS", "5"))
DEFAULT_CHUNK_SIZE = 4096
CHECKPOINT_PATH = os.getenv("BERT_CHECKPOINT_PATH", "./bert_sentiment_model")
BERT_CHECKPOINT_PATH = CHECKPOINT_PATH
BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "hfl/chinese-roberta-wwm-ext")


def get_checkpoint_path() -> str:
    """运行时读取 checkpoint 路径，避免 API 动态配置被 import 缓存吞掉。"""
    return os.getenv("BERT_CHECKPOINT_PATH", CHECKPOINT_PATH)


def get_epochs() -> int:
    """运行时读取训练轮数。"""
    return int(os.getenv("BERT_EPOCHS", str(EPOCHS)))


def get_model_name() -> str:
    """运行时读取 BERT 模型名称。"""
    return os.getenv("BERT_MODEL_NAME", BERT_MODEL_NAME)


@dataclass(frozen=True)
class RuntimeSettings:
    """运行期可调参数。"""

    batch_size: int
    eval_batch_size: int
    num_workers: int
    grad_accum_steps: int
    chunk_size: int
    learning_rate: float
    use_weighted_loss: bool
    early_stop_patience: int
    early_stop_min_delta: float


def get_runtime_settings(device_type: str) -> RuntimeSettings:
    """基于设备类型和环境变量，生成训练配置。"""
    batch_size = int(os.getenv("BERT_TRAIN_BATCH_SIZE", "32" if device_type == "cuda" else "16"))
    eval_batch_size = int(os.getenv("BERT_EVAL_BATCH_SIZE", "64" if device_type == "cuda" else "32"))
    num_workers = int(os.getenv("BERT_TRAIN_NUM_WORKERS", "2" if device_type == "cuda" else "0"))
    if os.name == "nt" and num_workers > 4:
        print(
            f"BERT_TRAIN_NUM_WORKERS={num_workers} is high on Windows; "
            "capping to 4 for stability."
        )
        num_workers = 4

    grad_accum_steps = int(os.getenv("BERT_TRAIN_ACCUM_STEPS", "1"))
    chunk_size = int(os.getenv("BERT_TRAIN_CHUNK_SIZE", str(max(batch_size * 8, DEFAULT_CHUNK_SIZE))))
    learning_rate = float(os.getenv("BERT_TRAIN_LR", "2e-5"))
    use_weighted_loss = os.getenv("BERT_TRAIN_WEIGHTED_LOSS", "1") == "1"
    early_stop_patience = max(0, int(os.getenv("BERT_EARLY_STOP_PATIENCE", "2")))
    early_stop_min_delta = max(0.0, float(os.getenv("BERT_EARLY_STOP_MIN_DELTA", "0.0005")))

    return RuntimeSettings(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        grad_accum_steps=grad_accum_steps,
        chunk_size=chunk_size,
        learning_rate=learning_rate,
        use_weighted_loss=use_weighted_loss,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )
