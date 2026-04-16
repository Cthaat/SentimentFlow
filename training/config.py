"""训练配置模块。

集中定义默认超参数和环境变量读取逻辑，
避免这些配置散落在各个业务函数中。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# 固定超参数（除非你明确希望改算法结构，一般不需要频繁改动）。
MAX_LEN = 100
EPOCHS = 25
VOCAB_SIZE = 65536
DEFAULT_CHUNK_SIZE = 4096
CHECKPOINT_PATH = "./sentiment_model.pt"


@dataclass(frozen=True)
class RuntimeSettings:
    """运行期可调参数。

    这类参数通过环境变量注入，适合按机器资源快速调参。
    """

    batch_size: int
    num_workers: int
    grad_accum_steps: int
    chunk_size: int
    learning_rate: float
    use_weighted_loss: bool
    early_stop_patience: int
    early_stop_min_delta: float


def get_runtime_settings(device_type: str) -> RuntimeSettings:
    """基于设备类型和环境变量，生成训练配置。"""
    # TRAIN_BATCH_SIZE 默认值根据设备类型调整：GPU 推荐较大 batch size，CPU 则较小以避免内存压力。
    batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "256" if device_type == "cuda" else "128"))
    # TRAIN_NUM_WORKERS 默认值：GPU 推荐使用多线程数据加载以提升性能，CPU 则默认单线程以避免过度竞争。
    num_workers = int(os.getenv("TRAIN_NUM_WORKERS", "1" if device_type == "cuda" else "0"))
    if os.name == "nt" and num_workers > 6:
        print(
            f"TRAIN_NUM_WORKERS={num_workers} is high on Windows; "
            "capping to 6 for stability with jieba tokenization."
        )
        num_workers = 6
    # TRAIN_ACCUM_STEPS 默认值为 1，表示不使用梯度累积。你可以根据显存大小和 batch size 调整这个值以实现更大的有效 batch size。
    grad_accum_steps = int(os.getenv("TRAIN_ACCUM_STEPS", "1"))
    # TRAIN_CHUNK_SIZE 默认值根据 batch size 调整，确保每个 chunk 包含足够的数据以充分利用 GPU 的并行计算能力，同时避免过大导致内存压力。你可以根据实际情况调整这个值以获得最佳性能。
    chunk_size = int(os.getenv("TRAIN_CHUNK_SIZE", str(max(batch_size * 8, DEFAULT_CHUNK_SIZE))))
    # TRAIN_LR 默认值为 0.0005，适合大多数情况。你可以根据模型的收敛情况调整这个值以获得更好的性能。
    learning_rate = float(os.getenv("TRAIN_LR", "0.0005"))
    # TRAIN_WEIGHTED_LOSS 默认值为 "1"（启用），表示在训练过程中使用加权损失函数以处理类别不平衡问题。你可以根据数据集的标签分布情况调整这个值，如果你的数据集类别非常不平衡，启用加权损失可能会有助于提升模型的性能。
    use_weighted_loss = os.getenv("TRAIN_WEIGHTED_LOSS", "1") == "1"
    # EARLY_STOP_PATIENCE 默认值为 2，表示验证指标连续 2 个 epoch 无提升时提前停止。
    early_stop_patience = max(0, int(os.getenv("EARLY_STOP_PATIENCE", "2")))
    # EARLY_STOP_MIN_DELTA 默认值为 0.0005，表示至少提升该幅度才视为“有效提升”。
    early_stop_min_delta = max(0.0, float(os.getenv("EARLY_STOP_MIN_DELTA", "0.0005")))

    return RuntimeSettings(
        batch_size=batch_size,
        num_workers=num_workers,
        grad_accum_steps=grad_accum_steps,
        chunk_size=chunk_size,
        learning_rate=learning_rate,
        use_weighted_loss=use_weighted_loss,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )
