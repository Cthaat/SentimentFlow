"""训练模块总入口。

这个包把原来单文件脚本拆成多个小模块，便于你按职责阅读和维护：
- 配置读取
- 数据集构建
- 模型定义
- 训练与评估
- 推理
"""

from .config import MAX_LEN, EPOCHS, VOCAB_SIZE, DEFAULT_CHUNK_SIZE, CHECKPOINT_PATH
from .model import SentimentLSTMModel
from .pipeline import load_checkpoint, load_or_train
from .trainer import train_model
from .inference import predict_text

__all__ = [
    "MAX_LEN",
    "EPOCHS",
    "VOCAB_SIZE",
    "DEFAULT_CHUNK_SIZE",
    "CHECKPOINT_PATH",
    "SentimentLSTMModel",
    "train_model",
    "load_checkpoint",
    "load_or_train",
    "predict_text",
]
