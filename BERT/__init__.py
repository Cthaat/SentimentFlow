"""BERT 训练模块总入口。"""

from .config import (
    BERT_CHECKPOINT_PATH,
    BERT_MODEL_NAME,
    CHECKPOINT_PATH,
    EPOCHS,
    MAX_LEN,
)
from .inference import predict_text
from .model import SentimentBertModel
from .pipeline import load_checkpoint, load_or_train
from .trainer import train_model

__all__ = [
    "MAX_LEN",
    "EPOCHS",
    "CHECKPOINT_PATH",
    "BERT_CHECKPOINT_PATH",
    "BERT_MODEL_NAME",
    "SentimentBertModel",
    "train_model",
    "load_checkpoint",
    "load_or_train",
    "predict_text",
]
