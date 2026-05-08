"""模型定义模块。"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

from sentiment_scale import NUM_SENTIMENT_CLASSES

from .config import BERT_MODEL_NAME


class SentimentBertModel(nn.Module):
    """BERT 0-5 情感评分模型。

    默认使用 hybrid ordinal architecture：
    - classification logits：保持现有 API 概率契约；
    - ordinal threshold logits：用于 CORAL/CORN 风格累计二分类损失；
    - regression score：用于评分距离约束。

    ``architecture="sequence"`` 保留旧的 AutoModelForSequenceClassification
    兼容路径，用于加载历史 checkpoint。
    """

    def __init__(
        self,
        model_name: str = BERT_MODEL_NAME,
        num_labels: int = NUM_SENTIMENT_CLASSES,
        *,
        architecture: str | None = None,
        dropout: float | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.architecture = (architecture or os.getenv("BERT_MODEL_ARCHITECTURE", "hybrid")).strip().lower()
        if self.architecture == "sequence":
            self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            self.dropout = None
            self.classifier = None
            self.ordinal_head = None
            self.regression_head = None
            return

        if self.architecture != "hybrid":
            raise ValueError(f"Unsupported BERT_MODEL_ARCHITECTURE={self.architecture!r}.")

        config = AutoConfig.from_pretrained(model_name)
        attention_impl = os.getenv("BERT_ATTENTION_IMPLEMENTATION", "sdpa").strip()
        try:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                config=config,
                attn_implementation=attention_impl or None,
            )
        except (TypeError, ValueError):
            self.backbone = AutoModel.from_pretrained(model_name, config=config)

        hidden_size = int(getattr(config, "hidden_size", 768))
        dropout_prob = (
            float(dropout)
            if dropout is not None
            else float(os.getenv("BERT_CLASSIFIER_DROPOUT", str(getattr(config, "hidden_dropout_prob", 0.1))))
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.ordinal_head = nn.Linear(hidden_size, num_labels - 1)
        self.regression_head = nn.Linear(hidden_size, 1)

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        if hasattr(self.backbone.config, "use_cache"):
            self.backbone.config.use_cache = False

    def forward(self, input_ids, attention_mask, *, return_dict: bool = False):
        if self.architecture == "sequence":
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            return {"logits": outputs.logits} if return_dict else outputs.logits

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        class_logits = self.classifier(pooled)
        ordinal_logits = self.ordinal_head(pooled)
        score = self.regression_head(pooled).squeeze(-1)
        if return_dict:
            return {
                "logits": class_logits,
                "ordinal_logits": ordinal_logits,
                "score": score,
            }
        return class_logits

    def save_pretrained(self, save_dir) -> None:
        """Save backbone plus custom heads in a HF-compatible directory."""
        self.backbone.save_pretrained(save_dir)
        if self.architecture == "hybrid":
            torch.save(self.state_dict(), os.path.join(str(save_dir), "sentiment_model_state.pt"))
