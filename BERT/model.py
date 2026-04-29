"""模型定义模块。"""

from __future__ import annotations

import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from .config import BERT_MODEL_NAME


class SentimentBertModel(nn.Module):
    """BERT 二分类网络封装。"""

    def __init__(self, model_name: str = BERT_MODEL_NAME):
        super().__init__()
        self.model_name = model_name
        self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
