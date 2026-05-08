"""模型定义模块。"""

from __future__ import annotations

import torch.nn as nn

from sentiment_scale import NUM_SENTIMENT_CLASSES


class SentimentLSTMModel(nn.Module):
    """Embedding + LSTM + Linear 的 0-5 情感评分分类网络。"""

    def __init__(self, vocab_size: int, num_classes: int = NUM_SENTIMENT_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 256, num_layers=2, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        emb = self.embedding(x)
        outputs, _ = self.lstm(emb)

        # 取每个样本最后一个“非padding token”的输出，避免被尾部大量 0 冲淡。
        # lengths: [batch_size], 至少为 1（全 0 时兜底取首位）
        lengths = (x != 0).sum(dim=1).clamp(min=1)
        last_index = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, outputs.size(2))
        last_valid_output = outputs.gather(1, last_index).squeeze(1)
        return self.fc(last_valid_output)
