"""模型定义模块。"""

from __future__ import annotations

import torch.nn as nn


class SentimentLSTMModel(nn.Module):
    """Embedding + LSTM + Linear 的二分类网络。"""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 256, num_layers=2, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])
