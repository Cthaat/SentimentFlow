import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """LSTM 情感分类模型。

    结构：Embedding -> LSTM -> Dropout -> Linear。
    输入为 token id 序列，输出为各类别的 logits。
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        # 将离散 token id 映射为稠密向量表示。
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        # 时序编码层，提取文本上下文特征。
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        # 双向 LSTM 时输出维度翻倍。
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        # 分类头：将句向量映射到类别空间。
        self.classifier = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数:
        - input_ids: [batch_size, seq_len]

        返回:
        - logits: [batch_size, num_classes]
        """
        # 1) token id -> embedding 向量序列。
        x = self.embedding(input_ids)
        # 2) LSTM 编码，h_n 为各层最后时刻隐藏状态。
        out, (h_n, _) = self.lstm(x)
        # 3) 取顶层最后隐藏状态作为句级表示。
        last_hidden = h_n[-1]
        # 4) Dropout + 线性层得到分类 logits。
        logits = self.classifier(self.dropout(last_hidden))
        return logits