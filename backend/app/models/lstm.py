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
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        pad_idx: int = 0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数:
        - input_ids: [batch_size, seq_len]

        返回:
        - logits: [batch_size, num_classes]
        """
        x = self.embedding(input_ids)
        out, _ = self.lstm(x)

        # 对齐训练时的做法：取每个样本最后一个非 padding token 的输出。
        lengths = (input_ids != 0).sum(dim=1).clamp(min=1)
        last_index = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2))
        last_hidden = out.gather(1, last_index).squeeze(1)
        logits = self.classifier(self.dropout(last_hidden))
        return logits