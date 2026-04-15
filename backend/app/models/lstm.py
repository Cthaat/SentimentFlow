import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """LSTM 情感分类模型。

    结构：Embedding -> LSTM -> Dropout -> Linear。
    输入为 token id 序列，输出为各类别的 logits。
    """

    def __init__(
        self,
        # 模型参数，后续可通过配置文件或环境变量传入。
        vocab_size: int,
        # 词嵌入维度，影响模型对语义的捕捉能力和训练效率。
        embed_dim: int = 128,
        # LSTM 隐藏层维度，影响模型容量和表达能力。
        hidden_dim: int = 128,
        # LSTM 层数，增加层数可提升模型复杂度但可能导致过拟合。
        num_layers: int = 1,
        # 分类类别数，默认为二分类（正面/负面），可扩展为多分类。
        num_classes: int = 2,
        # Dropout 比例，防止过拟合，通常在 0.1-0.5 之间调整。
        dropout: float = 0.3,
        # padding token id，确保模型正确忽略填充部分的影响。
        pad_idx: int = 0,
        # 是否使用双向 LSTM，双向可捕捉前后文信息，但会增加模型参数和计算量。
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        # 将离散 token id 映射为稠密向量表示。
        self.embedding = nn.Embedding(
            # 词表大小，必须与训练时一致，确保正确加载预训练权重。
            num_embeddings=vocab_size,
            # 词嵌入维度，影响模型对语义的捕捉能力和训练效率。
            embedding_dim=embed_dim,
            # padding_idx 参数确保模型在计算时忽略填充 token 的影响。
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