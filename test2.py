# ==================================================================

texts = [
    "good movie",
    "bad movie",
    "great film",
    "terrible film",
    "nice one",
    "awful one"
]

labels = [5, 0, 4, 1, 5, 0]  # 0-5 情感评分

# ==================================================================

# 构建词汇表，简单地将每个单词映射到一个唯一的整数 ID，0 用于填充。
def build_vocab(texts):
    vocab = {"<pad>": 0}
    ids = 1
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = ids
                ids += 1
    return vocab

# 编码文本为定长整数序列，使用词汇表进行映射，超过 max_len 的部分截断，不足的部分用 0 填充。
def encode_text(text, vocab, max_len=5):
    ids = [vocab.get(word, 0) for word in text.split()]
    ids = ids[:max_len] + [0] * (max_len - len(ids))
    return ids

# 简单的 Dataset 类，接受文本、标签和词汇表，支持按索引访问和长度查询。
import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        x = encode_text(self.texts[index], self.vocab)
        y = self.labels[index]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

vocab = build_vocab(texts)

from torch.utils.data import DataLoader

dataset = SimpleDataset(texts, labels, vocab)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 单词ID -> 词向量，词向量维度 4，0 用于 padding。
        self.embedding = nn.Embedding(vocab_size, 4)
        # LSTM 理解文本序列，输入维度 16，隐藏状态维度 32，batch_first=True 表示输入输出的 batch 维在第一位。
        self.lstm = nn.LSTM(4, 8, batch_first=True)
        # 线性层将 LSTM 的输出映射到 6 个评分类别的 logits。
        self.fc = nn.Linear(8, 6)
    
    # 前向传播：输入 x 经过嵌入层、LSTM 层，取最后一个时间步的输出，经过线性层得到 logits。
    def forward(self, x):
        print("\n=== Forward 开始 ===")

        # 1️⃣ Embedding
        emb = self.embedding(x)
        print("\n[Embedding输出] shape:", emb.shape)
        print(emb)

        # 2️⃣ LSTM
        out, (h, c) = self.lstm(emb)
        print("\n[LSTM每一步输出] shape:", out.shape)
        print(out)

        print("\n[LSTM最终隐藏状态 h] shape:", h.shape)
        print(h)

        # 3️⃣ 取句子表示
        x = h.squeeze(0)
        print("\n[句子向量] shape:", x.shape)
        print(x)

        # 4️⃣ 分类
        logits = self.fc(x)
        print("\n[logits] shape:", logits.shape)
        print(logits)

        print("\n=== Forward 结束 ===")
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型，输入词汇表大小，移动到设备（GPU 或 CPU）。
model = SimpleModel(len(vocab)).to(device)

# 定义损失函数和优化器，使用交叉熵损失和 Adam 优化器。
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(10):
    total_loss = 0.0
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        output = model(x)
        
        loss = loss_fn(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader)}")
