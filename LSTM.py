import pandas as pd
import jieba
import torch
import torch.nn as nn

df = pd.read_csv('data.csv')

def tokenize(test):
    return list(jieba.cut(test))

df["tokens"] = df["text"].apply(tokenize)

vocab = {}
idx = 1

for tokens in df["tokens"]:
    for word in tokens:
        if word not in vocab:
            vocab[word] = idx
            idx += 1
            
def encode(tokens):
    return [vocab[word] for word in tokens]

df["input"] = df["tokens"].apply(encode)

def pad_sequences(sequences, maxlen, padding_value=0):
    padded_sequences = []

    for sequence in sequences:
        sequence = list(sequence)
        if len(sequence) >= maxlen:
            padded_sequences.append(sequence[:maxlen])
        else:
            padded_sequences.append(sequence + [padding_value] * (maxlen - len(sequence)))

    return padded_sequences

X = pad_sequences(df["input"], maxlen=10)
Y = df["label"].values

class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

model = Model(len(vocab) + 1)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.tensor(X, dtype=torch.long)
Y_tensor = torch.tensor(Y, dtype=torch.long)

for epoch in range(10):
    output = model(X_tensor)
    loss = loss_fn(output, Y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss.item()}")
    
def predict(text):
    tokens = list(jieba.cut(text))
    ids = [vocab.get(w, 0) for w in tokens]
    ids = pad_sequences([ids], maxlen=10)
    ids = torch.tensor(ids, dtype=torch.long)
    output = model(ids)
    
    pred = torch.argmax(output, dim=1).item()
    
    return "Positive" if pred == 1 else "Negative"

print(predict("这个电影太棒了！"))
print(predict("这个电影太差了！"))

