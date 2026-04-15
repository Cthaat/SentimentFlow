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

X = pad_sequences(df["input"], maxlen=100)
Y = df["label"].values

class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 256, num_layers=2, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(256, 2)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(len(vocab) + 1).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.tensor(X, dtype=torch.long).to(device)
Y_tensor = torch.tensor(Y, dtype=torch.long).to(device)

from torch.utils.data import DataLoader, TensorDataset
dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True)

for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

def predict(text):
    tokens = list(jieba.cut(text))
    ids = [vocab.get(w, 0) for w in tokens]
    ids = pad_sequences([ids], maxlen=100)
    ids = torch.tensor(ids, dtype=torch.long).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(ids)
    
    pred = torch.argmax(output, dim=1).item()
    return "正面" if pred == 1 else "负面"

print(predict("这个电影太棒了！"))
print(predict("这个电影太差了！"))

