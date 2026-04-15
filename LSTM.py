import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


MAX_LEN = 100
EPOCHS = 10


def tokenize(text):
    return list(str(text).strip())


def pad_sequences(sequences, maxlen, padding_value=0):
    padded_sequences = []

    for sequence in sequences:
        sequence = list(sequence)
        if len(sequence) >= maxlen:
            padded_sequences.append(sequence[:maxlen])
        else:
            padded_sequences.append(sequence + [padding_value] * (maxlen - len(sequence)))

    return padded_sequences


def encode_text(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(token, 0) for token in tokenize(text)]
    if len(ids) >= maxlen:
        return ids[:maxlen]
    return ids + [0] * (maxlen - len(ids))


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


def build_vocab(texts):
    vocab = {}
    idx = 1

    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = idx
                idx += 1

    return vocab


def prepare_data(csv_path):
    df = pd.read_csv(csv_path, usecols=["text", "label"], dtype={"text": "string", "label": "int8"})
    texts = df["text"].fillna("").astype(str).tolist()
    labels = df["label"].to_numpy(dtype="int64")

    vocab = build_vocab(texts)
    encoded_inputs = [encode_text(text, vocab) for text in texts]

    x_tensor = torch.tensor(encoded_inputs, dtype=torch.long)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return x_tensor, y_tensor, vocab


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "16384" if device.type == "cuda" else "8192"))
    num_workers = int(os.getenv("TRAIN_NUM_WORKERS", str(min(12, max(1, (os.cpu_count() or 1) - 2)) if device.type == "cuda" else min(4, os.cpu_count() or 1))))

    x_tensor, y_tensor, vocab = prepare_data("data.csv")

    dataset = TensorDataset(x_tensor, y_tensor)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": max(0, num_workers),
        "pin_memory": device.type == "cuda",
        "drop_last": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(dataset, **loader_kwargs)

    model = Model(len(vocab) + 1).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = torch.zeros((), device=device)

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(batch_x)
                loss = loss_fn(output, batch_y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.detach()

        print(f"Epoch {epoch + 1}, Loss: {(total_loss / len(loader)).item():.4f}")

    return model, vocab, device


def predict(text, model, vocab, device):
    ids = encode_text(text, vocab)
    ids = torch.tensor([ids], dtype=torch.long).to(device, non_blocking=True)

    model.eval()
    with torch.inference_mode():
        output = model(ids)

    pred = torch.argmax(output, dim=1).item()
    return "正面" if pred == 1 else "负面"


if __name__ == "__main__":
    model, vocab, device = train()
    print(predict("这个电影太棒了！", model, vocab, device))
    print(predict("这个电影太差了！", model, vocab, device))

