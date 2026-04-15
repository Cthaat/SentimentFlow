import os
import zlib

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset


MAX_LEN = 100
EPOCHS = 10
VOCAB_SIZE = 65536
DEFAULT_CHUNK_SIZE = 4096


def tokenize(text):
    return list(str(text).strip())


def encode_text(text, maxlen=MAX_LEN, vocab_size=VOCAB_SIZE):
    ids = [zlib.crc32(token.encode("utf-8")) % (vocab_size - 1) + 1 for token in tokenize(text)]
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


class CsvStreamDataset(IterableDataset):
    def __init__(self, csv_path, chunk_size=DEFAULT_CHUNK_SIZE):
        super().__init__()
        self.csv_path = csv_path
        self.chunk_size = chunk_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        chunk_reader = pd.read_csv(
            self.csv_path,
            usecols=["text", "label"],
            dtype={"text": "string", "label": "int8"},
            chunksize=self.chunk_size,
        )

        for chunk_index, chunk in enumerate(chunk_reader):
            if chunk_index % num_workers != worker_id:
                continue

            chunk = chunk.sample(frac=1).reset_index(drop=True)
            texts = chunk["text"].fillna("").astype(str).tolist()
            labels = chunk["label"].to_numpy(dtype="int64")

            for text, label in zip(texts, labels):
                yield torch.tensor(encode_text(text), dtype=torch.long), torch.tensor(label, dtype=torch.long)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print("Using CUDA device for training.")
    else:
        print("CUDA is not available in this environment. Training will run on CPU.")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "1024" if device.type == "cuda" else "256"))
    num_workers = int(os.getenv("TRAIN_NUM_WORKERS", "1" if device.type == "cuda" else "0"))
    grad_accum_steps = int(os.getenv("TRAIN_ACCUM_STEPS", "4" if device.type == "cuda" else "1"))
    chunk_size = int(os.getenv("TRAIN_CHUNK_SIZE", str(max(batch_size * 8, DEFAULT_CHUNK_SIZE))))

    dataset = CsvStreamDataset("data.csv", chunk_size=chunk_size)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": max(0, num_workers),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1

    loader = DataLoader(dataset, **loader_kwargs)

    model = Model(VOCAB_SIZE).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = torch.zeros((), device=device)
        batch_count = 0
        step = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (batch_x, batch_y) in enumerate(loader, start=1):
            batch_count += 1
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(batch_x)
                loss = loss_fn(output, batch_y) / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.detach()

        if batch_count > 0 and step % grad_accum_steps != 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        print(f"Epoch {epoch + 1}, Loss: {(total_loss / max(1, batch_count)).item():.4f}")

    return model, device


def predict(text, model, device):
    ids = encode_text(text)
    ids = torch.tensor([ids], dtype=torch.long).to(device, non_blocking=True)

    model.eval()
    with torch.inference_mode():
        output = model(ids)

    pred = torch.argmax(output, dim=1).item()
    return "正面" if pred == 1 else "负面"


if __name__ == "__main__":
    model, device = train()
    print(predict("这个电影太棒了！", model, device))
    print(predict("这个电影太差了！", model, device))
    print(predict("这个手机的电池续航很不错！", model, device))
    print(predict("这个手机的屏幕质量很差！", model, device))
    print(predict("这个电脑的性能非常给力！", model, device))
    print(predict("这个电脑的系统经常崩溃！", model, device))
    print(predict("这个耳机的音质非常清晰！", model, device))
    print(predict("这个耳机的做工很糟糕！", model, device))
    print(predict("这个屏幕的显示效果非常好！", model, device))
    print(predict("这个屏幕的亮度很差！", model, device))
    print(predict("这个系统的界面非常流畅！", model, device))
    print(predict("这个系统的响应速度很慢！", model, device))
    print(predict("这个电池的续航时间非常长！", model, device))
    print(predict("这个电池的充电速度很慢！", model, device))
    print(predict("这个物流的配送速度非常快！", model, device))
    print(predict("这个物流的配送速度很慢！", model, device))
    print(predict("这个客服的服务态度非常好！", model, device))
    print(predict("这个客服的服务态度很差！", model, device))
    print(predict("这个包装的质量非常好！", model, device))
    print(predict("这个包装的质量很差！", model, device))
    print(predict("这个做工的质量非常给力！", model, device))
    print(predict("这个做工的质量很差！", model, device))

