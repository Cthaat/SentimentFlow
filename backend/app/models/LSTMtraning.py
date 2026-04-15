import os
import zlib
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset


# 序列长度、训练轮数和词表大小都是训练时最核心的三个超参数。
MAX_LEN = 100
EPOCHS = 10
VOCAB_SIZE = 65536
DEFAULT_CHUNK_SIZE = 4096

# 固定为项目/脚本级绝对路径，避免受启动目录影响。
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_CSV_PATH = Path(os.getenv("TRAIN_DATA_PATH", str(PROJECT_ROOT / "data.csv"))).resolve()
CHECKPOINT_PATH = Path(
    os.getenv("TRAIN_CHECKPOINT_PATH", str(SCRIPT_DIR / "sentiment_model.pt"))
).resolve()


# 这份脚本的目标很简单：
# 1. 从 CSV 里流式读取文本和标签
# 2. 把文本变成数字序列
# 3. 用 LSTM 做二分类训练
# 4. 训练完后直接拿几个样例做预测
#
# 之所以不用一次性把全部数据读进内存，是因为你的数据量很大，
# 如果全量载入，内存会先被撑爆，GPU 反而吃不到足够的数据。
def tokenize(text):
    # 这里用字符级切分：
    # 1. 不依赖 jieba，速度更快
    # 2. 对你这种模板化短句，按字符切分已经足够
    # 3. 可以避免分词器把字符串处理得太慢
    return list(str(text).strip())


def encode_text(text, maxlen=MAX_LEN, vocab_size=VOCAB_SIZE):
    # 这里把每个字符映射成一个整数 id。
    # 不是先统计全量词表，而是直接对字符做哈希映射，优点是：
    # 1. 不需要预先扫描全量数据
    # 2. 内存更省
    # 3. 训练开始更快
    # 代价是会有少量哈希冲突，但对这种简单二分类任务通常可以接受。
    ids = [zlib.crc32(token.encode("utf-8")) % (vocab_size - 1) + 1 for token in tokenize(text)]

    # 所有样本都要变成同样长度，方便批量送进 LSTM。
    # 长的截断，短的补 0。
    if len(ids) >= maxlen:
        return ids[:maxlen]
    return ids + [0] * (maxlen - len(ids))


class Model(nn.Module):
    # Embedding + LSTM + Linear 是一个很标准的文本分类结构。
    # Embedding 把离散 id 变成向量，LSTM 学习顺序关系，Linear 输出二分类结果。
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 256, num_layers=2, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        # 输入 x 的形状通常是 [batch_size, seq_len]。
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)

        # h[-1] 是最后一层 LSTM 的最后时刻隐状态。
        # 对分类任务来说，这一向量相当于整句话的摘要。
        out = self.fc(h[-1])
        return out


class CsvStreamDataset(IterableDataset):
    # 这个数据集只保存文件路径和 chunk 大小，
    # 真正读取数据是在迭代的时候进行的。
    # 这样可以避免一次性占用大量内存。
    def __init__(self, csv_path, chunk_size=DEFAULT_CHUNK_SIZE):
        super().__init__()
        self.csv_path = csv_path
        self.chunk_size = chunk_size

    def __iter__(self):
        # 迭代器真正开始工作时，才从磁盘按 chunk 读取 CSV。
        # 这样每次只处理一小块数据，适合大文件。
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
            # 多 worker 之间分片，避免不同 worker 读到同一块数据。
            if chunk_index % num_workers != worker_id:
                continue

            # chunk 内部再打乱一次，减少样本顺序太固定带来的影响。
            chunk = chunk.sample(frac=1).reset_index(drop=True)

            # 文本列和标签列分别取出来，方便逐条编码和训练。
            texts = chunk["text"].fillna("").astype(str).tolist()
            labels = chunk["label"].to_numpy(dtype="int64")

            for text, label in zip(texts, labels):
                # 每次 yield 一条样本，DataLoader 会自动把多条样本拼成 batch。
                yield torch.tensor(encode_text(text), dtype=torch.long), torch.tensor(label, dtype=torch.long)


def train():
    # 自动判断当前机器有没有 CUDA。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print("Using CUDA device for training.")
    else:
        print("CUDA is not available in this environment. Training will run on CPU.")

    if device.type == "cuda":
        # 让 cuDNN 针对固定输入形状做优化，提升吞吐。
        # 你的输入序列长度是固定的，所以这个优化通常有效。
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            # 允许 matmul 选择更快的精度路径，通常对训练速度更友好。
            # 这会用一点点数值精度，换取更好的速度。
            torch.set_float32_matmul_precision("high")

    # 这些环境变量的目的，是让你不用改代码就能调训练参数：
    # - TRAIN_BATCH_SIZE：每次喂给 GPU 多少样本
    # - TRAIN_NUM_WORKERS：几个 CPU worker 同时读数据
    # - TRAIN_ACCUM_STEPS：梯度累积步数
    # - TRAIN_CHUNK_SIZE：每次从 CSV 读多少行
    batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "1024" if device.type == "cuda" else "256"))
    num_workers = int(os.getenv("TRAIN_NUM_WORKERS", "1" if device.type == "cuda" else "0"))
    grad_accum_steps = int(os.getenv("TRAIN_ACCUM_STEPS", "4" if device.type == "cuda" else "1"))
    chunk_size = int(os.getenv("TRAIN_CHUNK_SIZE", str(max(batch_size * 8, DEFAULT_CHUNK_SIZE))))

    # chunk_size 越大，每次读盘越少，但单次 CPU 内存峰值也会更高。
    # 如果你机器内存偏小，就不要把它设得太大。
    dataset = CsvStreamDataset(str(DATA_CSV_PATH), chunk_size=chunk_size)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": max(0, num_workers),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        # 预取只保留 1 个 batch，减少后台堆积的数据占用。
        # 这样更省内存，但吞吐不一定是最高的。
        loader_kwargs["prefetch_factor"] = 1

    # DataLoader 负责把单条样本拼成 batch，并在需要时并行读取。
    loader = DataLoader(dataset, **loader_kwargs)

    # 固定词表大小后，模型就不需要先扫描全量文本统计词典。
    # 这让训练启动更快，也更省内存。
    model = Model(VOCAB_SIZE).to(device)

    # 交叉熵损失用于二分类，是分类任务最常见的损失函数之一。
    loss_fn = nn.CrossEntropyLoss()

    # AdamW 是比较稳的优化器，通常比普通 SGD 更容易收敛。
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 混合精度训练需要 GradScaler，防止 float16 下梯度太小而下溢。
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    model.train()
    for epoch in range(EPOCHS):
        # total_loss 用于打印每个 epoch 的平均训练损失。
        # 这里只是统计用，不参与反向传播。
        total_loss = torch.zeros((), device=device)
        batch_count = 0
        step = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (batch_x, batch_y) in enumerate(loader, start=1):
            batch_count += 1

            # non_blocking=True 的前提是 pin_memory=True，
            # 这样 CPU -> GPU 拷贝更容易和计算重叠。
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # 混合精度可以减少显存占用并提高 CUDA 训练速度。
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(batch_x)
                loss = loss_fn(output, batch_y) / grad_accum_steps

            # 梯度累积：小 batch 训练时模拟更大的有效 batch。
            # 这样可以降低显存压力，但总训练时间通常不会比真大 batch 更快。
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_accum_steps == 0:
                # 到达累积步数后再更新一次参数，显存更省。
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # 记录损失时用的是缩放后的 loss，主要用于观察趋势，不必特别精确。
            total_loss += loss.detach()

        # 如果最后几个 batch 没有凑满累积步数，也要补一次参数更新。
        if batch_count > 0 and step % grad_accum_steps != 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # 这里只打印一个 epoch 的平均 loss，方便你看训练是不是在下降。
        print(f"Epoch {epoch + 1}, Loss: {(total_loss / max(1, batch_count)).item():.4f}")

    # 训练结束后保存模型参数。下次启动时可以直接加载，不用重新训练。
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "max_len": MAX_LEN,
            "vocab_size": VOCAB_SIZE,
        },
        CHECKPOINT_PATH,
    )
    print(f"Model saved to {CHECKPOINT_PATH}")

    return model, device


def load_checkpoint(device):
    # 这里的编码方式是固定的哈希映射，所以只需要保存模型参数即可。
    if not CHECKPOINT_PATH.exists():
        return None

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = Model(checkpoint.get("vocab_size", VOCAB_SIZE)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {CHECKPOINT_PATH}")
    return model


def load_or_train():
    # 默认先加载已有模型；如果你想强制重新训练，设置 FORCE_RETRAIN=1。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    force_retrain = os.getenv("FORCE_RETRAIN", "0") == "1"

    if not force_retrain:
        model = load_checkpoint(device)
        if model is not None:
            return model, device

    model, device = train()
    return model, device


def retrain_and_replace_model():
    """强制重新训练并覆盖当前目录下的模型文件。"""
    return train()


def predict(text, model, device):
    # 推理时同样沿用训练时的编码方式，保证输入格式一致。
    # 如果训练和预测的预处理不一致，模型再好也会判断错。
    ids = encode_text(text)
    ids = torch.tensor([ids], dtype=torch.long).to(device, non_blocking=True)

    # eval() 会关闭 dropout 等训练时行为。
    model.eval()
    with torch.inference_mode():
        output = model(ids)

    # 输出两个类别里概率更高的那个。
    pred = torch.argmax(output, dim=1).item()
    return "正面" if pred == 1 else "负面"


if __name__ == "__main__":
    # 直接执行脚本时，默认强制重训并覆盖 backend/app/models/sentiment_model.pt。
    model, device = retrain_and_replace_model()
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

