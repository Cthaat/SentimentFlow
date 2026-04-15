import os
import zlib
from pathlib import Path
from typing import Tuple
from copy import deepcopy

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset


# 序列长度、训练轮数和词表大小都是训练时最核心的三个超参数。
MAX_LEN = 100
EPOCHS = 10
VOCAB_SIZE = 65536
DEFAULT_CHUNK_SIZE = 4096
CHECKPOINT_PATH = "sentiment_model.pt"


def load_env_file(env_path: Path, override: bool = True) -> None:
    """显式加载 .env 文件，避免依赖终端注入行为。"""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue

        if override or key not in os.environ:
            os.environ[key] = value


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
    # 同时支持两种输入：
    # 1) CSV 文件路径（流式分块读取）
    # 2) HuggingFace Dataset 对象（按索引遍历）
    def __init__(self, source, chunk_size=DEFAULT_CHUNK_SIZE):
        super().__init__()
        self.source = source
        self.chunk_size = chunk_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        # 情况 A：CSV 文件路径，按 chunk 流式读取。
        if isinstance(self.source, (str, Path)):
            chunk_reader = pd.read_csv(
                self.source,
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
                    yield torch.tensor(encode_text(text), dtype=torch.long), torch.tensor(label, dtype=torch.long)

            return

        # 情况 B：HuggingFace Dataset（或任何支持 __len__ / __getitem__ 的对象）。
        if hasattr(self.source, "__len__") and hasattr(self.source, "__getitem__"):
            total = len(self.source)
            for idx in range(worker_id, total, num_workers):
                row = self.source[idx]
                text = str(row.get("text", ""))
                label = int(row.get("label", 0))
                yield torch.tensor(encode_text(text), dtype=torch.long), torch.tensor(label, dtype=torch.long)

            return

        raise TypeError(f"Unsupported dataset source type: {type(self.source)!r}")


def build_train_split_and_val_split():
    """加载并拼接多个训练/验证数据集。"""
    from datasets import concatenate_datasets, load_dataset

    dataset_names = [
        item.strip()
        for item in os.getenv("TRAIN_DATASETS", "lansinuote/ChnSentiCorp").split(",")
        if item.strip()
    ]

    train_splits = []
    val_splits = []
    for name in dataset_names:
        ds = load_dataset(name)
        train_splits.append(ds["train"])
        if "validation" in ds:
            val_splits.append(ds["validation"])
        elif "test" in ds:
            val_splits.append(ds["test"])
        else:
            raise ValueError(f"Dataset {name} has neither validation nor test split.")

    train_split = train_splits[0] if len(train_splits) == 1 else concatenate_datasets(train_splits)
    val_split = val_splits[0] if len(val_splits) == 1 else concatenate_datasets(val_splits)
    train_split = train_split.shuffle(seed=42)
    val_split = val_split.shuffle(seed=42)

    max_samples = int(os.getenv("TRAIN_MAX_SAMPLES", "0"))
    if max_samples > 0:
        max_samples = min(max_samples, len(train_split))
        train_split = train_split.select(range(max_samples))

    max_val_samples = int(os.getenv("TRAIN_MAX_VAL_SAMPLES", "0"))
    if max_val_samples > 0:
        max_val_samples = min(max_val_samples, len(val_split))
        val_split = val_split.select(range(max_val_samples))

    return dataset_names, train_split, val_split


def get_label_distribution(split) -> Tuple[int, int]:
    labels = split["label"]
    neg = sum(1 for x in labels if int(x) == 0)
    pos = sum(1 for x in labels if int(x) == 1)
    return neg, pos


@torch.no_grad()
def evaluate(model: nn.Module, split, device: torch.device, batch_size: int = 512) -> Tuple[float, float]:
    """在验证集上评估 accuracy 与 F1。"""
    eval_loader = DataLoader(
        CsvStreamDataset(split),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model.eval()
    tp = fp = fn = tn = 0
    for batch_x, batch_y in eval_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        logits = model(batch_x)
        pred = torch.argmax(logits, dim=1)

        tp += int(((pred == 1) & (batch_y == 1)).sum().item())
        tn += int(((pred == 0) & (batch_y == 0)).sum().item())
        fp += int(((pred == 1) & (batch_y == 0)).sum().item())
        fn += int(((pred == 0) & (batch_y == 1)).sum().item())

    total = max(1, tp + tn + fp + fn)
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return accuracy, f1


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
    # - TRAIN_DATASETS：多个 HF 数据集，逗号分隔
    #   例如：lansinuote/ChnSentiCorp,dataset2
    # - TRAIN_MAX_SAMPLES / TRAIN_MAX_VAL_SAMPLES：可选截断样本数
    batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "256" if device.type == "cuda" else "128"))
    num_workers = int(os.getenv("TRAIN_NUM_WORKERS", "1" if device.type == "cuda" else "0"))
    grad_accum_steps = int(os.getenv("TRAIN_ACCUM_STEPS", "1"))
    chunk_size = int(os.getenv("TRAIN_CHUNK_SIZE", str(max(batch_size * 8, DEFAULT_CHUNK_SIZE))))

    dataset_names, train_split, val_split = build_train_split_and_val_split()
    neg_count, pos_count = get_label_distribution(train_split)
    total_count = max(1, neg_count + pos_count)
    print(f"Datasets: {', '.join(dataset_names)}")
    print(
        f"Train samples: {total_count}, neg={neg_count}, pos={pos_count}, "
        f"pos_ratio={pos_count / total_count:.4f}"
    )
    print(
        f"Training config: batch_size={batch_size}, grad_accum_steps={grad_accum_steps}, "
        f"num_workers={num_workers}"
    )

    # chunk_size 越大，每次读盘越少，但单次 CPU 内存峰值也会更高。
    # 如果你机器内存偏小，就不要把它设得太大。
    dataset = CsvStreamDataset(train_split, chunk_size=chunk_size)
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
    # 当类别不平衡时，可打开加权损失降低“全判单一类别”的风险。
    use_weighted_loss = os.getenv("TRAIN_WEIGHTED_LOSS", "1") == "1"
    if use_weighted_loss and neg_count > 0 and pos_count > 0:
        class_weights = torch.tensor(
            [total_count / (2 * neg_count), total_count / (2 * pos_count)],
            dtype=torch.float32,
            device=device,
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print(
            f"Using weighted loss: neg_w={class_weights[0].item():.4f}, "
            f"pos_w={class_weights[1].item():.4f}"
        )
    else:
        loss_fn = nn.CrossEntropyLoss()

    # AdamW 是比较稳的优化器，通常比普通 SGD 更容易收敛。
    learning_rate = float(os.getenv("TRAIN_LR", "0.0005"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 混合精度训练需要 GradScaler，防止 float16 下梯度太小而下溢。
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
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

        # 每个 epoch 后在验证集评估一次，避免只看 loss 误判效果。
        val_acc, val_f1 = evaluate(model, val_split, device=device)
        avg_loss = (total_loss / max(1, batch_count)).item()
        print(
            f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, "
            f"ValAcc: {val_acc:.4f}, ValF1: {val_f1:.4f}"
        )

        # evaluate() 内部会切到 eval 模式，这里切回 train，确保下一轮可正常反向传播。
        model.train()

        # 按验证集 F1 保存最优 checkpoint。
        if val_f1 >= best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_state_dict = deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "max_len": MAX_LEN,
                    "vocab_size": VOCAB_SIZE,
                    "best_val_f1": round(best_f1, 6),
                    "best_epoch": best_epoch,
                },
                CHECKPOINT_PATH,
            )
            print(f"Best model updated at epoch {epoch + 1}, ValF1={best_f1:.4f}")

    # 训练结束后回滚到验证集最优权重，避免“最后一轮退化”影响最终预测。
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best in-memory checkpoint from epoch {best_epoch}, ValF1={best_f1:.4f}")

    print(f"Training finished. Best model saved to {CHECKPOINT_PATH}")

    return model, device


def load_checkpoint(device):
    # 这里的编码方式是固定的哈希映射，所以只需要保存模型参数即可。
    if not os.path.exists(CHECKPOINT_PATH):
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


def predict(text, model, device):
    # 推理时同样沿用训练时的编码方式，保证输入格式一致。
    # 如果训练和预测的预处理不一致，模型再好也会判断错。
    ids = encode_text(text)
    ids = torch.tensor([ids], dtype=torch.long).to(device, non_blocking=True)

    # eval() 会关闭 dropout 等训练时行为。
    model.eval()
    with torch.inference_mode():
        output = model(ids)
        probs = torch.softmax(output, dim=1)[0]

    # 输出两个类别里概率更高的那个。
    neg_score = float(probs[0].item())
    pos_score = float(probs[1].item())
    pred = 1 if pos_score >= neg_score else 0
    label = "正面" if pred == 1 else "负面"

    return {
        "text": text,
        "label": label,
        "confidence": round(max(neg_score, pos_score), 6),
        "negative_score": round(neg_score, 6),
        "positive_score": round(pos_score, 6),
    }


if __name__ == "__main__":
    # 直接读取项目根目录 .env，确保命令行直跑脚本也能拿到配置。
    env_path = Path(__file__).resolve().parent / ".env"
    load_env_file(env_path, override=True)

    # 默认不重训；当 FORCE_RETRAIN=1 时才强制重训并覆盖 checkpoint。
    force_retrain = os.getenv("FORCE_RETRAIN", "1") == "1"
    print(f"Effective FORCE_RETRAIN={os.getenv('FORCE_RETRAIN', '0')} (from {env_path})")
    if force_retrain:
        print("FORCE_RETRAIN=1 -> Retraining model and overwriting checkpoint.")
        model, device = train()
    else:
        model, device = load_or_train()
    samples = [
        "这个电影太棒了！",
        "这个电影太差了！",
        "这个手机的电池续航很不错！",
        "这个手机的屏幕质量很差！",
        "这个电脑的性能非常给力！",
        "这个电脑的系统经常崩溃！",
        "这个耳机的音质非常清晰！",
        "这个耳机的做工很糟糕！",
        "这个屏幕的显示效果非常好！",
        "这个屏幕的亮度很差！",
        "这个系统的界面非常流畅！",
        "这个系统的响应速度很慢！",
        "这个电池的续航时间非常长！",
        "这个电池的充电速度很慢！",
        "这个物流的配送速度非常快！",
        "这个物流的配送速度很慢！",
        "这个客服的服务态度非常好！",
        "这个客服的服务态度很差！",
        "这个包装的质量非常好！",
        "这个包装的质量很差！",
        "这个做工的质量非常给力！",
        "这个做工的质量很差！",
        # 一些英文数据集
        "I love this product! It's amazing.",
        "I hate this product! It's terrible.",
        "This is a great movie!",
        "This is a terrible movie!",
        "The battery life is amazing!",
        "The battery life is awful!",
        "The screen quality is fantastic!",
        "The screen quality is poor!",
        "The performance is excellent!",
        "The performance is terrible!",
    ]

    for text in samples:
        result = predict(text, model, device)
        print(
            f"{result['text']} -> {result['label']} "
            f"(neg={result['negative_score']:.6f}, pos={result['positive_score']:.6f}, conf={result['confidence']:.6f})"
        )

