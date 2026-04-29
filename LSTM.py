"""兼容入口：原 LSTM.py 已拆分到 training 包。

你现在可以按模块阅读：
- training/config.py: 配置与环境变量映射
- training/dataset.py: 流式数据集与容错 CSV 解析
- training/model.py: LSTM 模型结构
- training/trainer.py: 训练循环
- training/evaluate.py: 验证指标
- training/inference.py: 推理
- training/main.py: 入口脚本

为了不破坏已有调用，这里保留旧名称导出。
"""

from training.config import CHECKPOINT_PATH, DEFAULT_CHUNK_SIZE, EPOCHS, MAX_LEN, VOCAB_SIZE
from training.data_sources import build_train_split_and_val_split, get_label_distribution
from training.dataset import CsvStreamDataset
from training.env_utils import load_env_file
from training.evaluate import evaluate
from training.inference import predict_text
from training.main import run as _run
from training.model import SentimentLSTMModel
from training.pipeline import load_checkpoint, load_or_train
from training.text_processing import encode_text, tokenize
from training.trainer import train_model

# 旧 API 兼容别名。
Model = SentimentLSTMModel
train = train_model


def predict(text, model, device):
    """旧函数名兼容：内部转调新推理函数。"""
    return predict_text(text, model, device, max_len=MAX_LEN, vocab_size=VOCAB_SIZE)


if __name__ == "__main__":
    _run()

