from pathlib import Path
from typing import List, Tuple

import torch

from app.models.lstm import SentimentLSTM


# 模型单例缓存：避免每次请求重复加载权重。
_model = None
_device = None


def _extract_state_dict(ckpt: object) -> dict:
    """从 checkpoint 中提取模型权重字典。"""
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError("Unsupported checkpoint format.")


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    """兼容旧脚本中 fc 命名，映射到当前 classifier 命名。"""
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("fc."):
            normalized[key.replace("fc.", "classifier.", 1)] = value
        else:
            normalized[key] = value
    return normalized


def get_device() -> torch.device:
    """选择推理设备。

    优先使用 CUDA；若不可用则回退到 CPU。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_path: str,
    vocab_size: int,
    num_classes: int = 2,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.3,
    pad_idx: int = 0,
) -> SentimentLSTM:
    """加载并缓存情感模型。

    约定：
    - 首次调用时构建模型并加载权重。
    - 后续调用直接返回缓存实例。
    - 兼容两种 checkpoint 格式：state_dict 包装 / 纯权重字典。
    """
    global _model, _device
    # 已加载则直接复用，保证服务启动后推理稳定高效。
    if _model is not None:
        return _model

    # 1) 选择设备并实例化网络结构。
    _device = get_device()
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        pad_idx=pad_idx,
    )

    # 2) 加载权重，兼容不同保存格式。
    ckpt = torch.load(Path(model_path), map_location=_device)
    state_dict = _extract_state_dict(ckpt)
    state_dict = _normalize_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)

    # 3) 切换到推理模式并写入缓存。
    model.to(_device)
    model.eval()

    _model = model
    return _model


@torch.no_grad()
def predict_batch(input_ids: List[List[int]]) -> Tuple[List[int], List[float]]:
    """批量推理入口。

    参数 input_ids 为 token id 二维列表，返回：
    - pred: 预测类别 id 列表
    - conf: 每条样本的最大类别概率
    """
    # 防止在模型未初始化时误调用推理。
    if _model is None:
        raise RuntimeError("Model is not loaded. Call load_model first.")

    # 将输入转换为张量并放到与模型一致的设备。
    tensor = torch.tensor(input_ids, dtype=torch.long, device=_device)
    logits = _model(tensor)

    # softmax 后取最大概率及对应类别。
    probs = torch.softmax(logits, dim=-1)
    conf, pred = torch.max(probs, dim=-1)

    # 转换回列表返回，方便上层调用。
    # pred 是类别 id 列表，conf 是对应的置信分数列表。
    return pred.tolist(), conf.tolist()