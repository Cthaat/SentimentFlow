from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch

_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sentiment_scale import NUM_SENTIMENT_CLASSES

from app.models.LSTM.architecture import SentimentLSTM
from app.models.common import (
    extract_state_dict,
    get_device,
    normalize_state_dict_keys,
    resolve_model_path,
)


_model = None
_device = None
_loaded_ckpt_path: Path | None = None
_loaded_ckpt_mtime: float | None = None
_loaded_num_classes: int | None = None


def load_model(
    model_path: str,
    vocab_size: int,
    num_classes: int = NUM_SENTIMENT_CLASSES,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    pad_idx: int = 0,
) -> SentimentLSTM:
    """加载并缓存 LSTM 模型。"""
    global _model, _device, _loaded_ckpt_path, _loaded_ckpt_mtime, _loaded_num_classes

    app_models_dir = Path(__file__).resolve().parents[1]
    ckpt_path = resolve_model_path(model_path=model_path, app_models_dir=app_models_dir)
    ckpt_mtime = ckpt_path.stat().st_mtime

    if (
        _model is not None
        and _loaded_ckpt_path == ckpt_path
        and _loaded_ckpt_mtime == ckpt_mtime
    ):
        return _model

    _device = get_device()
    ckpt = torch.load(ckpt_path, map_location=_device)
    state_dict = extract_state_dict(ckpt)
    state_dict = normalize_state_dict_keys(state_dict, rename_prefix=("fc.", "classifier."))
    num_classes = _infer_num_classes(state_dict, fallback=num_classes)
    if num_classes != NUM_SENTIMENT_CLASSES:
        raise RuntimeError(
            f"Unsupported LSTM class count: {num_classes}. "
            f"Expected {NUM_SENTIMENT_CLASSES}; retrain the model with the 0-5 score contract."
        )
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        pad_idx=pad_idx,
    )
    model.load_state_dict(state_dict)

    model.to(_device)
    model.eval()

    _model = model
    _loaded_ckpt_path = ckpt_path
    _loaded_ckpt_mtime = ckpt_mtime
    _loaded_num_classes = num_classes
    print(f"Loaded LSTM checkpoint: {ckpt_path} (mtime={ckpt_mtime:.0f})")
    return _model


@torch.no_grad()
def predict_batch(input_ids: List[List[int]]) -> Tuple[List[int], List[float], List[List[float]]]:
    """LSTM 批量推理。"""
    if _model is None:
        raise RuntimeError("LSTM model is not loaded. Call load_model first.")

    tensor = torch.tensor(input_ids, dtype=torch.long, device=_device)
    logits = _model(tensor)

    probs = torch.softmax(logits, dim=-1)
    if _loaded_num_classes != NUM_SENTIMENT_CLASSES:
        raise RuntimeError(f"Unsupported LSTM class count: {_loaded_num_classes}")

    conf, pred = torch.max(probs, dim=-1)
    return pred.tolist(), conf.tolist(), probs.tolist()


def _infer_num_classes(state_dict: dict, fallback: int) -> int:
    weight = state_dict.get("classifier.weight")
    if weight is not None and hasattr(weight, "shape"):
        return int(weight.shape[0])
    return fallback
