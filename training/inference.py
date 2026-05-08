"""推理模块。"""

from __future__ import annotations

import torch

from sentiment_scale import probabilities_to_prediction

from .text_processing import encode_text


def predict_text(text: str, model, device: torch.device, max_len: int, vocab_size: int):
    """执行单条文本情感预测。"""
    ids = encode_text(text, max_len=max_len, vocab_size=vocab_size)
    ids = torch.tensor([ids], dtype=torch.long).to(device, non_blocking=True)

    model.eval()
    with torch.inference_mode():
        output = model(ids)
        probs = torch.softmax(output, dim=1)[0]

    prediction = probabilities_to_prediction(probs.detach().cpu().tolist())
    return {
        "text": text,
        **prediction,
    }
