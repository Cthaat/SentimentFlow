"""推理模块。"""

from __future__ import annotations

import torch

from sentiment_scale import probabilities_to_prediction

from .config import MAX_LEN
from .text_processing import encode_text


def predict_text(text: str, model, device: torch.device, max_len: int = MAX_LEN, vocab_size: int | None = None):
    """执行单条文本情感预测。"""
    encoded = encode_text(text, max_len=max_len)
    input_ids = encoded["input_ids"].unsqueeze(0).to(device, non_blocking=True)
    attention_mask = encoded["attention_mask"].unsqueeze(0).to(device, non_blocking=True)

    model.eval()
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)[0]

    prediction = probabilities_to_prediction(probs.detach().cpu().tolist())
    return {
        "text": text,
        **prediction,
    }
