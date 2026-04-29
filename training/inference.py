"""推理模块。"""

from __future__ import annotations

import torch

from .text_processing import encode_text


def predict_text(text: str, model, device: torch.device, max_len: int, vocab_size: int):
    """执行单条文本情感预测。"""
    ids = encode_text(text, max_len=max_len, vocab_size=vocab_size)
    ids = torch.tensor([ids], dtype=torch.long).to(device, non_blocking=True)

    model.eval()
    with torch.inference_mode():
        output = model(ids)
        probs = torch.softmax(output, dim=1)[0]

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
