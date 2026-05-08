"""BERT 推理与导出模块。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from sentiment_scale import (
    NUM_SENTIMENT_CLASSES,
    probabilities_to_prediction,
    score_to_display_label,
    score_to_label,
    score_to_reasoning,
)

from .config import MAX_LEN, get_model_name
from .text_processing import get_tokenizer


def predict_text(
    text: str,
    model,
    device: torch.device,
    max_len: int = MAX_LEN,
    vocab_size: int | None = None,
):
    """执行单条文本情感预测。"""
    return predict_batch([text], model, device, max_len=max_len, batch_size=1)[0]


def predict_batch(
    texts: Iterable[str],
    model,
    device: torch.device,
    *,
    max_len: int = MAX_LEN,
    batch_size: int | None = None,
) -> list[dict]:
    """批量推理，使用动态 padding 和 tokenizer cache 降低 CPU 开销。"""
    text_list = [str(text or "") for text in texts]
    if not text_list:
        return []

    batch_size = batch_size or max(1, int(os.getenv("BERT_INFERENCE_BATCH_SIZE", "64")))
    tokenizer = get_tokenizer(get_model_name())
    predictions: list[dict] = []

    model.eval()
    for start in range(0, len(text_list), batch_size):
        batch_texts = text_list[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(device, non_blocking=True)

        with torch.inference_mode():
            outputs = _forward_for_inference(model, input_ids=input_ids, attention_mask=attention_mask)

        for text, prediction in zip(batch_texts, outputs_to_predictions(outputs)):
            predictions.append({"text": text, **prediction})
    return predictions


def _forward_for_inference(model, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    except TypeError:
        return model(input_ids=input_ids, attention_mask=attention_mask)


def outputs_to_predictions(outputs: torch.Tensor | dict[str, torch.Tensor]) -> list[dict]:
    """Convert model outputs to the stable 0-5 API contract.

    Hybrid checkpoints use ordinal/regression heads to choose the final score.
    The returned probability vector is still the 6-class softmax, so downstream
    API/UI contracts remain unchanged.
    """
    output_dict = _as_output_dict(outputs)
    probabilities = output_probabilities(output_dict)
    scores = prediction_scores_from_outputs(output_dict, probabilities=probabilities)
    predictions: list[dict] = []
    for probability_row, score_value in zip(probabilities.detach().cpu(), scores.detach().cpu()):
        score = int(torch.round(score_value).clamp(0, NUM_SENTIMENT_CLASSES - 1).item())
        base = probabilities_to_prediction(probability_row.tolist())
        if score != int(base["score"]):
            base.update(
                {
                    "score": score,
                    "label": score_to_label(score),
                    "label_zh": score_to_display_label(score),
                    "confidence": round(float(probability_row[score]), 6),
                    "reasoning": score_to_reasoning(score),
                }
            )
        predictions.append(base)
    return predictions


def output_probabilities(outputs: torch.Tensor | dict[str, torch.Tensor], *, temperature: float | None = None) -> torch.Tensor:
    """Return calibrated class probabilities from tensor or hybrid outputs."""
    output_dict = _as_output_dict(outputs)
    logits = output_dict["logits"]
    scale = float(temperature if temperature is not None else os.getenv("BERT_INFERENCE_TEMPERATURE", "1.0"))
    scale = max(scale, 1e-6)
    return torch.softmax(logits.float() / scale, dim=1)


def prediction_scores_from_outputs(
    outputs: torch.Tensor | dict[str, torch.Tensor],
    *,
    probabilities: torch.Tensor | None = None,
) -> torch.Tensor:
    """Predict continuous 0-5 scores from classification + ordinal heads."""
    output_dict = _as_output_dict(outputs)
    if probabilities is None:
        probabilities = output_probabilities(output_dict)

    score_values = torch.arange(NUM_SENTIMENT_CLASSES, device=probabilities.device, dtype=probabilities.dtype)
    class_expected = (probabilities * score_values).sum(dim=1)
    components: list[torch.Tensor] = [class_expected]
    weights = [float(os.getenv("BERT_INFERENCE_CLASS_WEIGHT", "0.7"))]

    ordinal_logits = output_dict.get("ordinal_logits")
    if ordinal_logits is not None and os.getenv("BERT_INFERENCE_USE_ORDINAL_HEAD", "1") == "1":
        ordinal_expected = torch.sigmoid(ordinal_logits.float()).sum(dim=1)
        components.append(ordinal_expected.to(probabilities.device))
        weights.append(float(os.getenv("BERT_INFERENCE_ORDINAL_WEIGHT", "0.2")))

    score_prediction = output_dict.get("score")
    if score_prediction is not None and os.getenv("BERT_INFERENCE_USE_REGRESSION_HEAD", "1") == "1":
        components.append(score_prediction.float().to(probabilities.device).clamp(0, NUM_SENTIMENT_CLASSES - 1))
        weights.append(float(os.getenv("BERT_INFERENCE_REGRESSION_WEIGHT", "0.1")))

    positive_weights = [max(0.0, weight) for weight in weights]
    weight_total = sum(positive_weights)
    if weight_total <= 0:
        return torch.argmax(probabilities, dim=1).float()

    score = torch.zeros_like(class_expected)
    for component, weight in zip(components, positive_weights):
        score = score + component * (weight / weight_total)
    return score.clamp(0, NUM_SENTIMENT_CLASSES - 1)


def predicted_classes_from_outputs(outputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """Return discrete 0-5 classes for evaluation."""
    scores = prediction_scores_from_outputs(outputs)
    return torch.round(scores).clamp(0, NUM_SENTIMENT_CLASSES - 1).long()


def prepare_inference_model(model, device: torch.device, *, compile_model: bool | None = None, quantize: bool | None = None):
    """Apply optional inference-time optimizations and return an eval model."""
    model = model.to(device).eval()
    should_quantize = (
        quantize
        if quantize is not None
        else os.getenv("BERT_DYNAMIC_QUANTIZATION", "0") == "1"
    )
    if should_quantize and device.type == "cpu":
        try:
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8).eval()
            print("Dynamic int8 quantization enabled for CPU BERT inference.")
        except Exception as exc:
            print(f"Warning: dynamic quantization disabled: {type(exc).__name__}: {exc}")

    should_compile = (
        compile_model
        if compile_model is not None
        else os.getenv("BERT_INFERENCE_TORCH_COMPILE", "0") == "1"
    )
    if should_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=os.getenv("BERT_INFERENCE_COMPILE_MODE", "reduce-overhead"))
            print("torch.compile enabled for BERT inference.")
        except Exception as exc:
            print(f"Warning: inference torch.compile disabled: {type(exc).__name__}: {exc}")
    return model


def export_onnx(
    model,
    output_path: str | Path,
    device: torch.device,
    *,
    max_len: int = MAX_LEN,
    opset_version: int = 17,
) -> Path:
    """Export class logits to ONNX with dynamic batch/sequence axes."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _OnnxLogitsWrapper(model).to(device).eval()
    dummy_input_ids = torch.ones((1, max_len), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((1, max_len), dtype=torch.long, device=device)
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        str(output),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=opset_version,
    )
    return output


def _as_output_dict(outputs: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(outputs, dict):
        return outputs
    return {"logits": outputs}


class _OnnxLogitsWrapper(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = _forward_for_inference(self.model, input_ids=input_ids, attention_mask=attention_mask)
        return _as_output_dict(outputs)["logits"]
