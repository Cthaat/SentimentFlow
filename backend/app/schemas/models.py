"""模型管理相关 Pydantic 模型。"""

from __future__ import annotations

from pydantic import BaseModel


class ModelInfo(BaseModel):
    model_id: str
    model_type: str  # "lstm" | "bert"
    path: str
    size_mb: float | None = None
    best_f1: float | None = None
    best_epoch: int | None = None


class ModelListResponse(BaseModel):
    models: list[ModelInfo]
    active_lstm_path: str | None = None
    active_bert_path: str | None = None


class SetActiveModelRequest(BaseModel):
    model_type: str  # "lstm" | "bert"
    model_path: str


class ActiveModelResponse(BaseModel):
    lstm_path: str | None = None
    bert_path: str | None = None
    predict_model_type: str
