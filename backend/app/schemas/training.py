"""训练相关 Pydantic 模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TrainingStartRequest(BaseModel):
    model_type: str  # "lstm" | "bert"
    config: dict[str, Any] = {}


class TrainingStartResponse(BaseModel):
    job_id: str
    status: str
    model_type: str


class TrainingProgressSchema(BaseModel):
    current_epoch: int
    total_epochs: int
    loss: float | None
    val_acc: float | None
    val_f1: float | None
    best_f1: float | None


class TrainingStatusResponse(BaseModel):
    job_id: str
    model_type: str
    status: str
    progress: TrainingProgressSchema
    logs: list[str]
    started_at: str | None
    finished_at: str | None
    config: dict[str, Any]
    error: str | None


class TrainingJobItem(BaseModel):
    job_id: str
    model_type: str
    status: str
    started_at: str | None
    finished_at: str | None


class TrainingJobListResponse(BaseModel):
    jobs: list[TrainingJobItem]
