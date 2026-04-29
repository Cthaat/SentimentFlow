"""模型管理 API 路由。

所有训练产出统一存放在项目根目录的 models/ 下，每个模型一个子目录（含时间戳）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.config import get_active_model_config, set_active_model
from app.schemas.models import (
    ActiveModelResponse,
    ModelInfo,
    ModelListResponse,
    SetActiveModelRequest,
)

router = APIRouter()


def _get_models_dir() -> Path:
    """返回统一模型目录（项目根目录下的 models/）。"""
    backend_dir = Path(__file__).resolve().parents[2]  # backend/
    project_root = backend_dir.parent  # SentimentFlow/
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _detect_model_type(path: Path) -> str:
    """通过目录内容推断模型类型。"""
    if path.is_dir():
        if (path / "config.json").exists() and (path / "model.safetensors").exists():
            return "bert"
        # LSTM 训练产出的 .pt 文件
        if list(path.glob("*.pt")):
            return "lstm"
    elif path.suffix == ".pt":
        return "lstm"
    return "unknown"


def _read_meta(path: Path) -> dict[str, Any]:
    """读取模型的元信息。"""
    meta: dict[str, Any] = {}
    meta_path = path / "training_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not meta and path.is_dir():
        # 对于 LSTM，尝试从 .pt 文件中读取
        for pt_file in sorted(path.glob("*.pt")):
            try:
                import torch
                ck = torch.load(pt_file, map_location="cpu")
                if isinstance(ck, dict):
                    meta["best_val_f1"] = ck.get("best_val_f1")
                    meta["best_epoch"] = ck.get("best_epoch")
                break
            except Exception:
                pass
    return meta


def _scan_models() -> list[dict[str, Any]]:
    """扫描 models/ 目录，每个子目录作为一个模型条目。"""
    models: list[dict[str, Any]] = []
    models_dir = _get_models_dir()

    for entry in sorted(models_dir.iterdir()):
        if entry.name.startswith("."):
            continue

        model_type = _detect_model_type(entry)
        if model_type == "unknown":
            continue

        # model_id 使用目录名或文件名（不含扩展名）
        model_id = entry.name if entry.is_dir() else entry.stem
        meta = _read_meta(entry) if entry.is_dir() else {}

        models.append({
            "model_id": model_id,
            "model_type": model_type,
            "path": str(entry),
            "best_f1": meta.get("best_val_f1"),
            "best_epoch": meta.get("best_epoch"),
        })

    return models


@router.get("/", response_model=ModelListResponse)
def list_models():
    models = _scan_models()
    active = get_active_model_config()

    model_infos = [
        ModelInfo(
            model_id=m["model_id"],
            model_type=m["model_type"],
            path=m["path"],
            best_f1=m.get("best_f1"),
            best_epoch=m.get("best_epoch"),
        )
        for m in models
    ]
    return ModelListResponse(
        models=model_infos,
        active_lstm_path=active.get("lstm_path"),
        active_bert_path=active.get("bert_path"),
    )


@router.get("/active", response_model=ActiveModelResponse)
def get_active():
    active = get_active_model_config()
    return ActiveModelResponse(
        lstm_path=active.get("lstm_path"),
        bert_path=active.get("bert_path"),
        predict_model_type=active.get("predict_model_type", "lstm"),
    )


@router.put("/active", response_model=ActiveModelResponse)
def set_active(req: SetActiveModelRequest):
    model_type = req.model_type.strip().lower()
    if model_type not in ("lstm", "bert"):
        raise HTTPException(status_code=400, detail="model_type must be 'lstm' or 'bert'")

    set_active_model(model_type, req.model_path)
    active = get_active_model_config()
    return ActiveModelResponse(
        lstm_path=active.get("lstm_path"),
        bert_path=active.get("bert_path"),
        predict_model_type=active.get("predict_model_type", model_type),
    )


@router.delete("/{model_id}")
def delete_model(model_id: str):
    models = _scan_models()
    target = next((m for m in models if m["model_id"] == model_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Model not found")

    path = Path(target["path"])
    try:
        if path.is_dir():
            import shutil
            shutil.rmtree(path)
        else:
            path.unlink()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {exc}")

    return {"ok": True, "model_id": model_id}
