"""模型管理 API 路由。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from fastapi import APIRouter, HTTPException

from app.core.config import get_active_model_config, set_active_model
from app.schemas.models import (
    ActiveModelResponse,
    ModelInfo,
    ModelListResponse,
    SetActiveModelRequest,
)

router = APIRouter()

# 模型文件搜索目录（相对于 backend 目录）
_MODEL_SEARCH_DIRS: list[Path] = []


def _get_search_dirs() -> list[Path]:
    if _MODEL_SEARCH_DIRS:
        return _MODEL_SEARCH_DIRS

    backend_dir = Path(__file__).resolve().parent.parent  # backend/app
    candidates = [
        backend_dir / "app" / "models",
        backend_dir / "app" / "models" / "LSTM",
        backend_dir / "app" / "models" / "BERT",
    ]
    _MODEL_SEARCH_DIRS.extend([d for d in candidates if d.exists()])
    return _MODEL_SEARCH_DIRS


def _detect_model_type(path: Path) -> str:
    """通过文件内容和名称推断模型类型。"""
    name = path.name.lower()
    if path.is_dir():
        # BERT 模型目录包含 config.json 和 model.safetensors
        if (path / "config.json").exists() and (path / "model.safetensors").exists():
            return "bert"
    if path.suffix == ".pt":
        # 检查 checkpoint 内容
        try:
            ck = torch.load(path, map_location="cpu")
            if isinstance(ck, dict):
                # BERT 的 training_meta.json 有 model_name；LSTM 的 .pt 有 vocab_size
                if "vocab_size" in ck:
                    return "lstm"
        except Exception:
            pass
        return "lstm"
    return "unknown"


def _scan_models() -> list[dict[str, Any]]:
    """扫描模型目录，返回模型信息列表。"""
    models: list[dict[str, Any]] = []
    seen = set()

    for search_dir in _get_search_dirs():
        if not search_dir.exists():
            continue

        for entry in search_dir.iterdir():
            if entry.name.startswith("."):
                continue

            if entry.is_dir():
                if (entry / "config.json").exists() and (entry / "model.safetensors").exists():
                    model_id = hashlib.md5(str(entry).encode()).hexdigest()[:8]
                    if model_id in seen:
                        continue
                    seen.add(model_id)

                    # 读取 training_meta.json
                    meta = {}
                    meta_path = entry / "training_meta.json"
                    if meta_path.exists():
                        import json
                        try:
                            meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        except Exception:
                            pass

                    models.append({
                        "model_id": model_id,
                        "model_type": "bert",
                        "path": str(entry),
                        "best_f1": meta.get("best_val_f1"),
                        "best_epoch": meta.get("best_epoch"),
                    })
            elif entry.suffix == ".pt":
                model_id = hashlib.md5(str(entry).encode()).hexdigest()[:8]
                if model_id in seen:
                    continue
                seen.add(model_id)

                try:
                    ck = torch.load(entry, map_location="cpu")
                    models.append({
                        "model_id": model_id,
                        "model_type": "lstm",
                        "path": str(entry),
                        "best_f1": ck.get("best_val_f1") if isinstance(ck, dict) else None,
                        "best_epoch": ck.get("best_epoch") if isinstance(ck, dict) else None,
                    })
                except Exception:
                    models.append({
                        "model_id": model_id,
                        "model_type": "lstm",
                        "path": str(entry),
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
