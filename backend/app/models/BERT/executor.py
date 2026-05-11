from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict


_model = None
_device = None
_loaded_ckpt_path: str | None = None


def _ensure_project_root_in_sys_path() -> None:
    """确保 backend 进程可导入项目根目录下的 BERT 包。"""
    project_root = Path(__file__).resolve().parents[4]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def load_model(force_reload: bool = False):
    """加载或训练 BERT 模型（遵循 BERT_FORCE_RETRAIN 逻辑）。"""
    global _model, _device, _loaded_ckpt_path

    current_path = os.getenv("BERT_CHECKPOINT_PATH", "")
    if not force_reload and _model is not None and _loaded_ckpt_path == current_path:
        return _model, _device

    _ensure_project_root_in_sys_path()
    from BERT.pipeline import load_or_train
    from BERT.inference import prepare_inference_model

    _model, _device = load_or_train()
    _model = prepare_inference_model(_model, _device)
    _loaded_ckpt_path = current_path
    return _model, _device


def predict_text(text: str, max_len: int | None = None) -> Dict[str, Any]:
    """执行 BERT 单条文本推理。"""
    model, device = load_model(force_reload=False)

    _ensure_project_root_in_sys_path()
    from BERT.config import MAX_LEN
    from BERT.inference import predict_text as bert_predict_text

    effective_max_len = max_len if max_len is not None else int(os.getenv("BERT_MODEL_MAX_LEN", str(MAX_LEN)))
    return bert_predict_text(text=text, model=model, device=device, max_len=effective_max_len)
