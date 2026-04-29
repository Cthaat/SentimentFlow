from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def resolve_model_path(model_path: str, app_models_dir: Path) -> Path:
    """解析模型文件路径，兼容不同启动目录。"""
    raw = Path(model_path)
    backend_dir = app_models_dir.parents[1]

    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                Path.cwd() / raw,
                backend_dir / raw,
                app_models_dir / raw.name,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    search_list = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Model checkpoint not found. "
        f"MODEL_PATH={model_path!r}. Tried:\n{search_list}"
    )


def extract_state_dict(ckpt: Any) -> dict:
    """从 checkpoint 中提取模型权重字典。"""
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        return ckpt
    raise TypeError("Unsupported checkpoint format.")


def normalize_state_dict_keys(state_dict: dict, rename_prefix: tuple[str, str] | None = None) -> dict:
    """按需重命名 state_dict key 前缀。"""
    if rename_prefix is None:
        return state_dict

    src_prefix, dst_prefix = rename_prefix
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith(src_prefix):
            normalized[key.replace(src_prefix, dst_prefix, 1)] = value
        else:
            normalized[key] = value
    return normalized


def get_device() -> torch.device:
    """统一设备选择：优先 CUDA。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
