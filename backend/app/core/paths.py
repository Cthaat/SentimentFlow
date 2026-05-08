from __future__ import annotations

import os
from pathlib import Path


def get_backend_dir() -> Path:
    """Return the backend package root directory."""
    return Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """Return the SentimentFlow project root.

    The backend can run either from the repository layout
    `SentimentFlow/backend` or from a container workdir that mirrors it. An
    explicit env var wins, otherwise the parent of backend is used.
    """
    configured = os.getenv("SENTIMENTFLOW_PROJECT_ROOT") or os.getenv("PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return get_backend_dir().parent


def get_models_dir(create: bool = False) -> Path:
    """Return the shared model artifact directory."""
    models_dir = get_project_root() / "models"
    if create:
        models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def normalize_lstm_model_dir(model_path: str | None) -> str | None:
    """Return the containing model directory for a LSTM checkpoint path."""
    if not model_path:
        return None

    path = Path(model_path)
    if path.suffix == ".pt":
        return str(path.parent)
    return str(path)
