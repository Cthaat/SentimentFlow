from __future__ import annotations

import os
from pathlib import Path

from app.core.paths import get_backend_dir, get_models_dir, normalize_lstm_model_dir

# 在导入 transformers 之前抑制 HF Hub 后台线程请求，避免 403 报错
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

_ENV_LOADED = False


def _candidate_env_paths() -> list[Path]:
	return [
		get_backend_dir().parent / ".env",
		get_backend_dir() / ".env",
	]


def load_backend_env(override: bool = False) -> Path:
	"""加载 backend/.env 到进程环境变量。

	- 默认不覆盖已有环境变量。
	- 支持 `KEY=VALUE` 简单格式与注释行。
	"""
	global _ENV_LOADED

	env_path = next((path for path in _candidate_env_paths() if path.exists()), _candidate_env_paths()[0])
	if not env_path.exists():
		_ENV_LOADED = True
		return env_path

	with env_path.open("r", encoding="utf-8") as f:
		for raw_line in f:
			line = raw_line.strip()
			if not line or line.startswith("#") or "=" not in line:
				continue

			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip().strip('"').strip("'")
			if not key:
				continue

			if override or key not in os.environ:
				os.environ[key] = value

	_ENV_LOADED = True
	return env_path


def ensure_backend_env_loaded() -> None:
	"""确保环境变量仅加载一次。"""
	if not _ENV_LOADED:
		load_backend_env(override=False)


def get_predict_model_type(default: str = "lstm") -> str:
	"""获取默认预测模型类型：lstm / bert。"""
	ensure_backend_env_loaded()
	model_type = os.getenv("PREDICT_MODEL_TYPE", default).strip().lower()
	return model_type if model_type in {"lstm", "bert"} else default


# 运行时模型配置（可在不重启的情况下通过 API 切换）
_active_config: dict[str, str] = {}


def get_active_model_config() -> dict[str, str | None]:
    """获取当前活跃模型配置。"""
    ensure_backend_env_loaded()
    return {
        "lstm_path": _active_config.get("lstm_path") or _default_lstm_path(),
        "bert_path": _active_config.get("bert_path") or os.getenv("BERT_CHECKPOINT_PATH"),
        "predict_model_type": _active_config.get("predict_model_type") or get_predict_model_type(),
    }


def _default_lstm_path() -> str | None:
    configured = os.getenv("MODEL_PATH")
    if configured:
        return normalize_lstm_model_dir(configured)

    legacy_path = get_backend_dir() / "app" / "models" / "sentiment_model.pt"
    if legacy_path.exists():
        return str(legacy_path.parent)

    models_dir = get_models_dir(create=False)
    if not models_dir.exists():
        return None

    for model_dir in sorted(models_dir.iterdir(), reverse=True):
        if model_dir.is_dir() and list(model_dir.glob("*.pt")):
            return str(model_dir)
    return None


def set_active_model(model_type: str, model_path: str) -> None:
    """设置活跃模型路径。"""
    ensure_backend_env_loaded()

    from pathlib import Path

    if model_type == "lstm":
        p = Path(model_path)
        # 若传入的是目录，自动查找其中的 .pt 文件
        if p.is_dir():
            pt_files = sorted(p.glob("*.pt"))
            if pt_files:
                model_path = str(pt_files[0])
        display_path = str(Path(model_path).parent if Path(model_path).is_file() else p)
        _active_config["lstm_path"] = display_path
        os.environ["MODEL_PATH"] = model_path
    elif model_type == "bert":
        _active_config["bert_path"] = model_path
        os.environ["BERT_CHECKPOINT_PATH"] = model_path

    # 自动切换预测模型类型
    if _active_config.get("predict_model_type") != model_type:
        _active_config["predict_model_type"] = model_type
        os.environ["PREDICT_MODEL_TYPE"] = model_type

