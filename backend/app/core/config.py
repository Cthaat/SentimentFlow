from __future__ import annotations

import os
from pathlib import Path


_ENV_LOADED = False


def load_backend_env(override: bool = False) -> Path:
	"""加载 backend/.env 到进程环境变量。

	- 默认不覆盖已有环境变量。
	- 支持 `KEY=VALUE` 简单格式与注释行。
	"""
	global _ENV_LOADED

	env_path = Path(__file__).resolve().parents[2] / ".env"
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

