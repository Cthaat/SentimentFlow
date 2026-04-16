"""环境变量工具。

用于显式加载 `.env`，确保脚本在命令行直跑时也能拿到配置。
"""

import os
from pathlib import Path


def load_env_file(env_path: Path, override: bool = True) -> None:
    """显式加载 .env 文件，避免依赖终端注入行为。"""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
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
