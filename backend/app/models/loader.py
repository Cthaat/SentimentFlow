"""兼容入口：LSTM 推理加载器。

新实现已迁移到 `app.models.LSTM.executor`，此文件仅保留旧导入路径兼容。
"""

from app.models.LSTM.executor import load_model, predict_batch

__all__ = ["load_model", "predict_batch"]