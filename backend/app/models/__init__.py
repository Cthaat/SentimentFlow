"""模型层对外导出入口。

约定：
- 在包外统一从 app.models 引用模型相关能力。
- 通过 __all__ 显式声明稳定公开接口。
"""

from app.models.lstm import SentimentLSTM
from app.models.loader import load_model, predict_batch
from app.models.LSTMtraning import retrain_and_replace_model

# 对外公开的模型层 API，供 service 层或启动阶段调用。
__all__ = ["SentimentLSTM", "load_model", "predict_batch", "retrain_and_replace_model"]