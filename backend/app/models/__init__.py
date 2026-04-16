"""模型层对外导出入口。

结构约定：
- app.models.common: 公共能力（路径解析、设备选择等）。
- app.models.LSTM: LSTM 模型结构/执行/训练。
- app.models.BERT: BERT 执行入口。
"""

from app.models.BERT import load_model as load_bert_model
from app.models.BERT import predict_text as predict_bert_text
from app.models.LSTM import SentimentLSTM, load_model, predict_batch
from app.models.LSTMtraning import retrain_and_replace_model

__all__ = [
	"SentimentLSTM",
	"load_model",
	"predict_batch",
	"retrain_and_replace_model",
	"load_bert_model",
	"predict_bert_text",
]