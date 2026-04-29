from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.core.config import ensure_backend_env_loaded, get_active_model_config, get_predict_model_type
from app.models.BERT import predict_text as predict_text_with_bert_model
from app.models.LSTM import load_model as load_lstm_model
from app.models.LSTM import predict_batch as predict_batch_with_lstm_model
from app.utils.tokenizer import encode_text


POSITIVE_WORDS = {
	"good",
	"great",
	"excellent",
	"love",
	"awesome",
	"happy",
	"满意",
	"喜欢",
	"推荐",
	"优秀",
}


NEGATIVE_WORDS = {
	"bad",
	"terrible",
	"awful",
	"hate",
	"poor",
	"sad",
	"差",
	"失望",
	"垃圾",
	"讨厌",
}


@dataclass
class PredictResult:
	"""预测结果数据模型。

	用于在 service 层与 API 层之间传递统一结果结构。
	"""
	# 原始输入文本。
	text: str
	# 情感标签：正面/负面/中性。
	label: str
	# 置信分数：范围通常在 0 到 1。
	score: float
	# 结果来源：用于区分规则基线或模型推理。
	source: str


def _keyword_baseline(text: str) -> PredictResult:
	"""关键词基线预测逻辑。

	流程：
	- 统计正向和负向关键词命中次数。
	- 根据命中差值给出标签和一个可解释的置信分数。
	"""
	# 将英文关键词匹配统一为小写比较；中文按原文匹配。
	lowered = text.lower()
	# 统计命中次数，作为情绪倾向的简单信号。
	pos_hits = sum(1 for w in POSITIVE_WORDS if w in lowered or w in text)
	neg_hits = sum(1 for w in NEGATIVE_WORDS if w in lowered or w in text)

	# 没有明显情绪词时返回中性结果。
	if pos_hits == 0 and neg_hits == 0:
		return PredictResult(text=text, label="中性", score=0.5, source="keyword-baseline")
	# 正向命中不小于负向命中，判为正面。
	if pos_hits >= neg_hits:
		score = min(0.55 + 0.1 * (pos_hits - neg_hits + 1), 0.99)
		return PredictResult(text=text, label="正面", score=round(score, 4), source="keyword-baseline")

	# 否则判为负面。
	score = min(0.55 + 0.1 * (neg_hits - pos_hits + 1), 0.99)
	return PredictResult(text=text, label="负面", score=round(score, 4), source="keyword-baseline")


def _predict_with_lstm(text: str) -> PredictResult:
	"""LSTM 推理分支。"""
	model_path = os.getenv("MODEL_PATH", "./app/models/sentiment_model.pt")
	vocab_size = int(os.getenv("MODEL_VOCAB_SIZE", "65536"))
	max_len = int(os.getenv("MODEL_MAX_LEN", "100"))
	# 与根目录旧训练脚本默认结构保持一致。
	embed_dim = int(os.getenv("MODEL_EMBED_DIM", "128"))
	hidden_dim = int(os.getenv("MODEL_HIDDEN_DIM", "256"))
	num_layers = int(os.getenv("MODEL_NUM_LAYERS", "2"))
	dropout = float(os.getenv("MODEL_DROPOUT", "0.5"))
	pad_idx = int(os.getenv("MODEL_PAD_IDX", "0"))

	load_lstm_model(
		model_path=model_path,
		vocab_size=vocab_size,
		embed_dim=embed_dim,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		dropout=dropout,
		pad_idx=pad_idx,
	)

	input_ids = [encode_text(text=text, max_len=max_len, vocab_size=vocab_size)]
	preds, confs = predict_batch_with_lstm_model(input_ids)
	pred = preds[0]
	score = confs[0]
	label = "正面" if pred == 1 else "负面"

	return PredictResult(text=text, label=label, score=round(float(score), 4), source="lstm")


def _predict_with_bert(text: str) -> PredictResult:
	"""BERT 推理分支。"""
	result = predict_text_with_bert_model(text)
	return PredictResult(
		text=result["text"],
		label=result["label"],
		score=round(float(result["confidence"]), 4),
		source="bert",
	)


def _check_model_exists(model_type: str) -> str | None:
	"""检查指定类型的模型文件是否存在。返回错误信息，若存在则返回 None。"""
	if model_type == "lstm":
		model_path = os.getenv("MODEL_PATH", "./app/models/sentiment_model.pt")
		raw = Path(model_path)
		backend_dir = Path(__file__).resolve().parents[2]  # backend/
		candidates = []
		if raw.is_absolute():
			candidates.append(raw)
		else:
			candidates.extend([Path.cwd() / raw, backend_dir / raw])
		if not any(c.exists() for c in candidates):
			return f"LSTM 模型文件不存在（{model_path}），请先在「模型训练」页面训练 LSTM 模型"
	elif model_type == "bert":
		checkpoint_path = os.getenv("BERT_CHECKPOINT_PATH", "./checkpoints/bert")
		raw = Path(checkpoint_path)
		backend_dir = Path(__file__).resolve().parents[2]
		candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, backend_dir / raw]
		if not any((c / "config.json").exists() for c in candidates):
			return f"BERT 模型文件不存在（{checkpoint_path}），请先在「模型训练」页面训练 BERT 模型"
	return None


def predict_text(text: str, model_type: str | None = None) -> PredictResult:
	"""对外统一预测入口。

	策略：
	- 若请求显式指定 model_type，则优先使用。
	- 否则读取 backend/.env 的 PREDICT_MODEL_TYPE（默认 lstm）。
	- 模型文件缺失时直接报错（提示用户先训练），不降级到关键词基线。
	- 模型分支运行时异常时降级到关键词基线。
	"""
	ensure_backend_env_loaded()
	effective_model = (model_type or get_predict_model_type("lstm")).strip().lower()

	# 先检查模型文件是否存在，缺失则直接报错
	missing_msg = _check_model_exists(effective_model)
	if missing_msg:
		raise FileNotFoundError(missing_msg)

	try:
		if effective_model == "bert":
			return _predict_with_bert(text)
		if effective_model == "lstm":
			return _predict_with_lstm(text)
	except FileNotFoundError:
		raise
	except Exception as exc:
		fallback = _keyword_baseline(text)
		fallback.source = f"{fallback.source}-fallback({effective_model}:{type(exc).__name__})"
		return fallback

	# 非法模型类型兜底。
	fallback = _keyword_baseline(text)
	fallback.source = f"{fallback.source}-fallback(invalid-model:{effective_model})"
	return fallback
