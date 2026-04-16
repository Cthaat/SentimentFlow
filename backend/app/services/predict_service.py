from __future__ import annotations

import os
from dataclasses import dataclass

from app.models.loader import load_model, predict_batch
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


def predict_text(text: str) -> PredictResult:
	"""对外统一预测入口。

	当前默认走 LSTM 推理分支。
	"""
	model_path = os.getenv("MODEL_PATH", "./app/models/sentiment_model.pt")
	vocab_size = int(os.getenv("MODEL_VOCAB_SIZE", "65536"))
	max_len = int(os.getenv("MODEL_MAX_LEN", "100"))
	# 与根目录旧训练脚本默认结构保持一致。
	embed_dim = int(os.getenv("MODEL_EMBED_DIM", "128"))
	hidden_dim = int(os.getenv("MODEL_HIDDEN_DIM", "256"))
	num_layers = int(os.getenv("MODEL_NUM_LAYERS", "2"))
	dropout = float(os.getenv("MODEL_DROPOUT", "0.5"))
	pad_idx = int(os.getenv("MODEL_PAD_IDX", "0"))

	load_model(
		model_path=model_path,
		vocab_size=vocab_size,
		embed_dim=embed_dim,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		dropout=dropout,
		pad_idx=pad_idx,
	)

	input_ids = [encode_text(text=text, max_len=max_len, vocab_size=vocab_size)]
	preds, confs = predict_batch(input_ids)
	pred = preds[0]
	score = confs[0]
	label = "正面" if pred == 1 else "负面"

	return PredictResult(text=text, label=label, score=round(float(score), 4), source="lstm")
