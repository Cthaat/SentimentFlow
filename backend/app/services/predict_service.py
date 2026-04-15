from __future__ import annotations

from dataclasses import dataclass


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

	当前使用关键词基线逻辑，后续可在不改 API 层代码的前提下替换为 LSTM 推理。
	"""
	# 保持统一入口，便于未来平滑切换模型实现。
	return _keyword_baseline(text)
