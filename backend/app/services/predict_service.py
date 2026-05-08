from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.config import ensure_backend_env_loaded, get_predict_model_type
from app.core.paths import get_models_dir
from app.models.BERT import predict_text as predict_text_with_bert_model
from app.models.LSTM import load_model as load_lstm_model
from app.models.LSTM import predict_batch as predict_batch_with_lstm_model
from app.utils.tokenizer import encode_text

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
	sys.path.insert(0, str(_project_root))

from sentiment_scale import (
	NUM_SENTIMENT_CLASSES,
	score_to_display_label,
	score_to_label,
	score_to_reasoning,
)


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
	"糟糕",
	"不推荐",
	"不能接受",
	"再也不",
}


@dataclass
class PredictResult:
	"""预测结果数据模型。

	用于在 service 层与 API 层之间传递统一结果结构。
	"""
	# 原始输入文本。
	text: str
	# 0-5 情感评分。
	score: int
	# 情感标签：如 extremely_negative / neutral / extremely_positive。
	label: str
	# 中文展示标签。
	label_zh: str
	# 置信分数：范围通常在 0 到 1。
	confidence: float
	# 6 个评分类别的概率，索引即 score。
	probabilities: list[float]
	# 简短可解释说明。
	reasoning: str
	# 结果来源：用于区分规则基线或模型推理。
	source: str
	# 模型名称：当前使用的模型标识。
	model_name: str = ""


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
		return _make_result(text, 3, 0.5, "keyword-baseline", "关键词基线")
	# 正向命中不小于负向命中，按命中差值映射为 4/5。
	if pos_hits >= neg_hits:
		sentiment_score = 5 if pos_hits - neg_hits >= 2 else 4
		confidence = min(0.55 + 0.1 * (pos_hits - neg_hits + 1), 0.99)
		return _make_result(text, sentiment_score, confidence, "keyword-baseline", "关键词基线")

	# 否则按命中差值映射为极端/明显负面。
	sentiment_score = 0 if neg_hits - pos_hits >= 2 else 1
	confidence = min(0.55 + 0.1 * (neg_hits - pos_hits + 1), 0.99)
	return _make_result(text, sentiment_score, confidence, "keyword-baseline", "关键词基线")


def _make_result(
	text: str,
	score: int,
	confidence: float,
	source: str,
	model_name: str,
	probabilities: list[float] | None = None,
) -> PredictResult:
	if probabilities is None:
		confidence = max(0.0, min(1.0, float(confidence)))
		remainder = max(0.0, 1.0 - confidence)
		background = remainder / max(1, NUM_SENTIMENT_CLASSES - 1)
		probabilities = [background for _ in range(NUM_SENTIMENT_CLASSES)]
		probabilities[score] = confidence
	return PredictResult(
		text=text,
		score=score,
		label=score_to_label(score),
		label_zh=score_to_display_label(score),
		confidence=round(float(confidence), 4),
		probabilities=[round(float(value), 6) for value in probabilities],
		reasoning=score_to_reasoning(score),
		source=source,
		model_name=model_name,
	)


def _predict_with_lstm(text: str) -> PredictResult:
	"""LSTM 推理分支。"""
	model_path = os.getenv("MODEL_PATH", "")
	# 兼容目录路径：自动查找其中的 .pt 文件
	p = Path(model_path)
	if p.is_dir():
		pt_files = sorted(p.glob("*.pt"))
		model_path = str(pt_files[0]) if pt_files else model_path
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
	preds, confs, probs = predict_batch_with_lstm_model(input_ids)
	pred = preds[0]
	confidence = confs[0]

	model_dir = Path(model_path).parent.name if Path(model_path).parent.name.startswith("lstm_") else Path(model_path).stem
	return _make_result(
		text=text,
		score=int(pred),
		confidence=float(confidence),
		source="lstm",
		model_name=model_dir,
		probabilities=probs[0],
	)


def _predict_with_bert(text: str) -> PredictResult:
	"""BERT 推理分支。"""
	result = predict_text_with_bert_model(text)
	model_name = Path(os.getenv("BERT_CHECKPOINT_PATH", "")).name
	return PredictResult(
		text=result["text"],
		score=int(result["score"]),
		label=result["label"],
		label_zh=result.get("label_zh", score_to_display_label(int(result["score"]))),
		confidence=round(float(result["confidence"]), 4),
		probabilities=[float(value) for value in result.get("probabilities", [])],
		reasoning=result.get("reasoning", score_to_reasoning(int(result["score"]))),
		source="bert",
		model_name=model_name,
	)


def _get_models_dir() -> Path:
	return get_models_dir(create=False)


def _scan_models_of_type(model_type: str) -> list[Path]:
	"""扫描 models/ 目录，返回指定类型模型的路径列表（按名称倒序，最新的在前）。"""
	models_dir = _get_models_dir()
	if not models_dir.exists():
		return []

	result = []
	for entry in sorted(models_dir.iterdir(), reverse=True):
		if entry.name.startswith("."):
			continue
		if model_type == "lstm":
			if entry.is_dir() and list(entry.glob("*.pt")):
				result.append(entry)
		elif model_type == "bert":
			has_bert_weights = any(
				(entry / filename).exists()
				for filename in ("model.safetensors", "pytorch_model.bin")
			)
			if entry.is_dir() and (entry / "config.json").exists() and has_bert_weights:
				result.append(entry)
	return result


def _check_model_exists(model_type: str) -> str | None:
	"""检查指定类型的模型文件是否存在。返回错误信息，若存在则返回 None。"""
	# 先检查当前活跃模型路径
	if model_type == "lstm":
		model_path = os.getenv("MODEL_PATH", "")
		if model_path:
			raw = Path(model_path)
			if raw.is_file() and raw.exists():
				return None
			if raw.is_dir():
				pt_files = sorted(raw.glob("*.pt"))
				if pt_files:
					os.environ["MODEL_PATH"] = str(pt_files[0])
					return None
	elif model_type == "bert":
		checkpoint_path = os.getenv("BERT_CHECKPOINT_PATH", "")
		if checkpoint_path:
			raw = Path(checkpoint_path)
			if raw.is_dir() and (raw / "config.json").exists():
				return None

	# 活跃路径不可用，扫描 models/ 目录
	available = _scan_models_of_type(model_type)
	if available:
		# 自动激活最新的模型
		latest = str(available[0])
		if model_type == "lstm":
			pt_files = sorted(available[0].glob("*.pt"))
			os.environ["MODEL_PATH"] = str(pt_files[0])
		else:
			os.environ["BERT_CHECKPOINT_PATH"] = latest
		return None

	return f"{model_type.upper()} 模型不存在，请先在「模型训练」页面训练 {model_type.upper()} 模型"


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
	if effective_model not in {"lstm", "bert"}:
		fallback = _keyword_baseline(text)
		fallback.source = f"{fallback.source}-fallback(invalid-model:{effective_model})"
		return fallback

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
