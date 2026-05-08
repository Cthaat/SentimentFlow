from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
	"""预测请求模型。

	约定：
	- text 为必填字段。
	- 通过长度限制拦截空文本和超长输入。
	"""
	# 待分析文本：最短 1 字符，最长 2000 字符。
	text: str = Field(..., min_length=1, max_length=2000, description="Input text")
	# 可选：指定模型类型，未传则使用 backend/.env 的 PREDICT_MODEL_TYPE。
	model: Literal["lstm", "bert"] | None = Field(default=None, description="Model type override")


class PredictResponse(BaseModel):
	"""预测响应模型。

	用于统一前后端约定，保证接口返回字段稳定。
	"""
	# 原始输入文本。
	text: str
	# 情感标签：通常为“正面/负面/中性”。
	label: str
	# 置信分数：范围通常在 0 到 1。
	score: float
	# 结果来源：用于区分基线规则或模型推理。
	source: str
	# 模型名称：当前使用的模型标识（如 lstm_20260429_143025）。
	model_name: str = ""
