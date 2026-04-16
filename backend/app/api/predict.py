from fastapi import APIRouter

from app.schemas.predict import PredictRequest, PredictResponse
from app.services.predict_service import predict_text

router = APIRouter()


@router.post("/")
def predict(req: PredictRequest) -> PredictResponse:
    """情感预测接口。

    约定：
    - 入参使用 PredictRequest 统一校验请求体。
    - 业务逻辑下沉到 service 层，API 层不直接处理模型细节。
    - 出参使用 PredictResponse，保证返回结构稳定。
    """
    # 1) 从请求中取出文本并调用服务层完成预测。
    result = predict_text(req.text, model_type=req.model)

    # 2) 将服务层结果映射为响应模型，避免接口字段漂移。
    return PredictResponse(
        text=result.text,
        label=result.label,
        score=result.score,
        source=result.source,
    )