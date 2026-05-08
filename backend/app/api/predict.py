from fastapi import APIRouter, HTTPException

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
    - 模型文件缺失时返回 404 提示用户先训练。
    """
    try:
        result = predict_text(req.text, model_type=req.model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return PredictResponse(
        text=result.text,
        label=result.label,
        score=result.score,
        source=result.source,
        model_name=result.model_name,
    )