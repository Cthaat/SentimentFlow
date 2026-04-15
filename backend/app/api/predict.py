from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class PredictRequest(BaseModel):
    text: str

@router.post("/")
def predict(req: PredictRequest):
    # 这里先写死，后续接你的 LSTM 推理
    return {"text": req.text, "label": "正面", "score": 0.9}