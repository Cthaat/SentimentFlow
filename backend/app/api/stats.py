from fastapi import APIRouter

router = APIRouter()

@router.get("/overview")
def overview():
    return {"total_predictions": 0, "positive_ratio": 0.0, "negative_ratio": 0.0}