from fastapi import APIRouter

router = APIRouter()

@router.get("/overview")
def overview():
    return {
        "total_predictions": 0,
        "score_distribution": {str(score): 0 for score in range(6)},
        "average_score": None,
    }
