from fastapi import APIRouter

router = APIRouter()

@router.get("/users")
def users():
    return {"message": "admin users stub"}