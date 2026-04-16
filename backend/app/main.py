from fastapi import FastAPI

from app.core.config import load_backend_env

_loaded_env_path = load_backend_env(override=False)

from app.api import auth, predict, admin, stats

app = FastAPI(title="Sentiment Analysis API", version="0.1.0")

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(predict.router, prefix="/api/predict", tags=["predict"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(stats.router, prefix="/api/stats", tags=["stats"])


@app.get("/health")
def health():
    return {"status": "ok", "env": str(_loaded_env_path)}