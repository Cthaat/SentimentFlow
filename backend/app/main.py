from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import load_backend_env

_loaded_env_path = load_backend_env(override=False)

from app.api import auth, predict, admin, stats, training, models

app = FastAPI(title="Sentiment Analysis API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(predict.router, prefix="/api/predict", tags=["predict"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(stats.router, prefix="/api/stats", tags=["stats"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(models.router, prefix="/api/models", tags=["models"])


@app.get("/health")
def health():
    return {"status": "ok", "env": str(_loaded_env_path)}