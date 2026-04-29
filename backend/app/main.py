import os
import threading

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


def _preload_models() -> None:
    """后台预加载活跃模型，避免首次预测请求时的冷启动延迟。"""
    try:
        from app.services.predict_service import _check_model_exists, predict_text
        active_type = os.getenv("PREDICT_MODEL_TYPE", "lstm").strip().lower()
        _check_model_exists(active_type)
        # 用一条短文本触发实际加载
        predict_text("preload", model_type=active_type)
        print(f"Preloaded {active_type.upper()} model")
    except FileNotFoundError:
        print(f"No {os.getenv('PREDICT_MODEL_TYPE', 'lstm')} model found, skip preload")
    except Exception as exc:
        print(f"Model preload skipped: {type(exc).__name__}: {exc}")


@app.on_event("startup")
def on_startup():
    threading.Thread(target=_preload_models, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok", "env": str(_loaded_env_path)}