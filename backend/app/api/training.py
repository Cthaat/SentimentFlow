"""训练 API 路由。"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse

from app.schemas.training import (
    TrainingJobItem,
    TrainingJobListResponse,
    TrainingStartRequest,
    TrainingStartResponse,
    TrainingStatusResponse,
)
from app.services.training_service import TrainingManager

router = APIRouter()
_manager = TrainingManager()


@router.post("/start", response_model=TrainingStartResponse)
def start_training(req: TrainingStartRequest):
    model_type = req.model_type.strip().lower()
    if model_type not in ("lstm", "bert"):
        raise HTTPException(status_code=400, detail="model_type must be 'lstm' or 'bert'")

    job = _manager.start_training(model_type, req.config)
    return TrainingStartResponse(job_id=job.job_id, status=job.status, model_type=job.model_type)


@router.get("/status/{job_id}", response_model=TrainingStatusResponse)
def get_status(job_id: str):
    job = _manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return TrainingStatusResponse(**job.to_dict())


@router.get("/stream/{job_id}")
async def stream_progress(job_id: str):
    job = _manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def generate():
        last_log_count = 0
        while job.status in ("pending", "running"):
            # 发送增量更新
            data = json.dumps(job.to_dict(), ensure_ascii=False)
            yield f"data: {data}\n\n"

            # 发送新增的日志行
            for log_line in job.logs[last_log_count:]:
                yield f"data: {json.dumps({'type': 'log', 'message': log_line}, ensure_ascii=False)}\n\n"

            last_log_count = len(job.logs)
            await asyncio.sleep(0.5)

        # 发送最终状态
        data = json.dumps(job.to_dict(), ensure_ascii=False)
        yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs", response_model=TrainingJobListResponse)
def list_jobs():
    job_dicts = _manager.list_jobs()
    items = [
        TrainingJobItem(
            job_id=j["job_id"],
            model_type=j["model_type"],
            status=j["status"],
            started_at=j.get("started_at"),
            finished_at=j.get("finished_at"),
        )
        for j in job_dicts
    ]
    return TrainingJobListResponse(jobs=items)


@router.post("/cancel/{job_id}")
def cancel_training(job_id: str):
    ok = _manager.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found or not running")
    return {"ok": True, "job_id": job_id}
