"""训练任务管理器。

在后台线程中运行 LSTM/BERT 训练，捕获进度并提供 SSE 兼容的状态查询。
训练产出统一保存到项目根目录 models/{type}_{timestamp}/ 下。
"""

from __future__ import annotations

import io
import os
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import ensure_backend_env_loaded, set_active_model
from app.core.paths import get_models_dir, get_project_root

# 确保 backend 进程可以导入项目根目录下的 training 和 BERT 包
_project_root = get_project_root()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@dataclass
class TrainingProgress:
    stage: str = "queued"
    stage_detail: str | None = None
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    loss: float | None = None
    val_acc: float | None = None
    val_f1: float | None = None
    best_f1: float | None = None


@dataclass
class TrainingJob:
    job_id: str
    model_type: str  # "lstm" | "bert"
    status: str  # "pending" | "running" | "completed" | "failed" | "cancelled"
    progress: TrainingProgress = field(default_factory=TrainingProgress)
    logs: list[str] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    model_path: str | None = None  # 训练产出的模型路径

    _cancel_flag: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "model_type": self.model_type,
            "status": self.status,
            "progress": {
                "stage": self.progress.stage,
                "stage_detail": self.progress.stage_detail,
                "current_epoch": self.progress.current_epoch,
                "total_epochs": self.progress.total_epochs,
                "current_step": self.progress.current_step,
                "total_steps": self.progress.total_steps,
                "loss": self.progress.loss,
                "val_acc": self.progress.val_acc,
                "val_f1": self.progress.val_f1,
                "best_f1": self.progress.best_f1,
            },
            "logs": self.logs[-50:],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config": self.config,
            "error": self.error,
            "model_path": self.model_path,
        }


_EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+)[,.]?\s*Loss:\s*([\d.]+)[,.]?\s*ValAcc:\s*([\d.]+)[,.]?\s*ValMacroF1:\s*([\d.]+)"
)
_STEP_PATTERN = re.compile(
    r"Epoch\s+(\d+)/(\d+),\s*Step\s+(\d+)/(\d+),\s*Loss:\s*([\d.]+)"
)
_BEST_MODEL_PATTERN = re.compile(r"Best model updated at epoch \d+, ValMacroF1=([\d.]+)")


class TrainingManager:
    """训练任务单例管理器。"""

    _instance: TrainingManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> TrainingManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._jobs: dict[str, TrainingJob] = {}
                    cls._instance._threads: dict[str, threading.Thread] = {}
                    cls._instance._state_lock = threading.Lock()
        return cls._instance

    def start_training(self, model_type: str, config: dict | None = None) -> TrainingJob:
        """启动训练并返回 job 对象。"""
        ensure_backend_env_loaded()

        if config is None:
            config = {}

        with self._state_lock:
            active_job = next(
                (j for j in self._jobs.values() if j.status in ("pending", "running")),
                None,
            )
            if active_job is not None:
                raise RuntimeError(f"Training job {active_job.job_id} is already {active_job.status}")

            job = TrainingJob(
                job_id=uuid.uuid4().hex[:12],
                model_type=model_type,
                status="pending",
                config=config,
            )
            self._jobs[job.job_id] = job

        thread = threading.Thread(
            target=self._run_training,
            args=(job,),
            daemon=True,
            name=f"training-{job.job_id}",
        )
        self._threads[job.job_id] = thread
        thread.start()

        return job

    def get_job(self, job_id: str) -> TrainingJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict]:
        return [j.to_dict() for j in self._jobs.values()]

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status not in ("pending", "running"):
            return False
        job._cancel_flag.set()
        return True

    def _run_training(self, job: TrainingJob) -> None:
        """在后台线程执行训练。"""
        job.status = "running"
        job.progress.stage = "starting"
        job.progress.stage_detail = "训练任务已创建，正在准备运行环境"
        job.started_at = datetime.now(timezone.utc).isoformat()
        job.logs.append(f"[INFO] Training job {job.job_id} accepted")
        time.sleep(0.5)

        # 将用户配置写入环境变量（仅影响当前线程的子进程）
        env_updates = {key: str(value) for key, value in job.config.items()}

        # 创建带时间戳的模型保存路径，统一放到 models/ 目录下
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        models_dir = get_models_dir(create=True)

        if job.model_type == "lstm":
            model_dir = models_dir / f"lstm_{ts}"
            model_dir.mkdir(parents=True, exist_ok=True)
            env_updates["MODEL_PATH"] = str(model_dir / "model.pt")
            job.model_path = str(model_dir)
        elif job.model_type == "bert":
            model_dir = models_dir / f"bert_{ts}"
            model_dir.mkdir(parents=True, exist_ok=True)
            env_updates["BERT_CHECKPOINT_PATH"] = str(model_dir)
            job.model_path = str(model_dir)
        job.logs.append(f"[INFO] Model output path: {job.model_path}")

        # 重定向 stdout 以捕获训练日志
        capture = _StreamCapture(job)

        try:
            with _temporary_env(env_updates), capture:
                if job.model_type == "lstm":
                    self._train_lstm(job)
                elif job.model_type == "bert":
                    self._train_bert(job)
                else:
                    raise ValueError(f"Unknown model_type: {job.model_type}")

            if job._cancel_flag.is_set():
                job.status = "cancelled"
                job.progress.stage = "cancelled"
                job.progress.stage_detail = "训练已取消"
            else:
                job.status = "completed"
                job.progress.stage = "completed"
                job.progress.stage_detail = "训练完成"
                if job.model_path:
                    set_active_model(job.model_type, job.model_path)
        except Exception as exc:
            job.status = "failed"
            job.progress.stage = "failed"
            job.progress.stage_detail = "训练失败"
            job.error = f"{type(exc).__name__}: {exc}"
            job.logs.append(f"[ERROR] {job.error}")
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()

    def _train_lstm(self, job: TrainingJob) -> None:
        from training.trainer import train_model

        job.progress.total_epochs = int(os.getenv("EPOCHS", "25"))
        job.progress.stage = "initializing"
        job.progress.stage_detail = "正在加载 LSTM 训练数据和运行配置"
        train_model(cancel_event=job._cancel_flag)

    def _train_bert(self, job: TrainingJob) -> None:
        from BERT.trainer import train_model

        job.progress.total_epochs = int(os.getenv("BERT_EPOCHS", "5"))
        job.progress.stage = "initializing"
        job.progress.stage_detail = "正在加载 BERT 训练数据和 tokenizer"
        train_model(cancel_event=job._cancel_flag)


class _StreamCapture:
    """上下文管理器：捕获 stdout 并写入 job.logs，同时解析训练指标。"""

    def __init__(self, job: TrainingJob):
        self.job = job
        self._buffer = io.StringIO()
        self._original_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout
        self._buffer.close()

    def write(self, s: str) -> None:
        self._original_stdout.write(s)
        self._buffer.write(s)

        if "\n" in s:
            self._buffer.seek(0)
            full_line = self._buffer.read().strip()
            self._buffer = io.StringIO()

            if full_line:
                self.job.logs.append(full_line)
                self._update_progress(full_line)

    def _update_progress(self, line: str) -> None:
        if line.startswith("Datasets:"):
            self.job.progress.stage = "data_ready"
            self.job.progress.stage_detail = line
        elif line.startswith("DATA SUMMARY"):
            self.job.progress.stage = "data_ready"
            self.job.progress.stage_detail = "训练数据已加载，正在统计样本分布"
        elif line.startswith("Training config:"):
            self.job.progress.stage = "training"
            self.job.progress.stage_detail = line
        elif line.startswith("Training cancellation requested"):
            self.job.progress.stage = "cancelling"
            self.job.progress.stage_detail = "正在停止训练循环"

        step_match = _STEP_PATTERN.search(line)
        if step_match:
            self.job.progress.stage = "training"
            self.job.progress.current_epoch = int(step_match.group(1))
            self.job.progress.total_epochs = int(step_match.group(2))
            self.job.progress.current_step = int(step_match.group(3))
            self.job.progress.total_steps = int(step_match.group(4))
            self.job.progress.loss = float(step_match.group(5))

        m = _EPOCH_PATTERN.search(line)
        if m:
            self.job.progress.stage = "evaluating"
            self.job.progress.stage_detail = f"Epoch {m.group(1)} 验证完成"
            self.job.progress.current_epoch = int(m.group(1))
            self.job.progress.current_step = self.job.progress.total_steps
            self.job.progress.loss = float(m.group(2))
            self.job.progress.val_acc = float(m.group(3))
            self.job.progress.val_f1 = float(m.group(4))

        m2 = _BEST_MODEL_PATTERN.search(line)
        if m2:
            self.job.progress.best_f1 = float(m2.group(1))

    def flush(self) -> None:
        self._original_stdout.flush()


@contextmanager
def _temporary_env(updates: dict[str, str]):
    """临时写入训练环境变量，训练结束后恢复非活跃配置。"""
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
