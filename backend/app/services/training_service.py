"""训练任务管理器。

在后台线程中运行 LSTM/BERT 训练，捕获进度并提供 SSE 兼容的状态查询。
"""

from __future__ import annotations

import io
import os
import re
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import ensure_backend_env_loaded

# 确保 backend 进程可以导入项目根目录下的 training 和 BERT 包
_project_root = Path(__file__).resolve().parents[3]  # SentimentFlow/
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@dataclass
class TrainingProgress:
    current_epoch: int = 0
    total_epochs: int = 0
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

    _cancel_flag: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "model_type": self.model_type,
            "status": self.status,
            "progress": {
                "current_epoch": self.progress.current_epoch,
                "total_epochs": self.progress.total_epochs,
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
        }


_EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+)[,.]?\s*Loss:\s*([\d.]+)[,.]?\s*ValAcc:\s*([\d.]+)[,.]?\s*ValMacroF1:\s*([\d.]+)"
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
        return cls._instance

    def start_training(self, model_type: str, config: dict | None = None) -> TrainingJob:
        """启动训练并返回 job 对象。"""
        ensure_backend_env_loaded()

        if config is None:
            config = {}

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
        job.started_at = datetime.now(timezone.utc).isoformat()

        # 将用户配置写入环境变量（仅影响当前线程的子进程）
        for key, value in job.config.items():
            os.environ[key] = str(value)

        # 重定向 stdout 以捕获训练日志
        capture = _StreamCapture(job)

        try:
            with capture:
                if job.model_type == "lstm":
                    self._train_lstm(job)
                elif job.model_type == "bert":
                    self._train_bert(job)
                else:
                    raise ValueError(f"Unknown model_type: {job.model_type}")

            job.status = "completed"
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.logs.append(f"[ERROR] {job.error}")
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()

    def _train_lstm(self, job: TrainingJob) -> None:
        from training.trainer import train_model

        # 写入 epochs 到环境变量
        epochs = job.config.get("EPOCHS")
        if epochs is not None:
            import training.config as cfg
            # 通过 monkey-patch 修改 epochs
            cfg.EPOCHS = int(epochs)

        job.progress.total_epochs = int(os.getenv("EPOCHS", "25"))
        train_model()

    def _train_bert(self, job: TrainingJob) -> None:
        from BERT.trainer import train_model

        epochs = job.config.get("BERT_EPOCHS")
        if epochs is not None:
            import BERT.config as cfg
            cfg.EPOCHS = int(epochs)

        job.progress.total_epochs = int(os.getenv("BERT_EPOCHS", "5"))
        train_model()


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

                # 解析 epoch 指标行
                m = _EPOCH_PATTERN.search(full_line)
                if m:
                    self.job.progress.current_epoch = int(m.group(1))
                    self.job.progress.loss = float(m.group(2))
                    self.job.progress.val_acc = float(m.group(3))
                    self.job.progress.val_f1 = float(m.group(4))

                # 解析最佳模型更新行
                m2 = _BEST_MODEL_PATTERN.search(full_line)
                if m2:
                    self.job.progress.best_f1 = float(m2.group(1))

    def flush(self) -> None:
        self._original_stdout.flush()
