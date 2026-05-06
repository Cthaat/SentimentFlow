"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";

interface JobStatus {
  job_id: string;
  model_type: string;
  status: string;
  progress: {
    stage: string;
    stage_detail: string | null;
    current_epoch: number;
    total_epochs: number;
    current_step: number;
    total_steps: number;
    loss: number | null;
    val_acc: number | null;
    val_f1: number | null;
    best_f1: number | null;
  };
  logs: string[];
  started_at: string | null;
  finished_at: string | null;
  config: Record<string, unknown>;
  error: string | null;
  model_path?: string | null;
}

const DATASET_OPTIONS = [
  { value: "lansinuote/ChnSentiCorp", label: "ChnSentiCorp (酒店)" },
  { value: "XiangPan/waimai_10k", label: "Waimai 10K (外卖)" },
  { value: "dirtycomputer/weibo_senti_100k", label: "Weibo Senti 100K" },
  { value: "dirtycomputer/JD_review", label: "JD Review" },
  { value: "ndiy/NLPCC14-SC", label: "NLPCC14-SC" },
  { value: "dirtycomputer/ChnSentiCorp_htl_all", label: "ChnSentiCorp Hotel" },
  { value: "BerlinWang/DMSC", label: "DMSC" },
];

const LSTM_DEFAULTS: Record<string, string> = {
  EPOCHS: "25",
  TRAIN_BATCH_SIZE: "512",
  TRAIN_LR: "0.0005",
  EARLY_STOP_PATIENCE: "3",
  EARLY_STOP_MIN_DELTA: "0.0005",
  TRAIN_WEIGHTED_LOSS: "1",
};

const BERT_DEFAULTS: Record<string, string> = {
  BERT_EPOCHS: "5",
  BERT_TRAIN_BATCH_SIZE: "32",
  BERT_TRAIN_LR: "0.00002",
  BERT_EARLY_STOP_PATIENCE: "2",
  BERT_EARLY_STOP_MIN_DELTA: "0.0005",
  BERT_TRAIN_WEIGHTED_LOSS: "1",
};

export function TrainingPanel() {
  const [modelType, setModelType] = useState<"lstm" | "bert">("lstm");
  const [params, setParams] = useState<Record<string, string>>({ ...LSTM_DEFAULTS });
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>(
    DATASET_OPTIONS.map((d) => d.value)
  );
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [connectionState, setConnectionState] = useState<"idle" | "connecting" | "live" | "polling" | "closed">("idle");
  const [now, setNow] = useState(() => Date.now());
  const eventSourceRef = useRef<EventSource | null>(null);
  const pollTimerRef = useRef<number | null>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // 切换模型类型时重置参数
  const handleModelTypeChange = useCallback((value: string) => {
    const mt = value as "lstm" | "bert";
    setModelType(mt);
    setParams(mt === "lstm" ? { ...LSTM_DEFAULTS } : { ...BERT_DEFAULTS });
  }, []);

  // 自动滚动日志
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logLines]);

  // 清理 SSE 连接
  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
      if (pollTimerRef.current != null) {
        window.clearInterval(pollTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!loading && !jobStatus) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [loading, jobStatus]);

  const appendLog = useCallback((line: string) => {
    setLogLines((prev) => {
      if (prev[prev.length - 1] === line) return prev;
      return [...prev.slice(-300), line];
    });
  }, []);

  const updateJobStatus = useCallback((next: JobStatus) => {
    setJobStatus(next);
    if (next.logs?.length) {
      setLogLines((prev) => {
        const seen = new Set(prev);
        const merged = [...prev];
        for (const line of next.logs) {
          if (!seen.has(line)) {
            merged.push(line);
            seen.add(line);
          }
        }
        return merged.slice(-300);
      });
    }
    if (next.error) {
      appendLog(`[ERROR] ${next.error}`);
    }
    if (isTerminalStatus(next.status)) {
      eventSourceRef.current?.close();
      if (pollTimerRef.current != null) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      setConnectionState("closed");
      setLoading(false);
    }
  }, [appendLog]);

  const startTraining = async () => {
    setLoading(true);
    setJobId(null);
    setJobStatus(null);
    setConnectionState("connecting");
    setLogLines([
      "正在提交训练任务...",
      "如果后端正在启动模型或加载训练模块，启动请求最多会等待 120 秒。",
    ]);

    const datasetKey = modelType === "lstm" ? "TRAIN_DATASETS" : "BERT_TRAIN_DATASETS";
    const config: Record<string, string> = {
      ...params,
      [datasetKey]: selectedDatasets.join(","),
    };

    try {
      const res = await fetch("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_type: modelType, config }),
      });
      const data = await res.json();
      if (data.ok && data.payload.job_id) {
        setJobId(data.payload.job_id);
        appendLog(`训练任务已创建: ${data.payload.job_id}`);
        connectSSE(data.payload.job_id);
      } else {
        appendLog("启动训练失败: " + getErrorMessage(data));
        setConnectionState("closed");
        setLoading(false);
      }
    } catch (err) {
      appendLog("请求失败: " + (err instanceof Error ? err.message : String(err)));
      setConnectionState("closed");
      setLoading(false);
    }
  };

  const connectSSE = (id: string) => {
    eventSourceRef.current?.close();
    if (pollTimerRef.current != null) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    setConnectionState("connecting");
    const es = new EventSource(`/api/training/stream/${id}`);
    eventSourceRef.current = es;

    es.onopen = () => {
      setConnectionState("live");
      appendLog("实时日志通道已连接");
    };

    es.onmessage = (event) => {
      if (event.data === "[DONE]") {
        es.close();
        setConnectionState("closed");
        setLoading(false);
        return;
      }
      try {
        const data = JSON.parse(event.data);
        if (data.type === "log") {
          appendLog(data.message as string);
        } else if (data.job_id) {
          updateJobStatus(data as JobStatus);
        }
      } catch {
        // 忽略非 JSON 消息
      }
    };

    es.onerror = () => {
      es.close();
      appendLog("实时日志通道断开，切换为状态轮询...");
      startPolling(id);
    };
  };

  const startPolling = (id: string) => {
    if (pollTimerRef.current != null) {
      window.clearInterval(pollTimerRef.current);
    }
    setConnectionState("polling");

    const fetchStatus = async () => {
      try {
        const res = await fetch(`/api/training/status/${id}`, { cache: "no-store" });
        const data = await res.json();
        if (data.ok && data.payload) {
          updateJobStatus(data.payload as JobStatus);
        } else {
          appendLog("状态轮询失败: " + getErrorMessage(data));
        }
      } catch (err) {
        appendLog("状态轮询请求失败: " + (err instanceof Error ? err.message : String(err)));
      }
    };

    void fetchStatus();
    pollTimerRef.current = window.setInterval(fetchStatus, 2000);
  };

  const cancelTraining = async () => {
    if (!jobId) return;
    try {
      await fetch("/api/training/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: jobId }),
      });
    } catch {
      // 忽略
    }
    eventSourceRef.current?.close();
    appendLog("已发送取消请求，等待训练循环停止...");
    startPolling(jobId);
  };

  const toggleDataset = (value: string) => {
    setSelectedDatasets((prev) =>
      prev.includes(value) ? prev.filter((d) => d !== value) : [...prev, value]
    );
  };

  const statusVariant = jobStatus?.status === "completed" ? "default" : jobStatus?.status === "failed" ? "destructive" : jobStatus?.status === "running" ? "default" : "outline";
  const statusText =
    jobStatus?.status === "running"
      ? "训练中"
      : jobStatus?.status === "completed"
        ? "已完成"
        : jobStatus?.status === "failed"
          ? "失败"
          : jobStatus?.status === "cancelled"
            ? "已取消"
            : "待启动";
  const stageText = getStageText(jobStatus?.progress.stage);
  const epochPercent =
    jobStatus && jobStatus.progress.total_epochs > 0
      ? (jobStatus.progress.current_epoch / jobStatus.progress.total_epochs) * 100
      : 0;
  const stepPercent =
    jobStatus && jobStatus.progress.total_steps > 0
      ? (jobStatus.progress.current_step / jobStatus.progress.total_steps) * 100
      : 0;
  const elapsedText = getElapsedText(jobStatus, loading, now);
  const latestLog = logLines.length > 0 ? logLines[logLines.length - 1] : "尚未收到训练日志";

  return (
    <Card className="w-full max-w-3xl">
      <CardHeader>
        <CardTitle>模型训练</CardTitle>
        <CardDescription>配置参数并启动 LSTM 或 BERT 情感分析模型训练</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* 模型类型选择 */}
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <label className="text-sm font-medium">模型类型</label>
            <Select
              options={[
                { value: "lstm", label: "LSTM" },
                { value: "bert", label: "BERT (RoBERTa)" },
              ]}
              value={modelType}
              onChange={(e) => handleModelTypeChange(e.target.value)}
              disabled={loading}
            />
          </div>
          {jobStatus && (
            <div className="flex items-end gap-2 pb-1">
              <span className="text-sm text-muted-foreground">状态</span>
              <Badge variant={statusVariant === "destructive" ? "destructive" : statusVariant === "outline" ? "outline" : "default"}>
                {statusText}
              </Badge>
              {jobStatus.progress.current_epoch > 0 && (
                <span className="text-sm text-muted-foreground">
                  Epoch {jobStatus.progress.current_epoch}/{jobStatus.progress.total_epochs}
                </span>
              )}
            </div>
          )}
        </div>

        {/* 超参数 */}
        <div className="space-y-2">
          <label className="text-sm font-medium">超参数</label>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            {Object.entries(params).map(([key, value]) => (
              <div key={key} className="space-y-1">
                <label className="text-xs text-muted-foreground">{key}</label>
                <Input
                  value={value}
                  onChange={(e) => setParams((p) => ({ ...p, [key]: e.target.value }))}
                  disabled={loading}
                  className="h-8 text-sm"
                />
              </div>
            ))}
          </div>
        </div>

        {/* 数据集选择 */}
        <div className="space-y-2">
          <label className="text-sm font-medium">
            训练数据集
            <span className="ml-2 text-xs text-muted-foreground">
              (已选 {selectedDatasets.length}/{DATASET_OPTIONS.length})
            </span>
          </label>
          <div className="flex flex-wrap gap-2">
            {DATASET_OPTIONS.map((ds) => (
              <button
                key={ds.value}
                disabled={loading}
                onClick={() => toggleDataset(ds.value)}
                className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors ${
                  selectedDatasets.includes(ds.value)
                    ? "border-transparent bg-primary text-primary-foreground hover:bg-primary/80"
                    : "border-muted-foreground/30 text-muted-foreground hover:border-primary/50"
                } disabled:cursor-not-allowed disabled:opacity-50`}
              >
                {ds.label}
              </button>
            ))}
          </div>
        </div>

        {/* 操作按钮 */}
        <div className="flex gap-2">
          <Button onClick={startTraining} disabled={loading || selectedDatasets.length === 0}>
            {loading ? (jobStatus ? "训练中..." : "启动中...") : "开始训练"}
          </Button>
          {loading && (
            <Button variant="outline" onClick={cancelTraining}>
              取消训练
            </Button>
          )}
          {connectionState !== "idle" && (
            <Badge variant={connectionState === "live" ? "default" : "outline"} className="self-center">
              {getConnectionText(connectionState)}
            </Badge>
          )}
        </div>

        {loading && !jobStatus && (
          <div className="space-y-3 rounded-lg border border-dashed p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-sm font-medium">正在启动训练任务</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  后端会先创建 job_id，再进入数据加载和训练循环。启动阶段最长等待 120 秒。
                </p>
              </div>
              <Badge variant="outline">等待响应</Badge>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div className="h-full w-1/3 animate-pulse rounded-full bg-primary" />
            </div>
          </div>
        )}

        {/* 训练进度 */}
        {jobStatus && (
          <div className="space-y-4 rounded-lg border p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h4 className="text-sm font-medium">训练进度</h4>
                <p className="mt-1 text-xs text-muted-foreground">
                  Job {jobStatus.job_id} · {jobStatus.model_type.toUpperCase()} · 已运行 {elapsedText}
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                <Badge variant={statusVariant === "destructive" ? "destructive" : statusVariant === "outline" ? "outline" : "default"}>
                  {statusText}
                </Badge>
                <Badge variant="outline">{stageText}</Badge>
              </div>
            </div>
            {jobStatus.model_path && (
              <p className="break-all text-xs text-muted-foreground">
                模型路径: {jobStatus.model_path}
              </p>
            )}
            {jobStatus.progress.stage_detail && (
              <p className="break-all rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">
                {jobStatus.progress.stage_detail}
              </p>
            )}
            {jobStatus.error && (
              <p className="break-all text-sm text-destructive">
                {jobStatus.error}
              </p>
            )}

            {/* 进度条 */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Epoch 进度</span>
                <span>{jobStatus.progress.current_epoch} / {jobStatus.progress.total_epochs}</span>
              </div>
              <div className="h-2 w-full rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary transition-all duration-500"
                  style={{ width: `${epochPercent}%` }}
                />
              </div>
            </div>

            {jobStatus.progress.total_steps > 0 && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>当前 Epoch Step</span>
                  <span>{jobStatus.progress.current_step} / {jobStatus.progress.total_steps}</span>
                </div>
                <div className="h-2 w-full rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-primary/70 transition-all duration-500"
                    style={{ width: `${stepPercent}%` }}
                  />
                </div>
              </div>
            )}

            {/* 指标 */}
            <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
              <MetricBox label="Elapsed" value={elapsedText} />
              <MetricBox label="Loss" value={jobStatus.progress.loss} suffix="" />
              <MetricBox label="Val Accuracy" value={jobStatus.progress.val_acc} suffix="%" isPercent />
              <MetricBox label="Val Macro F1" value={jobStatus.progress.val_f1} suffix="" />
              <MetricBox label="Best F1" value={jobStatus.progress.best_f1} suffix="" />
            </div>
          </div>
        )}

        {/* 训练日志 */}
        {(logLines.length > 0 || loading) && (
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-3">
              <label className="text-sm font-medium">训练日志</label>
              <span className="text-xs text-muted-foreground">
                {logLines.length} 行 · 最新: {latestLog.slice(0, 48)}
              </span>
            </div>
            <div
              ref={logContainerRef}
              className="h-64 overflow-auto rounded-lg border bg-zinc-950 p-3 font-mono text-xs text-zinc-300 dark:bg-zinc-950"
            >
              {logLines.map((line, i) => (
                <div key={i} className="leading-relaxed">
                  {line}
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function getErrorMessage(data: unknown): string {
  if (!data || typeof data !== "object") return String(data);
  const record = data as {
    error?: string;
    payload?: { detail?: string };
    detail?: string;
  };
  return record.payload?.detail || record.detail || record.error || JSON.stringify(data);
}

function isTerminalStatus(status: string): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

function getConnectionText(state: "idle" | "connecting" | "live" | "polling" | "closed"): string {
  switch (state) {
    case "connecting":
      return "连接中";
    case "live":
      return "实时日志";
    case "polling":
      return "轮询更新";
    case "closed":
      return "连接关闭";
    default:
      return "未连接";
  }
}

function getStageText(stage?: string): string {
  switch (stage) {
    case "starting":
      return "启动中";
    case "initializing":
      return "初始化";
    case "data_ready":
      return "数据就绪";
    case "training":
      return "训练中";
    case "evaluating":
      return "验证中";
    case "cancelling":
      return "取消中";
    case "completed":
      return "已完成";
    case "failed":
      return "失败";
    case "cancelled":
      return "已取消";
    default:
      return "排队中";
  }
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function getElapsedText(jobStatus: JobStatus | null, loading: boolean, now: number): string {
  if (!jobStatus?.started_at) {
    return loading ? "启动中" : "-";
  }

  const start = new Date(jobStatus.started_at).getTime();
  const end = jobStatus.finished_at ? new Date(jobStatus.finished_at).getTime() : now;
  return formatDuration(Math.max(0, end - start));
}

function MetricBox({
  label,
  value,
  suffix,
  isPercent,
}: {
  label: string;
  value: number | string | null;
  suffix?: string;
  isPercent?: boolean;
}) {
  const display =
    typeof value === "string"
      ? value
      : value != null
      ? isPercent
        ? `${(value * 100).toFixed(2)}${suffix ?? ""}`
        : `${value.toFixed(4)}${suffix ?? ""}`
      : "-";
  return (
    <div className="rounded-lg border p-2 text-center">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-lg font-semibold tabular-nums">{display}</div>
    </div>
  );
}
