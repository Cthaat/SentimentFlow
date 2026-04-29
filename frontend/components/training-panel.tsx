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
    current_epoch: number;
    total_epochs: number;
    loss: number | null;
    val_acc: number | null;
    val_f1: number | null;
    best_f1: number | null;
  };
  logs: string[];
  config: Record<string, unknown>;
  error: string | null;
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
  const eventSourceRef = useRef<EventSource | null>(null);
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
    };
  }, []);

  const startTraining = async () => {
    setLoading(true);
    setJobStatus(null);
    setLogLines([]);

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
        connectSSE(data.payload.job_id);
      } else {
        setLogLines(["启动训练失败: " + JSON.stringify(data)]);
        setLoading(false);
      }
    } catch (err) {
      setLogLines(["请求失败: " + (err instanceof Error ? err.message : String(err))]);
      setLoading(false);
    }
  };

  const connectSSE = (id: string) => {
    eventSourceRef.current?.close();
    const es = new EventSource(`/api/training/stream/${id}`);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      if (event.data === "[DONE]") {
        es.close();
        setLoading(false);
        return;
      }
      try {
        const data = JSON.parse(event.data);
        if (data.type === "log") {
          setLogLines((prev) => [...prev.slice(-200), data.message as string]);
        } else if (data.job_id) {
          setJobStatus(data as JobStatus);
          if (data.status === "completed" || data.status === "failed" || data.status === "cancelled") {
            es.close();
            setLoading(false);
          }
        }
      } catch {
        // 忽略非 JSON 消息
      }
    };

    es.onerror = () => {
      // SSE 连接断开时重试
    };
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
    setLoading(false);
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
            {loading ? "训练中..." : "开始训练"}
          </Button>
          {loading && (
            <Button variant="outline" onClick={cancelTraining}>
              取消训练
            </Button>
          )}
        </div>

        {/* 训练进度 */}
        {jobStatus && (
          <div className="space-y-4 rounded-lg border p-4">
            <h4 className="text-sm font-medium">训练进度</h4>

            {/* 进度条 */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Epoch 进度</span>
                <span>{jobStatus.progress.current_epoch} / {jobStatus.progress.total_epochs}</span>
              </div>
              <div className="h-2 w-full rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary transition-all duration-500"
                  style={{
                    width: `${jobStatus.progress.total_epochs > 0 ? (jobStatus.progress.current_epoch / jobStatus.progress.total_epochs) * 100 : 0}%`,
                  }}
                />
              </div>
            </div>

            {/* 指标 */}
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <MetricBox label="Loss" value={jobStatus.progress.loss} suffix="" />
              <MetricBox label="Val Accuracy" value={jobStatus.progress.val_acc} suffix="%" isPercent />
              <MetricBox label="Val Macro F1" value={jobStatus.progress.val_f1} suffix="" />
              <MetricBox label="Best F1" value={jobStatus.progress.best_f1} suffix="" />
            </div>
          </div>
        )}

        {/* 训练日志 */}
        {logLines.length > 0 && (
          <div className="space-y-2">
            <label className="text-sm font-medium">训练日志</label>
            <div
              ref={logContainerRef}
              className="h-48 overflow-auto rounded-lg border bg-zinc-950 p-3 font-mono text-xs text-zinc-300 dark:bg-zinc-950"
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

function MetricBox({
  label,
  value,
  suffix,
  isPercent,
}: {
  label: string;
  value: number | null;
  suffix: string;
  isPercent?: boolean;
}) {
  const display =
    value != null
      ? isPercent
        ? `${(value * 100).toFixed(2)}${suffix}`
        : `${value.toFixed(4)}${suffix}`
      : "-";
  return (
    <div className="rounded-lg border p-2 text-center">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-lg font-semibold tabular-nums">{display}</div>
    </div>
  );
}
