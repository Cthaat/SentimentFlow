"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";

type ApiResult = {
	ok: boolean;
	upstreamStatus?: number;
	baseUrl?: string;
	payload?: unknown;
	error?: string;
};

type PredictPayload = {
	text: string;
	score: number;
	label: string;
	label_zh: string;
	confidence: number;
	probabilities: number[];
	reasoning: string;
	source: string;
	model_name?: string;
};

const defaultText = "这个产品非常好用，体验超出预期。";

const scoreLabels = ["极端负面", "明显负面", "略微负面", "中性", "略微正面", "极端正面"];
const scoreStyles = [
	"bg-red-600 text-white",
	"bg-orange-500 text-white",
	"bg-amber-400 text-zinc-950",
	"bg-zinc-500 text-white",
	"bg-emerald-500 text-white",
	"bg-green-700 text-white",
];

export function IntegrationTestPanel() {
	const [text, setText] = useState(defaultText);
	const [healthLoading, setHealthLoading] = useState(false);
	const [predictLoading, setPredictLoading] = useState(false);
	const [healthResult, setHealthResult] = useState<ApiResult | null>(null);
	const [predictResult, setPredictResult] = useState<ApiResult | null>(null);
	const [activeModelName, setActiveModelName] = useState<string | null>(null);

	const fetchActiveModel = useCallback(async () => {
		try {
			const res = await fetch("/api/models/active");
			const data = await res.json();
			if (data.ok && data.payload) {
				const p = data.payload as {
					lstm_path?: string;
					bert_path?: string;
					predict_model_type?: string;
				};
				const usedType = p.predict_model_type || "lstm";
				const path = usedType === "lstm" ? p.lstm_path : p.bert_path;
				if (path) {
					const parts = path.replace(/\\/g, "/").split("/");
					const dir = parts[parts.length - 2];
					setActiveModelName(dir?.startsWith(usedType) ? dir : path);
				}
			}
		} catch {
			// ignore
		}
	}, []);

	useEffect(() => {
		fetchActiveModel();
	}, [fetchActiveModel]);

	const backendStatus = useMemo(() => {
		if (!healthResult) return "未检测";
		return healthResult.ok ? "在线" : "异常";
	}, [healthResult]);

	const runHealthCheck = async () => {
		setHealthLoading(true);
		try {
			const response = await fetch("/api/integration/health", { method: "GET" });
			const data = (await response.json()) as ApiResult;
			setHealthResult(data);
		} catch (error) {
			setHealthResult({
				ok: false,
				error: error instanceof Error ? error.message : "健康检查请求失败",
			});
		} finally {
			setHealthLoading(false);
		}
	};

	const runPredict = async () => {
		setPredictLoading(true);
		try {
			const body: Record<string, string> = { text };
			const response = await fetch("/api/integration/predict", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify(body),
			});
			const data = (await response.json()) as ApiResult;
			setPredictResult({ ...data, upstreamStatus: response.status });
		} catch (error) {
			setPredictResult({
				ok: false,
				error: error instanceof Error ? error.message : "预测请求失败",
			});
		} finally {
			setPredictLoading(false);
		}
	};

	const noModelError =
		predictResult && !predictResult.ok && predictResult.upstreamStatus === 404
			? (predictResult.payload as { detail?: string })?.detail ||
			  predictResult.error ||
			  "模型文件不存在，请先在「模型训练」页面训练模型"
			: null;

	const usedModelName =
		predictResult?.ok && predictResult.payload
			? (predictResult.payload as { model_name?: string })?.model_name || null
			: null;

	const prediction =
		predictResult?.ok && predictResult.payload
			? (predictResult.payload as PredictPayload)
			: null;

	return (
		<Card className="w-full max-w-3xl">
			<CardHeader>
				<CardTitle>情感预测</CardTitle>
				<CardDescription>输入文本，使用当前活跃模型获取情感分析结果</CardDescription>
			</CardHeader>
			<CardContent className="space-y-4">
				<div className="flex items-center gap-2">
					<span className="text-sm text-muted-foreground">后端状态</span>
					<Badge variant={backendStatus === "在线" ? "default" : "outline"}>{backendStatus}</Badge>
					<Button
						size="sm"
						variant="outline"
						onClick={runHealthCheck}
						disabled={healthLoading}
					>
						{healthLoading ? "检测中..." : "健康检查"}
					</Button>
				</div>

				<div className="space-y-2">
					<p className="text-sm font-medium">输入文本</p>
					<Textarea
						value={text}
						onChange={(event) => setText(event.target.value)}
						placeholder="输入要分析情感的文本"
					/>
					<Button
						onClick={runPredict}
						disabled={predictLoading || !text.trim()}
					>
						{predictLoading ? "请求中..." : "发送预测请求"}
					</Button>
				</div>

				{noModelError && (
					<div className="rounded-lg border border-amber-300 bg-amber-50 p-4 dark:border-amber-700 dark:bg-amber-900/30">
						<p className="text-sm font-medium text-amber-800 dark:text-amber-200">
							未检测到可用模型
						</p>
						<p className="mt-1 text-sm text-amber-700 dark:text-amber-300">
							{noModelError}
						</p>
					</div>
				)}

				{(usedModelName || activeModelName) && (
					<div className="flex items-center gap-2 rounded-lg border border-green-300 bg-green-50 px-3 py-2 dark:border-green-700 dark:bg-green-900/30">
						<span className="text-xs text-muted-foreground">当前使用模型</span>
						<Badge variant="default" className="text-xs">{usedModelName || activeModelName}</Badge>
					</div>
				)}

				{prediction && (
					<div className="space-y-4 rounded-lg border p-4">
						<div className="flex flex-wrap items-center justify-between gap-3">
							<div className="flex items-center gap-3">
								<span className={`inline-flex h-12 w-12 items-center justify-center rounded-md text-xl font-semibold ${scoreStyles[prediction.score] ?? "bg-zinc-500 text-white"}`}>
									{prediction.score}
								</span>
								<div>
									<p className="text-sm font-medium">{prediction.label_zh || scoreLabels[prediction.score]}</p>
									<p className="text-xs text-muted-foreground">{prediction.label}</p>
								</div>
							</div>
							<div className="text-right">
								<p className="text-xs text-muted-foreground">置信度</p>
								<p className="text-lg font-semibold tabular-nums">{(prediction.confidence * 100).toFixed(2)}%</p>
							</div>
						</div>
						<div className="space-y-2">
							{scoreLabels.map((label, score) => {
								const probability = prediction.probabilities?.[score] ?? 0;
								return (
									<div key={score} className="grid grid-cols-[4.5rem_1fr_3.5rem] items-center gap-2 text-xs">
										<span className="text-muted-foreground">{score} {label}</span>
										<div className="h-2 overflow-hidden rounded-full bg-muted">
											<div
												className={`h-full ${scoreStyles[score].split(" ")[0]}`}
												style={{ width: `${Math.max(0, Math.min(1, probability)) * 100}%` }}
											/>
										</div>
										<span className="text-right tabular-nums">{(probability * 100).toFixed(1)}%</span>
									</div>
								);
							})}
						</div>
						<p className="rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">
							{prediction.reasoning}
						</p>
					</div>
				)}

				<div className="grid gap-4 md:grid-cols-2">
					<div className="rounded-lg border p-3">
						<p className="mb-2 text-sm font-medium">健康检查响应</p>
						<pre className="max-h-56 overflow-auto text-xs">{JSON.stringify(healthResult, null, 2) || "尚未请求"}</pre>
					</div>
					<div className="rounded-lg border p-3">
						<p className="mb-2 text-sm font-medium">预测响应</p>
						<pre className="max-h-56 overflow-auto text-xs">{JSON.stringify(predictResult, null, 2) || "尚未请求"}</pre>
					</div>
				</div>
			</CardContent>
		</Card>
	);
}
