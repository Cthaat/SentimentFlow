"use client";

import { useMemo, useState } from "react";

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

const defaultText = "这个产品非常好用，体验超出预期。";

export function IntegrationTestPanel() {
	const [text, setText] = useState(defaultText);
	const [healthLoading, setHealthLoading] = useState(false);
	const [predictLoading, setPredictLoading] = useState(false);
	const [healthResult, setHealthResult] = useState<ApiResult | null>(null);
	const [predictResult, setPredictResult] = useState<ApiResult | null>(null);

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

				{usedModelName && (
					<div className="flex items-center gap-2 rounded-lg border border-green-300 bg-green-50 px-3 py-2 dark:border-green-700 dark:bg-green-900/30">
						<span className="text-xs text-muted-foreground">当前使用模型</span>
						<Badge variant="default" className="text-xs">{usedModelName}</Badge>
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
