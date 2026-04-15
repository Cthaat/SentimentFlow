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
			const response = await fetch("/api/integration/predict", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({ text }),
			});
			const data = (await response.json()) as ApiResult;
			setPredictResult(data);
		} catch (error) {
			setPredictResult({
				ok: false,
				error: error instanceof Error ? error.message : "预测请求失败",
			});
		} finally {
			setPredictLoading(false);
		}
	};

	return (
		<Card className="w-full max-w-3xl">
			<CardHeader>
				<CardTitle>前后端联调测试面板</CardTitle>
				<CardDescription>使用 shadcn 组件构建。先做健康检查，再发起情感预测请求。</CardDescription>
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
