"use client";

import { useCallback, useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface ModelInfo {
	model_id: string;
	model_type: "lstm" | "bert";
	path: string;
	best_f1: number | null;
	best_epoch: number | null;
}

interface ModelsResponse {
	models: ModelInfo[];
	active_lstm_path: string | null;
	active_bert_path: string | null;
}

export function ModelManagementPanel() {
	const [models, setModels] = useState<ModelInfo[]>([]);
	const [activeLstmPath, setActiveLstmPath] = useState<string | null>(null);
	const [activeBertPath, setActiveBertPath] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);
	const [deletingId, setDeletingId] = useState<string | null>(null);
	const [activatingId, setActivatingId] = useState<string | null>(null);

	const fetchModels = useCallback(async () => {
		setLoading(true);
		try {
			const res = await fetch("/api/models");
			const data = await res.json();
			if (data.ok && data.payload) {
				const p = data.payload as ModelsResponse;
				setModels(p.models || []);
				setActiveLstmPath(p.active_lstm_path || null);
				setActiveBertPath(p.active_bert_path || null);
			}
		} catch {
			// ignore
		} finally {
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		fetchModels();
	}, [fetchModels]);

	const handleActivate = async (model: ModelInfo) => {
		setActivatingId(model.model_id);
		try {
			const res = await fetch("/api/models/active", {
				method: "PUT",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model_type: model.model_type,
					model_path: model.path,
				}),
			});
			const data = await res.json();
			if (data.ok) {
				if (model.model_type === "lstm") {
					setActiveLstmPath(model.path);
				} else {
					setActiveBertPath(model.path);
				}
			}
		} catch {
			// ignore
		} finally {
			setActivatingId(null);
		}
	};

	const handleDelete = async (model: ModelInfo) => {
		if (!window.confirm(`确认删除模型「${model.model_id}」吗？此操作不可撤销。`)) return;
		setDeletingId(model.model_id);
		try {
			const res = await fetch("/api/models", {
				method: "DELETE",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ model_id: model.model_id }),
			});
			const data = await res.json();
			if (data.ok || data.payload?.ok) {
				setModels((prev) => prev.filter((m) => m.model_id !== model.model_id));
				// 如果删除的是活跃模型，清除活跃状态
				if (model.model_type === "lstm" && activeLstmPath === model.path) {
					setActiveLstmPath(null);
				}
				if (model.model_type === "bert" && activeBertPath === model.path) {
					setActiveBertPath(null);
				}
			}
		} catch {
			// ignore
		} finally {
			setDeletingId(null);
		}
	};

	const isActive = (model: ModelInfo) =>
		(model.model_type === "lstm" && activeLstmPath === model.path) ||
		(model.model_type === "bert" && activeBertPath === model.path);

	const formatTimestamp = (modelId: string): string => {
		// 从目录名提取时间戳，如 lstm_20260429_143025 -> 2026-04-29 14:30:25
		const m = modelId.match(/(\d{8})_(\d{6})$/);
		if (!m) return "";
		const d = m[1];
		const t = m[2];
		return `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)} ${t.slice(0, 2)}:${t.slice(2, 4)}:${t.slice(4, 6)}`;
	};

	return (
		<Card className="w-full max-w-3xl">
			<CardHeader>
				<div className="flex items-center justify-between">
					<div>
						<CardTitle>模型管理</CardTitle>
						<CardDescription>浏览已训练的模型，切换活跃模型或删除不需要的模型</CardDescription>
					</div>
					<Button size="sm" variant="outline" onClick={fetchModels} disabled={loading}>
						{loading ? "加载中..." : "刷新列表"}
					</Button>
				</div>
			</CardHeader>
			<CardContent className="space-y-4">
				{models.length === 0 && !loading && (
					<div className="rounded-lg border border-dashed p-8 text-center text-muted-foreground">
						<p className="text-sm">暂无已训练的模型</p>
						<p className="mt-1 text-xs">请先在「模型训练」页面完成训练后，再来这里管理模型</p>
					</div>
				)}

				{models.length > 0 && (
					<div className="space-y-2">
						{models.map((model) => (
							<div
								key={model.model_id}
								className={`flex items-center gap-3 rounded-lg border p-3 transition-colors ${
									isActive(model)
										? "border-primary/50 bg-primary/5"
										: ""
								}`}
							>
								{/* 类型标签 */}
								<Badge variant={model.model_type === "bert" ? "default" : "outline"}>
									{model.model_type.toUpperCase()}
								</Badge>

								{/* 模型信息 */}
								<div className="min-w-0 flex-1">
									<div className="flex items-center gap-2">
										<span className="truncate text-sm font-medium">
											{model.model_id}
										</span>
										{isActive(model) && (
											<Badge variant="default" className="shrink-0 text-xs">
												使用中
											</Badge>
										)}
									</div>
									<div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
										{formatTimestamp(model.model_id) && (
											<span>{formatTimestamp(model.model_id)}</span>
										)}
										{model.best_f1 != null && (
											<span>
												Best F1: {(model.best_f1 * 100).toFixed(2)}%
											</span>
										)}
										{model.best_epoch != null && (
											<span>Best Epoch: {model.best_epoch}</span>
										)}
									</div>
								</div>

								{/* 操作按钮 */}
								<div className="flex shrink-0 gap-1">
									{!isActive(model) && (
										<Button
											size="sm"
											variant="outline"
											onClick={() => handleActivate(model)}
											disabled={activatingId === model.model_id}
										>
											{activatingId === model.model_id ? "切换中..." : "启用"}
										</Button>
									)}
									<Button
										size="sm"
										variant="outline"
										onClick={() => handleDelete(model)}
										disabled={deletingId === model.model_id}
										className="text-destructive hover:bg-destructive/10 hover:text-destructive"
									>
										{deletingId === model.model_id ? "删除中..." : "删除"}
									</Button>
								</div>
							</div>
						))}
					</div>
				)}
			</CardContent>
		</Card>
	);
}
