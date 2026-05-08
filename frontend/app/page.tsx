"use client";
import { useState } from "react";
import { IntegrationTestPanel } from "@/components/integration-test-panel";
import { ModelManagementPanel } from "@/components/model-management-panel";
import { ThemeToggle } from "@/components/theme-toggle";
import { TrainingPanel } from "@/components/training-panel";

type Tab = "predict" | "train" | "models";

export default function Home() {
	const [activeTab, setActiveTab] = useState<Tab>("predict");

	return (
		<div className="min-h-screen bg-zinc-50 px-4 py-10 dark:bg-zinc-900">
			<main className="mx-auto flex w-full max-w-4xl flex-col items-center gap-6">
				<div className="flex w-full items-center justify-between">
					<h1 className="text-2xl font-semibold tracking-tight">SentimentFlow</h1>
					<ThemeToggle />
				</div>

				{/* 标签页导航 */}
				<div className="flex w-full gap-1 rounded-lg bg-muted p-1">
					<button
						type="button"
						onClick={() => setActiveTab("predict")}
						className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
							activeTab === "predict"
								? "bg-background text-foreground shadow-sm"
								: "text-muted-foreground hover:text-foreground"
						}`}
					>
						情感预测
					</button>
					<button
						type="button"
						onClick={() => setActiveTab("train")}
						className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
							activeTab === "train"
								? "bg-background text-foreground shadow-sm"
								: "text-muted-foreground hover:text-foreground"
						}`}
					>
						模型训练
					</button>
					<button
						type="button"
						onClick={() => setActiveTab("models")}
						className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
							activeTab === "models"
								? "bg-background text-foreground shadow-sm"
								: "text-muted-foreground hover:text-foreground"
						}`}
					>
						模型管理
					</button>
				</div>

				{activeTab === "predict" ? (
					<IntegrationTestPanel />
				) : activeTab === "train" ? (
					<TrainingPanel />
				) : (
					<ModelManagementPanel />
				)}
			</main>
		</div>
	);
}
