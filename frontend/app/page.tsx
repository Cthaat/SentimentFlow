"use client";
import { useCallback, useSyncExternalStore } from "react";
import { IntegrationTestPanel } from "@/components/integration-test-panel";
import { ModelManagementPanel } from "@/components/model-management-panel";
import { ThemeToggle } from "@/components/theme-toggle";
import { TrainingPanel } from "@/components/training-panel";

type Tab = "predict" | "train" | "models";
const ACTIVE_TAB_STORAGE_KEY = "sentimentflow.activeTab";
const TABS: Tab[] = ["predict", "train", "models"];
const activeTabListeners = new Set<() => void>();
let activeTabSnapshot: Tab = "predict";
let activeTabInitialized = false;

function normalizeTab(value: string | null): Tab {
	return TABS.includes(value as Tab) ? (value as Tab) : "predict";
}

function getStoredActiveTab(): Tab {
	if (typeof window === "undefined") return "predict";
	try {
		const storedTab = window.localStorage.getItem(ACTIVE_TAB_STORAGE_KEY);
		return normalizeTab(storedTab);
	} catch {
		return "predict";
	}
}

function getActiveTabSnapshot(): Tab {
	return activeTabSnapshot;
}

function getServerActiveTab(): Tab {
	return "predict";
}

function emitActiveTabChange(): void {
	for (const listener of activeTabListeners) {
		listener();
	}
}

function initializeActiveTabSnapshot(): void {
	if (activeTabInitialized || typeof window === "undefined") return;
	activeTabInitialized = true;
	activeTabSnapshot = getStoredActiveTab();
}

function subscribeActiveTab(listener: () => void): () => void {
	activeTabListeners.add(listener);
	initializeActiveTabSnapshot();
	queueMicrotask(listener);

	const handleStorage = (event: StorageEvent) => {
		if (event.key !== ACTIVE_TAB_STORAGE_KEY) return;
		activeTabSnapshot = normalizeTab(event.newValue);
		emitActiveTabChange();
	};
	window.addEventListener("storage", handleStorage);

	return () => {
		activeTabListeners.delete(listener);
		window.removeEventListener("storage", handleStorage);
	};
}

function writeStoredActiveTab(tab: Tab): void {
	activeTabSnapshot = tab;
	try {
		window.localStorage.setItem(ACTIVE_TAB_STORAGE_KEY, tab);
	} catch {
		// localStorage 不可用时仍保留当前页内状态。
	}
}

export default function Home() {
	const activeTab = useSyncExternalStore(subscribeActiveTab, getActiveTabSnapshot, getServerActiveTab);

	const setActiveTab = useCallback((tab: Tab) => {
		writeStoredActiveTab(tab);
		emitActiveTabChange();
	}, []);

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

				<div className={activeTab === "predict" ? "contents" : "hidden"} aria-hidden={activeTab !== "predict"}>
					<IntegrationTestPanel />
				</div>
				<div className={activeTab === "train" ? "contents" : "hidden"} aria-hidden={activeTab !== "train"}>
					<TrainingPanel />
				</div>
				<div className={activeTab === "models" ? "contents" : "hidden"} aria-hidden={activeTab !== "models"}>
					<ModelManagementPanel />
				</div>
			</main>
		</div>
	);
}
