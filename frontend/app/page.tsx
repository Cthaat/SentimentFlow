"use client";
import { IntegrationTestPanel } from "@/components/integration-test-panel";
import { ThemeToggle } from "@/components/theme-toggle";

export default function Home() {
	return (
		<div className="min-h-screen bg-zinc-50 px-4 py-10 dark:bg-zinc-900">
			<main className="mx-auto flex w-full max-w-4xl flex-col items-center gap-6">
				<div className="flex w-full items-center justify-between">
					<h1 className="text-2xl font-semibold tracking-tight">SentimentFlow 联调测试</h1>
					<ThemeToggle />
				</div>
				<IntegrationTestPanel />
			</main>
		</div>
	);
}
