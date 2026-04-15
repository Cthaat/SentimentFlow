import { IntegrationTestPanel } from "@/components/integration-test-panel";

export default function Home() {
	return (
		<div className="min-h-screen bg-zinc-50 px-4 py-10">
			<main className="mx-auto flex w-full max-w-4xl flex-col items-center gap-6">
				<h1 className="text-2xl font-semibold tracking-tight">SentimentFlow 联调测试</h1>
				<IntegrationTestPanel />
			</main>
		</div>
	);
}
