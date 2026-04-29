import { NextResponse } from "next/server";

const configuredBase = process.env.BACKEND_API_URL || process.env.NEXT_SERVER_API_BASE_URL;
const fallbackBases = ["http://backend:8000", "http://localhost:8000"];

type PredictRequestBody = {
	text?: string;
	model?: string;
};

async function requestPredict(text: string, model?: string) {
	const baseCandidates = configuredBase ? [configuredBase] : fallbackBases;
	let lastError: unknown = null;

	for (const base of baseCandidates) {
		try {
			const body: Record<string, string> = { text };
			if (model) body.model = model;
			const response = await fetch(`${base}/api/predict/`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify(body),
				cache: "no-store",
			});
			const data = await response.json();
			return { base, response, data };
		} catch (error) {
			lastError = error;
		}
	}

	throw lastError || new Error("Backend is unreachable");
}

export async function POST(request: Request) {
	let body: PredictRequestBody = {};

	try {
		body = (await request.json()) as PredictRequestBody;
	} catch {
		return NextResponse.json({ ok: false, error: "Invalid JSON body" }, { status: 400 });
	}

	const text = (body.text || "").trim();
	if (!text) {
		return NextResponse.json({ ok: false, error: "text is required" }, { status: 400 });
	}

	try {
		const { base, response, data } = await requestPredict(text, body.model);
		return NextResponse.json(
			{
				ok: response.ok,
				upstreamStatus: response.status,
				baseUrl: base,
				payload: data,
			},
			{ status: response.ok ? 200 : response.status },
		);
	} catch (error) {
		return NextResponse.json(
			{
				ok: false,
				error: error instanceof Error ? error.message : "Unknown error",
			},
			{ status: 500 },
		);
	}
}
