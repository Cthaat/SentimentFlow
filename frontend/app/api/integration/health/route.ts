import { NextResponse } from "next/server";

const configuredBase = process.env.BACKEND_API_URL || process.env.NEXT_SERVER_API_BASE_URL;
const fallbackBases = ["http://backend:8000", "http://localhost:8000"];

async function requestHealth() {
	const baseCandidates = configuredBase ? [configuredBase] : fallbackBases;
	let lastError: unknown = null;

	for (const base of baseCandidates) {
		try {
			const response = await fetch(`${base}/health`, {
				method: "GET",
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

export async function GET() {
	try {
		const { base, response, data } = await requestHealth();
		return NextResponse.json(
			{
				ok: response.ok,
				upstreamStatus: response.status,
				baseUrl: base,
				payload: data,
			},
			{ status: response.ok ? 200 : 502 },
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
