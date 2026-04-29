import { NextResponse } from "next/server";

const configuredBase = process.env.BACKEND_API_URL || process.env.NEXT_SERVER_API_BASE_URL;
const fallbackBases = ["http://backend:8000", "http://localhost:8000"];

export async function GET(
  request: Request,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params;
  const baseCandidates = configuredBase ? [configuredBase] : fallbackBases;

  let lastError: unknown = null;
  for (const base of baseCandidates) {
    try {
      const response = await fetch(`${base}/api/training/status/${jobId}`, { cache: "no-store" });
      const data = await response.json();
      return NextResponse.json({ ok: response.ok, upstreamStatus: response.status, baseUrl: base, payload: data }, { status: response.ok ? 200 : 502 });
    } catch (error) {
      lastError = error;
    }
  }
  return NextResponse.json({ ok: false, error: lastError instanceof Error ? lastError.message : "Backend unreachable" }, { status: 500 });
}
