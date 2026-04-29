import { NextResponse } from "next/server";

const configuredBase = process.env.BACKEND_API_URL || process.env.NEXT_SERVER_API_BASE_URL;
const fallbackBases = ["http://backend:8000", "http://localhost:8000"];

export async function GET() {
  const baseCandidates = configuredBase ? [configuredBase] : fallbackBases;
  let lastError: unknown = null;

  for (const base of baseCandidates) {
    try {
      const response = await fetch(`${base}/api/models/active`, { cache: "no-store" });
      const data = await response.json();
      return NextResponse.json({ ok: response.ok, upstreamStatus: response.status, baseUrl: base, payload: data }, { status: response.ok ? 200 : 502 });
    } catch (error) {
      lastError = error;
    }
  }
  return NextResponse.json({ ok: false, error: lastError instanceof Error ? lastError.message : "Backend unreachable" }, { status: 500 });
}

export async function PUT(request: Request) {
  const baseCandidates = configuredBase ? [configuredBase] : fallbackBases;

  let body: { model_type?: string; model_path?: string } = {};
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body" }, { status: 400 });
  }

  let lastError: unknown = null;
  for (const base of baseCandidates) {
    try {
      const response = await fetch(`${base}/api/models/active`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await response.json();
      return NextResponse.json({ ok: response.ok, upstreamStatus: response.status, baseUrl: base, payload: data }, { status: response.ok ? 200 : 502 });
    } catch (error) {
      lastError = error;
    }
  }
  return NextResponse.json({ ok: false, error: lastError instanceof Error ? lastError.message : "Backend unreachable" }, { status: 500 });
}
