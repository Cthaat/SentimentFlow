import { NextResponse } from "next/server";
import { proxyToBackend } from "@/lib/api-proxy";

const TRAINING_START_TIMEOUT_MS = 120000;

export async function POST(request: Request) {
  let body: { model_type?: string; config?: Record<string, unknown> } = {};
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body" }, { status: 400 });
  }
  if (!body.model_type) {
    return NextResponse.json({ ok: false, error: "model_type is required" }, { status: 400 });
  }

  try {
    const { response, baseUrl } = await proxyToBackend("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }, TRAINING_START_TIMEOUT_MS);
    const data = await response.json();
    return NextResponse.json({ ok: response.ok, upstreamStatus: response.status, baseUrl, payload: data }, { status: response.ok ? 200 : response.status });
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "Backend unreachable" }, { status: 500 });
  }
}
