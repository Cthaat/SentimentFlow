import { NextResponse } from "next/server";
import { proxyToBackend } from "@/lib/api-proxy";

type PredictRequestBody = {
  text?: string;
  model?: string;
};

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

  const fetchBody: Record<string, string> = { text };
  if (body.model) fetchBody.model = body.model;

  try {
    const { response, baseUrl } = await proxyToBackend("/api/predict/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(fetchBody),
    }, 30000);
    const data = await response.json();
    return NextResponse.json({ ok: response.ok, upstreamStatus: response.status, baseUrl, payload: data }, { status: response.ok ? 200 : response.status });
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "Backend unreachable" }, { status: 500 });
  }
}
