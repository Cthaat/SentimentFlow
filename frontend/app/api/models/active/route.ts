import { NextResponse } from "next/server";
import { proxyToBackend } from "@/lib/api-proxy";

export async function GET() {
  try {
    const { response, baseUrl } = await proxyToBackend("/api/models/active", { cache: "no-store" });
    const data = await response.json();
    return NextResponse.json({ ok: response.ok, upstreamStatus: response.status, baseUrl, payload: data }, { status: response.ok ? 200 : response.status });
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "Backend unreachable" }, { status: 500 });
  }
}

export async function PUT(request: Request) {
  let body: { model_type?: string; model_path?: string } = {};
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const { response, baseUrl } = await proxyToBackend("/api/models/active", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    return NextResponse.json({ ok: response.ok, upstreamStatus: response.status, baseUrl, payload: data }, { status: response.ok ? 200 : response.status });
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "Backend unreachable" }, { status: 500 });
  }
}
