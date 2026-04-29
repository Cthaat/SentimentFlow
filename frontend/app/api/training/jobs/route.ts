import { NextResponse } from "next/server";
import { proxyToBackend } from "@/lib/api-proxy";

export async function GET() {
  try {
    const { response, baseUrl } = await proxyToBackend("/api/training/jobs", { cache: "no-store" });
    const data = await response.json();
    return NextResponse.json({ ok: response.ok, upstreamStatus: response.status, baseUrl, payload: data }, { status: response.ok ? 200 : response.status });
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "Backend unreachable" }, { status: 500 });
  }
}
