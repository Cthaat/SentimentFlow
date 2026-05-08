import { NextResponse } from "next/server";
import { proxyToBackend } from "@/lib/api-proxy";

export async function POST(request: Request) {
  let body: { job_id?: string } = {};
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body" }, { status: 400 });
  }
  if (!body.job_id) {
    return NextResponse.json({ ok: false, error: "job_id is required" }, { status: 400 });
  }

  try {
    const { response, baseUrl } = await proxyToBackend(`/api/training/cancel/${body.job_id}`, { method: "POST" });
    const data = await response.json();
    return NextResponse.json({ ok: response.ok, upstreamStatus: response.status, baseUrl, payload: data }, { status: response.ok ? 200 : response.status });
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "Backend unreachable" }, { status: 500 });
  }
}
