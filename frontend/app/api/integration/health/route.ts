import { NextResponse } from "next/server";
import { proxyToBackend } from "@/lib/api-proxy";

const HEALTH_TOTAL_TIMEOUT_MS = 30000;
const HEALTH_ATTEMPT_TIMEOUT_MS = 5000;
const HEALTH_RETRY_DELAY_MS = 1000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function GET() {
  const deadline = Date.now() + HEALTH_TOTAL_TIMEOUT_MS;
  let lastError: unknown = null;

  while (Date.now() < deadline) {
    try {
      const { response, baseUrl } = await proxyToBackend(
        "/health",
        { cache: "no-store" },
        HEALTH_ATTEMPT_TIMEOUT_MS,
      );
      const data = await response.json();
      return NextResponse.json(
        { ok: response.ok, upstreamStatus: response.status, baseUrl, payload: data },
        { status: response.ok ? 200 : response.status },
      );
    } catch (error) {
      lastError = error;
      if (Date.now() + HEALTH_RETRY_DELAY_MS >= deadline) {
        break;
      }
      await sleep(HEALTH_RETRY_DELAY_MS);
    }
  }

  return NextResponse.json(
    { ok: false, error: lastError instanceof Error ? lastError.message : "Backend unreachable" },
    { status: 500 },
  );
}
