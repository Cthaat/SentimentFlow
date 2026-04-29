const configuredBase = process.env.BACKEND_API_URL || process.env.NEXT_SERVER_API_BASE_URL;
const fallbackBases = ["http://backend:8000", "http://localhost:8000"];

export async function GET(
  request: Request,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params;
  const baseCandidates = configuredBase ? [configuredBase] : fallbackBases;

  // SSE streaming proxy — use the first working backend
  for (const base of baseCandidates) {
    try {
      const response = await fetch(`${base}/api/training/stream/${jobId}`, {
        headers: { Accept: "text/event-stream" },
      });
      if (response.ok) {
        return new Response(response.body, {
          status: 200,
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
          },
        });
      }
    } catch {
      // try next
    }
  }
  return new Response("data: {\"error\":\"Backend unreachable\"}\n\n", {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}
