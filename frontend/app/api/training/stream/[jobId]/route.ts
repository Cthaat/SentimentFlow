import { getBaseCandidates } from "@/lib/api-proxy";

const CONNECTION_TIMEOUT_MS = 3000;

export async function GET(
  request: Request,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params;
  const bases = getBaseCandidates();

  // 竞速返回第一个可用的 SSE 流
  return new Promise<Response>((resolve) => {
    let pending = bases.length;

    for (const base of bases) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), CONNECTION_TIMEOUT_MS);

      fetch(`${base}/api/training/stream/${jobId}`, {
        headers: { Accept: "text/event-stream" },
        signal: controller.signal,
      })
        .then((response) => {
          clearTimeout(timer);
          if (response.ok) {
            resolve(new Response(response.body, {
              status: 200,
              headers: {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                Connection: "keep-alive",
              },
            }));
          } else {
            pending--;
            if (pending === 0) {
              resolve(new Response("data: {\"error\":\"Backend unreachable\"}\n\n", {
                status: 200,
                headers: { "Content-Type": "text/event-stream" },
              }));
            }
          }
        })
        .catch(() => {
          clearTimeout(timer);
          pending--;
          if (pending === 0) {
            resolve(new Response("data: {\"error\":\"Backend unreachable\"}\n\n", {
              status: 200,
              headers: { "Content-Type": "text/event-stream" },
            }));
          }
        });
    }
  });
}
