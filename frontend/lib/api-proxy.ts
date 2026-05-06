/**
 * 后端代理工具。
 *
 * 本地开发时优先走上一次成功的后端地址；没有命中时再顺序探测候选地址。
 * 避免每次请求都并发打到不可达 hostname，导致连接堆积和偶发长时间卡住。
 */

const DEFAULT_CONNECTION_TIMEOUT_MS = 10000;

const configuredBase = process.env.BACKEND_API_URL || process.env.NEXT_SERVER_API_BASE_URL;
const configuredTimeoutMs = Number(process.env.BACKEND_API_TIMEOUT_MS);
let preferredBaseUrl: string | null = configuredBase || null;

export const CONNECTION_TIMEOUT_MS =
  Number.isFinite(configuredTimeoutMs) && configuredTimeoutMs > 0
    ? configuredTimeoutMs
    : DEFAULT_CONNECTION_TIMEOUT_MS;

export function getBaseCandidates(): string[] {
  if (configuredBase) {
    return [configuredBase];
  }
  // 127.0.0.1 比 localhost 更稳定，避免本地 IPv6/DNS 抖动。
  return ["http://127.0.0.1:8000", "http://localhost:8000", "http://backend:8000"];
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    return response;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * 顺序探测候选后端，返回第一个成功建立 HTTP 响应的地址。
 * 若全部失败，抛出最后一个错误。
 */
export async function proxyToBackend(
  path: string,
  init: RequestInit = {},
  timeoutMs: number = CONNECTION_TIMEOUT_MS,
): Promise<{ response: Response; baseUrl: string }> {
  const bases = orderBaseCandidates(getBaseCandidates());
  let lastError: unknown = null;

  for (const base of bases) {
    try {
      const response = await fetchWithTimeout(`${base}${path}`, init, timeoutMs);
      preferredBaseUrl = base;
      return { response, baseUrl: base };
    } catch (error) {
      lastError = error;
      if (base === preferredBaseUrl) {
        preferredBaseUrl = null;
      }
    }
  }

  throw lastError || new Error("All backend candidates unreachable");
}

function orderBaseCandidates(bases: string[]): string[] {
  if (!preferredBaseUrl) {
    return bases;
  }

  return [
    preferredBaseUrl,
    ...bases.filter((base) => base !== preferredBaseUrl),
  ];
}
