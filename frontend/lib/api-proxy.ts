/**
 * 后端代理工具 — 同时尝试多个后端候选地址，取最先响应的结果。
 * 避免因不可达地址（如 Docker 内部域名 backend:8000）造成的长时间等待。
 */

const DEFAULT_CONNECTION_TIMEOUT_MS = 10000;

const configuredBase = process.env.BACKEND_API_URL || process.env.NEXT_SERVER_API_BASE_URL;
const configuredTimeoutMs = Number(process.env.BACKEND_API_TIMEOUT_MS);

export const CONNECTION_TIMEOUT_MS =
  Number.isFinite(configuredTimeoutMs) && configuredTimeoutMs > 0
    ? configuredTimeoutMs
    : DEFAULT_CONNECTION_TIMEOUT_MS;

export function getBaseCandidates(): string[] {
  if (configuredBase) {
    return [configuredBase];
  }
  // localhost 放前面，避免 backend:8000 DNS 超时拖慢本地开发
  return ["http://localhost:8000", "http://backend:8000"];
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
 * 向所有候选后端发送请求，竞速返回第一个成功响应。
 * 若全部失败，抛出最后一个错误。
 */
export async function proxyToBackend(
  path: string,
  init: RequestInit = {},
  timeoutMs: number = CONNECTION_TIMEOUT_MS,
): Promise<{ response: Response; baseUrl: string }> {
  const bases = getBaseCandidates();

  // 竞速：任一后端先响应即返回，不等待其他
  return new Promise((resolve, reject) => {
    let pending = bases.length;
    const errors: unknown[] = [];

    for (const base of bases) {
      fetchWithTimeout(`${base}${path}`, init, timeoutMs)
        .then((response) => {
          resolve({ response, baseUrl: base });
        })
        .catch((error) => {
          errors.push(error);
          pending--;
          if (pending === 0) {
            reject(errors[errors.length - 1] || new Error("All backend candidates unreachable"));
          }
        });
    }
  });
}
