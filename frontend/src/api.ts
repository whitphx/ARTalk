export const sleep = (ms: number) =>
  new Promise((resolve) => window.setTimeout(resolve, ms))

export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const text = await res.text()
    let detail: unknown
    if (text) {
      try {
        const payload = JSON.parse(text) as { detail?: unknown }
        detail = payload.detail
      } catch {
        // Fall through to the raw response body when the error is not JSON.
      }
    }
    if (typeof detail === 'string') {
      throw new Error(detail)
    }
    throw new Error(text || `${res.status} ${res.statusText}`)
  }
  return res.json()
}
