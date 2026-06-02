export const sleep = (ms: number) =>
  new Promise((resolve) => window.setTimeout(resolve, ms))

export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `${res.status} ${res.statusText}`)
  }
  return res.json()
}
