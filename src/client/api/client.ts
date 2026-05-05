// Thin fetch wrapper. The server response envelope is intentionally mixed
// (decision documented in the refactor plan: `{success}`, `{ok}`, and raw
// payloads coexist), so this wrapper does not normalise success shapes —
// it only handles error extraction.
//
// Use `getJson` / `postJson` for everything except SSE streams (use
// `startSSE` from ./sse.ts for those, since they need access to the raw
// streaming response).

interface ErrorBody {
  error?: string;
}

async function extractError(response: Response): Promise<string> {
  try {
    const body = (await response.clone().json()) as ErrorBody;
    if (body.error) return body.error;
  } catch {}
  try {
    const text = await response.text();
    if (text) return text;
  } catch {}
  return `Request failed: ${response.status}`;
}

/** GET a URL and return the parsed JSON. Throws with the server error
 * message on non-2xx. */
export async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await extractError(response));
  return (await response.json()) as T;
}

/** POST a JSON body and return the parsed JSON response. Throws with the
 * server error message on non-2xx. */
export async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await extractError(response));
  return (await response.json()) as T;
}

/** PUT a JSON body and return the parsed JSON response. Throws on non-2xx. */
export async function putJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await extractError(response));
  return (await response.json()) as T;
}

/** DELETE a URL and return the parsed JSON response (or null if 204).
 * Throws on non-2xx. */
export async function deleteJson<T>(url: string): Promise<T | null> {
  const response = await fetch(url, { method: "DELETE" });
  if (!response.ok) throw new Error(await extractError(response));
  if (response.status === 204) return null;
  return (await response.json()) as T;
}

/** POST a JSON body and return the raw `Response` (for SSE / streaming /
 * non-JSON responses). Throws on non-2xx (and non-409, which is a valid
 * "job already running" signal — use `startSSE` for that case). */
export async function postRaw(url: string, body: unknown): Promise<Response> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await extractError(response));
  return response;
}
