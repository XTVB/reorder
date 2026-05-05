// Server-Sent-Events parser for fetch responses.
//
// `consumeSSE` parses an SSE stream and dispatches `progress`, `result`, and
// `error` events to typed handlers. `startSSE` POSTs a JSON body and returns
// either the SSE response stream or a 409-conflict signal — centralising the
// "another job is already running" pattern that every SSE consumer uses.

export interface SSEHandlers<T = unknown> {
  onProgress?: (message: string) => void;
  onResult?: (data: T) => void;
  onError?: (error: string) => void;
}

/** Parse a fetch SSE response, dispatching events to handlers. */
export async function consumeSSE<T = unknown>(
  response: Response,
  handlers: SSEHandlers<T>,
  signal?: AbortSignal,
): Promise<void> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventType = "";
  let lastProgress: string | undefined;

  try {
    while (true) {
      if (signal?.aborted) break;
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("event: ")) {
          eventType = line.slice(7);
        } else if (line.startsWith("data: ")) {
          const data = JSON.parse(line.slice(6));
          if (eventType === "progress") {
            const msg = data.message;
            if (msg !== lastProgress) {
              lastProgress = msg;
              handlers.onProgress?.(msg);
            }
          } else if (eventType === "result") handlers.onResult?.(data as T);
          else if (eventType === "error") handlers.onError?.(data.error);
          eventType = "";
        } else if (line === "") {
          eventType = "";
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** Result of `startSSE`: either the response (start streaming) or a
 * conflict signal (409 — another job is already running). */
export type SSEStartResult =
  | { kind: "ok"; response: Response }
  | { kind: "conflict"; message: string };

/**
 * POST a body as JSON and return either the SSE response stream or a 409
 * conflict signal. Centralises the "another job is running" pattern that
 * every SSE consumer site reimplements.
 */
export async function startSSE(url: string, body: unknown): Promise<SSEStartResult> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (response.status === 409) {
    let message = "Another job is already running";
    try {
      const data = (await response.json()) as { error?: string };
      if (data.error) message = data.error;
    } catch {}
    return { kind: "conflict", message };
  }
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return { kind: "ok", response };
}
