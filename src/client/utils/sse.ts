/** Parse a fetch SSE response, dispatching events to handlers. */
export async function consumeSSE(
  response: Response,
  handlers: {
    onProgress?: (message: string) => void;
    onResult?: (data: unknown) => void;
    onError?: (error: string) => void;
  },
  signal?: AbortSignal,
): Promise<void> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventType = "";

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
          if (eventType === "progress") handlers.onProgress?.(data.message);
          else if (eventType === "result") handlers.onResult?.(data);
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
