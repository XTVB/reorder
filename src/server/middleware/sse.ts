// Server-sent events helpers. The `sseResponse` helper owns encoder, keepalive,
// and controller cleanup so route handlers only write business logic.
// `subscribeProgressSSE` re-broadcasts progress for /api/cluster/progress,
// allowing a reloading client to re-attach to a running cluster job.

import { getLastProgress, subscribeProgress } from "../../cluster/index.ts";

export type SSESend = (event: "progress" | "result" | "error", data: unknown) => void;

/**
 * Build an SSE Response that invokes `handler(send)` to produce events.
 */
export function sseResponse(handler: (send: SSESend) => Promise<void>): Response {
  const stream = new ReadableStream({
    async start(controller) {
      const enc = new TextEncoder();
      let closed = false;
      const send: SSESend = (event, data) => {
        if (closed) return;
        try {
          controller.enqueue(enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
        } catch {
          closed = true;
        }
      };
      const keepalive = setInterval(() => {
        if (closed) return;
        try {
          controller.enqueue(enc.encode(": keepalive\n\n"));
        } catch {
          closed = true;
        }
      }, 30_000);
      try {
        await handler(send);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        send("error", { error: msg });
      } finally {
        clearInterval(keepalive);
        if (!closed) {
          try {
            controller.close();
          } catch {}
        }
      }
    },
  });
  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

/**
 * SSE response that re-broadcasts progress from the global cluster job (via
 * subscribeProgress + getLastProgress). Emits `progress` events for each
 * message and a final `result` event when the job ends. Used by
 * /api/cluster/progress for re-attaching after a reload.
 */
export function subscribeProgressSSE(): Response {
  let cleanupSSE: (() => void) | null = null;
  const stream = new ReadableStream({
    start(controller) {
      const enc = new TextEncoder();
      let closed = false;
      const cleanup = () => {
        if (closed) return;
        closed = true;
        unsub();
        clearInterval(keepalive);
      };
      cleanupSSE = cleanup;
      const write = (s: string) => {
        if (closed) return;
        try {
          controller.enqueue(enc.encode(s));
        } catch {
          cleanup();
        }
      };
      const last = getLastProgress();
      if (last) write(`event: progress\ndata: ${JSON.stringify({ message: last })}\n\n`);
      const keepalive = setInterval(() => {
        write(": keepalive\n\n");
      }, 30_000);
      const unsub = subscribeProgress((msg) => {
        if (!msg) {
          write(`event: result\ndata: {}\n\n`);
          cleanup();
          try {
            controller.close();
          } catch {}
        } else {
          write(`event: progress\ndata: ${JSON.stringify({ message: msg })}\n\n`);
        }
      });
    },
    cancel() {
      cleanupSSE?.();
    },
  });
  return new Response(stream, {
    headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
  });
}
