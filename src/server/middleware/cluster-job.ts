// Wraps the SSE-route shape every long-running cluster operation shares:
// 409 if the job mutex is already held; otherwise take it, fan progress out
// to the broadcast channel + the SSE stream, and release the mutex on exit.

import {
  broadcastProgress,
  isClusterJobRunning,
  setClusterJobRunning,
} from "../../cluster/index.ts";
import { json } from "./response.ts";
import { type SSESend, sseResponse } from "./sse.ts";

export type ClusterJobFn<T> = (send: SSESend, onProgress: (line: string) => void) => Promise<T>;

/**
 * Run an exclusive cluster job behind an SSE response. Returns 409 JSON if
 * another job is in flight. Otherwise streams `progress`/`result` events from
 * `jobFn`. Progress lines are also broadcast for /api/cluster/progress
 * re-attachers. The mutex is released and progress is cleared on completion.
 */
export function runClusterJobSSE<T>(jobFn: ClusterJobFn<T>, conflictMessage: string): Response {
  if (isClusterJobRunning()) {
    return json({ error: conflictMessage }, 409);
  }
  setClusterJobRunning(true);
  return sseResponse(async (send) => {
    try {
      const onProgress = (line: string) => {
        broadcastProgress(line);
        send("progress", { message: line });
      };
      const result = await jobFn(send, onProgress);
      broadcastProgress("");
      send("result", result);
    } finally {
      broadcastProgress("");
      setClusterJobRunning(false);
    }
  });
}
