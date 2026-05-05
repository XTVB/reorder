// One-at-a-time slot for expensive cluster compute jobs (extract, cluster,
// recut, scoped). Distinct from withRenameLock (fs/lock.ts) which guards
// on-disk filesystem state. A second compute job request returns 409 while
// one is running, and aborting cancels the running subprocess via SIGINT.

import { log } from "../log.ts";

let _abort: AbortController | null = null;

export function isClusterJobRunning(): boolean {
  return _abort !== null;
}

export function setClusterJobRunning(v: boolean): void {
  _abort = v ? new AbortController() : null;
}

export function getClusterAbortSignal(): AbortSignal | undefined {
  return _abort?.signal;
}

export function cancelClusterJob(): void {
  if (_abort) {
    _abort.abort();
    log("cluster", "Cluster job cancellation requested");
  }
}
