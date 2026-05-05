// Progress broadcasting — lets clients reconnect to an in-progress job and
// catch up from the last announced message.

let _lastProgress = "";
const _progressListeners = new Set<(msg: string) => void>();

export function broadcastProgress(msg: string) {
  if (msg === _lastProgress) return;
  _lastProgress = msg;
  for (const listener of _progressListeners) listener(msg);
}

export function getLastProgress(): string {
  return _lastProgress;
}

export function subscribeProgress(listener: (msg: string) => void): () => void {
  _progressListeners.add(listener);
  return () => _progressListeners.delete(listener);
}
