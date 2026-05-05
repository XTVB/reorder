// Single mutex serializing all on-disk filesystem mutations in the target
// directory (rename, organize, group save, folder save, delete, recovery).
// Read endpoints that depend on rename consistency (e.g. /api/images,
// /api/groups GET) also hold this lock, so they don't observe a half-applied
// rename.
//
// This is distinct from the cluster compute-job lock (cluster/job-mutex.ts),
// which guards GPU/CPU-expensive work and not on-disk consistency.

let _lock: Promise<unknown> = Promise.resolve();

export function withRenameLock<T>(fn: () => Promise<T>): Promise<T> {
  const prev = _lock;
  let resolve: () => void;
  _lock = new Promise<void>((r) => {
    resolve = r;
  });
  return prev.then(fn).finally(() => resolve!());
}
