// Paths to subprocess executables.

import { dirname, join } from "node:path";

export const SCRIPTS_DIR = join(dirname(import.meta.dir), "..", "scripts");
export const RUST_BINARY = join(
  dirname(import.meta.dir),
  "..",
  "rust",
  "cluster-tool",
  "target",
  "release",
  "cluster-tool",
);
export const GROUP_SIM_BINARY = join(
  dirname(import.meta.dir),
  "..",
  "rust",
  "group-similarity",
  "target",
  "release",
  "group-similarity",
);
export const ORDER_BINARY = join(
  dirname(import.meta.dir),
  "..",
  "rust",
  "order-tool",
  "target",
  "release",
  "order-tool",
);
export const PYTHON =
  process.env.CLUSTER_PYTHON || `${process.env.HOME}/.venvs/imgcluster-env/bin/python3`;
