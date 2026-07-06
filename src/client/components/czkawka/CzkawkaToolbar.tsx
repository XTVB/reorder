import { useState } from "react";
import {
  HASH_ALGS,
  HASH_SIZES,
  type HashSize,
  IMAGE_FILTERS,
  useCzkawkaStore,
} from "../../stores/czkawkaStore.ts";
import { cn } from "../../utils/helpers.ts";
import { DirsPanel } from "./DirsPanel.tsx";

export function CzkawkaToolbar() {
  const hashAlg = useCzkawkaStore((s) => s.hashAlg);
  const imageFilter = useCzkawkaStore((s) => s.imageFilter);
  const similarity = useCzkawkaStore((s) => s.similarity);
  const hashSize = useCzkawkaStore((s) => s.hashSize);
  const loading = useCzkawkaStore((s) => s.loading);
  const dirCount = useCzkawkaStore((s) => s.dirs.length);
  const hasReference = useCzkawkaStore((s) => s.dirs.some((d) => d.reference));
  const dirsDirty = useCzkawkaStore((s) => s.dirsDirty);
  const setHashAlg = useCzkawkaStore((s) => s.setHashAlg);
  const setImageFilter = useCzkawkaStore((s) => s.setImageFilter);
  const setSimilarity = useCzkawkaStore((s) => s.setSimilarity);
  const setHashSize = useCzkawkaStore((s) => s.setHashSize);
  const runComparison = useCzkawkaStore((s) => s.runComparison);

  const [dirsOpen, setDirsOpen] = useState(false);

  return (
    <div className="czkawka-toolbar">
      <div className="czkawka-toolbar-config">
        <label className="czkawka-toolbar-field">
          Hash
          <select
            value={hashAlg}
            onChange={(e) => setHashAlg(e.target.value as typeof hashAlg)}
            disabled={loading}
          >
            {HASH_ALGS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </label>
        <label className="czkawka-toolbar-field">
          Filter
          <select
            value={imageFilter}
            onChange={(e) => setImageFilter(e.target.value as typeof imageFilter)}
            disabled={loading}
          >
            {IMAGE_FILTERS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </label>
        <label className="czkawka-toolbar-field">
          Size
          <select
            value={String(hashSize)}
            onChange={(e) => setHashSize(Number(e.target.value) as HashSize)}
            disabled={loading}
          >
            {HASH_SIZES.map((opt) => (
              <option key={String(opt)} value={String(opt)}>
                {opt}
              </option>
            ))}
          </select>
        </label>
        <label className="czkawka-toolbar-field">
          Similarity
          <input
            type="number"
            min={0}
            max={500}
            value={similarity}
            onChange={(e) => setSimilarity(e.target.valueAsNumber || 0)}
            className="czkawka-input czkawka-input--narrow"
            disabled={loading}
          />
        </label>
      </div>
      <button
        className={cn("btn", "czkawka-dirs-btn", dirsDirty && "is-dirty")}
        onClick={() => setDirsOpen((v) => !v)}
        title="Directories in the comparison"
      >
        Folders{dirCount > 1 ? ` · ${dirCount}` : ""}
        {hasReference && <span className="czkawka-dirs-btn-ref">REF</span>}
      </button>
      <button className="btn btn-primary" onClick={runComparison} disabled={loading}>
        {loading ? "Running…" : "Run Comparison"}
      </button>
      <DirsPanel open={dirsOpen} onClose={() => setDirsOpen(false)} />
    </div>
  );
}
