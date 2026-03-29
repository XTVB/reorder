import React, { useRef, useState } from "react";
import { useTagStore } from "../../stores/tagStore.ts";
import { useUIStore } from "../../stores/uiStore.ts";

export function IngestPrompt() {
  const ingestFile = useTagStore((s) => s.ingestFile);
  const showToast = useUIStore((s) => s.showToast);
  const inputRef = useRef<HTMLInputElement>(null);
  const [ingesting, setIngesting] = useState(false);

  async function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setIngesting(true);
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const result = await ingestFile(data);
      showToast(`Ingested ${result.ingested} images (${result.skipped} skipped)`, "success");
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Ingest failed", "error");
    } finally {
      setIngesting(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className="ingest-prompt">
      <div className="ingest-prompt-icon">🏷</div>
      <div className="ingest-prompt-text">
        No tags database found. Upload a JSON file matching the tag schema to get started.
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".json"
        className="ingest-prompt-input"
        onChange={handleFileSelect}
      />
      <button
        className="btn btn-primary"
        onClick={() => inputRef.current?.click()}
        disabled={ingesting}
      >
        {ingesting ? "Ingesting..." : "Load Tags File"}
      </button>
    </div>
  );
}
