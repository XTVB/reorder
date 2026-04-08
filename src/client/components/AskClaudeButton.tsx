import { useUIStore } from "../stores/uiStore.ts";
import { postJson } from "../utils/helpers.ts";

const DEFAULT_PROMPT = "Give me a descriptive sexy title for this photoset, 4-6 words";

interface AskClaudeButtonProps {
  images: string[];
  name: string;
  prompt?: string;
}

export function AskClaudeButton({ images, name, prompt = DEFAULT_PROMPT }: AskClaudeButtonProps) {
  const showToast = useUIStore((s) => s.showToast);

  async function handleClick() {
    try {
      const res = await postJson("/api/cluster/contact-sheet", {
        filenames: images,
        clusterName: name,
      });
      const result: { filename?: string } = await res.json();
      if (!result.filename) return;

      const imgRes = await fetch(`/api/contact-sheet/${encodeURIComponent(result.filename)}`);
      const jpegBlob = await imgRes.blob();

      // Clipboard API requires image/png
      const bitmap = await createImageBitmap(jpegBlob);
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(bitmap, 0, 0);
      const pngBlob = await canvas.convertToBlob({ type: "image/png" });

      await navigator.clipboard.write([
        new ClipboardItem({
          "text/plain": new Blob([prompt], { type: "text/plain" }),
          "image/png": pngBlob,
        }),
      ]);
      showToast("Contact sheet + prompt copied to clipboard", "success");
    } catch (err) {
      showToast(`Failed to generate contact sheet: ${err}`, "error");
    }
  }

  return (
    <button
      className="btn btn-small"
      onClick={handleClick}
      title="Generate contact sheet for Claude"
    >
      Ask Claude
    </button>
  );
}
