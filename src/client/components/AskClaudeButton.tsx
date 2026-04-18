import { useUIStore } from "../stores/uiStore.ts";
import { copyContactSheetToClipboard, getErrorMessage, postJson } from "../utils/helpers.ts";

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
      await copyContactSheetToClipboard(result.filename, prompt);
      showToast("Contact sheet + prompt copied to clipboard", "success");
    } catch (err) {
      showToast(getErrorMessage(err, "Failed to generate contact sheet"), "error");
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
