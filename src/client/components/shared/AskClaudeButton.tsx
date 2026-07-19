import { postJson } from "../../api/client.ts";
import { useToastStore } from "../../stores/core/toastStore.ts";
import { copyContactSheetToClipboard, getErrorMessage } from "../../utils/helpers.ts";

const DEFAULT_PROMPT = "Give me a descriptive sexy title for this photoset, 4-6 words";

interface AskClaudeButtonProps {
  images: string[];
  name: string;
  prompt?: string;
}

export function AskClaudeButton({ images, name, prompt = DEFAULT_PROMPT }: AskClaudeButtonProps) {
  const showToast = useToastStore((s) => s.showToast);

  async function handleClick() {
    try {
      const result = await postJson<{ filename?: string }>("/api/cluster/contact-sheet", {
        filenames: images,
        clusterName: name,
      });
      if (!result.filename) return;
      await copyContactSheetToClipboard(result.filename, prompt);
      showToast("Contact sheet + prompt copied to clipboard", "success");
    } catch (err) {
      showToast(getErrorMessage(err, "Failed to generate contact sheet"), "error");
    }
  }

  return (
    <button
      className="btn btn-small btn-row-action"
      onClick={handleClick}
      title="Generate contact sheet for Claude"
    >
      Ask Claude
    </button>
  );
}
