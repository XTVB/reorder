import { useCallback, useRef, useState } from "react";
import { useDismissOnOutside } from "../../hooks/useDismissOnOutside.ts";
import { useToastStore } from "../../stores/core/toastStore.ts";
import { useListStore } from "../../stores/modes/cluster/listStore.ts";
import type {
  DistanceProfile,
  ImportClusterInput,
  LinkageMethod,
  WeightConfig,
} from "../../types.ts";
import { getErrorMessage } from "../../utils/helpers.ts";
import { OverflowMenu, OverflowMenuDivider, OverflowMenuItem } from "../shared/OverflowMenu.tsx";

const WEIGHT_PRESETS: { label: string; weights: WeightConfig }[] = [
  {
    label: "Learned 3-head",
    weights: { learned_proj: 0.3, learned_proj_peg: 0.55, learned_proj_color: 0.15 },
  },
  { label: "Learned 60% + zs", weights: { pecore_g: 1.0, color: 0.7, learned_proj: 0.6 } },
  { label: "PE-G + Color", weights: { pecore_g: 1.0, color: 0.5 } },
  { label: "PE-G + Color (high)", weights: { pecore_g: 1.0, color: 0.8 } },
  { label: "PE-G + Color + DINOv3", weights: { pecore_g: 2.0, color: 1.0, dinov3: 0.5 } },
  { label: "DINOv3", weights: { dinov3: 1.0 } },
];

// The learned dials are target fractions of the final signal (rendered as %,
// rescaled server-side); the zero-shot rows are raw concat weights.
const LEARNED_WEIGHT_KEYS = new Set(["learned_proj", "learned_proj_peg", "learned_proj_color"]);

const WEIGHT_LABELS: { key: keyof Required<WeightConfig>; label: string }[] = [
  { key: "dinov3", label: "DINOv3" },
  { key: "pecore_g", label: "PE-G" },
  { key: "color", label: "Color" },
  { key: "learned_proj", label: "l-head" },
  { key: "learned_proj_peg", label: "l-PE-G" },
  { key: "learned_proj_color", label: "l-color" },
];

interface Props {
  loading: boolean;
  progress: string;
  totalClusters: number;
  hasError: boolean;
  distanceProfile: DistanceProfile | null;
  weights: WeightConfig;
  usePatches: boolean;
  useRerank: boolean;
  rerankBlend: number;
  linkage: LinkageMethod;
  onRecut: (n: number) => void;
  onRecutAdaptive: (minClusterSize: number) => void;
  onWeightsChange: (w: WeightConfig) => void;
  onUsePatchesChange: (v: boolean) => void;
  onUseRerankChange: (v: boolean) => void;
  onRerankBlendChange: (v: number) => void;
  onLinkageChange: (v: LinkageMethod) => void;
  onExpandAll: () => void;
  onCollapseAll: () => void;
  onAcceptAll: (minSize: number) => void;
  onImportClusters: (payload: { clusters: ImportClusterInput[] }) => void;
  onClearImported: () => void;
}

export function ClusterToolbar({
  loading,
  progress,
  totalClusters,
  hasError,
  distanceProfile,
  weights,
  onRecut,
  onRecutAdaptive,
  onWeightsChange,
  usePatches,
  onUsePatchesChange,
  useRerank,
  rerankBlend,
  onUseRerankChange,
  onRerankBlendChange,
  linkage,
  onLinkageChange,
  onExpandAll,
  onCollapseAll,
  onAcceptAll,
  onImportClusters,
  onClearImported,
}: Props) {
  const desiredN = useListStore((s) => s.desiredN);
  const setDesiredN = useListStore((s) => s.setDesiredN);
  const [showSettings, setShowSettings] = useState(false);
  const [customMin, setCustomMin] = useState("5");
  const minClusterSize = parseInt(customMin, 10) || 5;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const settingsRef = useRef<HTMLDivElement>(null);
  const showToast = useToastStore((s) => s.showToast);

  const handleImportFile = useCallback(
    async (file: File) => {
      try {
        const text = await file.text();
        const parsed = JSON.parse(text);
        if (!parsed || !Array.isArray(parsed.clusters)) {
          showToast("Import JSON must be an object with a `clusters` array", "error");
          return;
        }
        onImportClusters(parsed);
      } catch (err) {
        showToast(`Failed to parse JSON: ${getErrorMessage(err, "parse error")}`, "error");
      }
    },
    [onImportClusters, showToast],
  );

  useDismissOnOutside(settingsRef, showSettings, () => setShowSettings(false));

  const hasProfile = distanceProfile && distanceProfile.distances.length > 0;

  // Short summary of the active blend for the popover trigger
  const activeWeightParts = WEIGHT_LABELS.filter(({ key }) => (weights[key] ?? 0) > 0);
  const blendSummary =
    activeWeightParts.length === 3 &&
    activeWeightParts.every(({ key }) => LEARNED_WEIGHT_KEYS.has(key))
      ? "Learned 3-head"
      : activeWeightParts.map(({ label }) => label).join(" + ") || "No weights";

  return (
    <>
      {/* One segmented island: cut tuning · blend popover · overflow */}
      <div className="toolbar-group">
        <label className="toolbar-field" title="Cut the tree into exactly N clusters">
          N
          <input
            type="number"
            value={desiredN}
            onChange={(e) => setDesiredN(parseInt(e.target.value, 10) || 0)}
            onKeyDown={(e) => {
              if (e.key === "Enter") onRecut(desiredN);
            }}
            className="cluster-n-input"
            min={2}
            disabled={loading}
          />
        </label>
        <button
          className="btn btn-secondary"
          onClick={() => onRecut(desiredN)}
          disabled={loading || !totalClusters}
          title="Re-cut the cached tree into exactly N clusters"
        >
          Re-cut
        </button>
        {hasProfile && (
          <>
            <div className="toolbar-divider" />
            <label
              className="toolbar-field"
              title="Min cluster size — adaptive (HDBSCAN-style) cut; smaller = more granular clusters"
            >
              Min
              <input
                type="number"
                value={customMin}
                onChange={(e) => setCustomMin(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") onRecutAdaptive(minClusterSize);
                }}
                className="cluster-n-input cluster-n-input--narrow"
                min={2}
                max={30}
                disabled={loading}
              />
            </label>
            <button
              className="btn btn-secondary"
              onClick={() => onRecutAdaptive(minClusterSize)}
              disabled={loading || !totalClusters}
              title="Adaptive re-cut with this minimum cluster size"
            >
              Adaptive
            </button>
          </>
        )}
        <div className="toolbar-divider" />

        {/* Blend + rarely-touched knobs live in one settings popover */}
        <div className="cluster-weights-control" ref={settingsRef}>
          <button
            className="btn btn-secondary"
            onClick={() => setShowSettings(!showSettings)}
            aria-expanded={showSettings}
            title="Embedding blend, linkage and distance options"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" role="presentation">
              <path
                d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h13M20 18h0"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
              <circle cx="16" cy="6" r="2" stroke="currentColor" strokeWidth="2" />
              <circle cx="10" cy="12" r="2" stroke="currentColor" strokeWidth="2" />
              <circle cx="18.5" cy="18" r="2" stroke="currentColor" strokeWidth="2" />
            </svg>
            {blendSummary}
          </button>
          {showSettings && (
            <div className="cluster-weights-dropdown">
              <div className="cluster-settings-section">Presets</div>
              <div className="cluster-weights-presets">
                {WEIGHT_PRESETS.map((p) => (
                  <button
                    key={p.label}
                    className="btn btn-small"
                    onClick={() => {
                      onWeightsChange(p.weights);
                    }}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
              <div className="cluster-weights-sliders">
                {WEIGHT_LABELS.map(({ key, label }) => {
                  // The learned-head sliders are target contribution fractions
                  // vs raw concat-weights for the others
                  const isLearned = LEARNED_WEIGHT_KEYS.has(key);
                  const max = isLearned ? "1" : "2";
                  const step = isLearned ? "0.05" : "0.1";
                  const v = weights[key] ?? 0;
                  const displayValue = isLearned ? `${Math.round(v * 100)}%` : v.toFixed(1);
                  return (
                    <label key={key} className="cluster-weight-row">
                      <span className="cluster-weight-label">{label}</span>
                      <input
                        type="range"
                        min="0"
                        max={max}
                        step={step}
                        value={v}
                        onChange={(e) =>
                          onWeightsChange({ ...weights, [key]: parseFloat(e.target.value) })
                        }
                      />
                      <span className="cluster-weight-value">{displayValue}</span>
                    </label>
                  );
                })}
              </div>
              <div className="cluster-settings-section">Distance</div>
              <label
                className="cluster-patches-toggle"
                title="Linkage method for the cluster tree. Ward (default) suits evenly-sized compact shoots; Average suits a few large or uneven sets; Complete suits tight clusters of moderately uneven size."
              >
                Linkage
                <select
                  className="cluster-linkage-select"
                  value={linkage}
                  onChange={(e) => onLinkageChange(e.target.value as LinkageMethod)}
                >
                  <option value="ward">Ward</option>
                  <option value="average">Average</option>
                  <option value="complete">Complete</option>
                </select>
              </label>
              <label
                className="cluster-patches-toggle"
                title="k-reciprocal re-ranking — kNN-graph structure on top of cosine (~3s precompute). Useful on datasets with large, well-separated shoots."
              >
                <input
                  type="checkbox"
                  checked={useRerank}
                  onChange={(e) => onUseRerankChange(e.target.checked)}
                />
                Re-rank
              </label>
              {useRerank && (
                <label
                  className="cluster-patches-toggle"
                  title="Re-rank blend: 0 = cosine only, 1 = re-rank only. Default 0.7."
                >
                  <span className="cluster-rerank-blend__hint">blend</span>
                  <input
                    className="cluster-rerank-blend__slider"
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={rerankBlend}
                    onChange={(e) => onRerankBlendChange(parseFloat(e.target.value))}
                  />
                  <span className="cluster-rerank-blend__value">{rerankBlend.toFixed(2)}</span>
                </label>
              )}
              <label
                className="cluster-patches-toggle"
                title="Use DINOv3 patch-level distances (mutually exclusive with re-rank — re-rank wins if both enabled)"
              >
                <input
                  type="checkbox"
                  checked={usePatches}
                  disabled={useRerank}
                  onChange={(e) => onUsePatchesChange(e.target.checked)}
                />
                Patches
              </label>
            </div>
          )}
        </div>

        <div className="toolbar-divider" />
        <OverflowMenu label="More cluster actions" align="right">
          <OverflowMenuItem onClick={onExpandAll} disabled={!totalClusters}>
            Expand all
          </OverflowMenuItem>
          <OverflowMenuItem onClick={onCollapseAll} disabled={!totalClusters}>
            Collapse all
          </OverflowMenuItem>
          <OverflowMenuDivider />
          <OverflowMenuItem onClick={() => onAcceptAll(minClusterSize)} disabled={!totalClusters}>
            Accept all
          </OverflowMenuItem>
          <OverflowMenuDivider />
          <OverflowMenuItem
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            title="Import clusters from a JSON file (bypasses the clustering pipeline)"
          >
            Import JSON…
          </OverflowMenuItem>
          <OverflowMenuItem
            danger
            closeBeforeClick
            onClick={() => {
              if (
                confirm("Clear imported clusters? The linkage-tree cache (if any) will remain.")
              ) {
                onClearImported();
              }
            }}
            disabled={loading}
            title="Delete the imported-clusters cache so the view falls back to the linkage tree"
          >
            Clear import
          </OverflowMenuItem>
        </OverflowMenu>
      </div>

      {(loading || hasError) && (
        <span className={`cluster-progress ${hasError ? "cluster-error" : ""}`}>{progress}</span>
      )}
      <input
        ref={fileInputRef}
        type="file"
        accept="application/json,.json"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleImportFile(file);
          e.target.value = "";
        }}
      />
    </>
  );
}
