import { useCallback, useEffect, useRef, useState } from "react";
import { useToastStore } from "../../stores/core/toastStore.ts";
import type { DistanceProfile, ImportClusterInput, WeightConfig } from "../../types.ts";
import { getErrorMessage } from "../../utils/helpers.ts";
import { OverflowMenu, OverflowMenuDivider, OverflowMenuItem } from "../shared/OverflowMenu.tsx";

const DEFAULT_N_CLUSTERS = 200;

const WEIGHT_PRESETS: { label: string; weights: WeightConfig }[] = [
  { label: "PE-G + Color", weights: { pecore_g: 1.0, color: 0.5 } },
  { label: "PE-G + Color (high)", weights: { pecore_g: 1.0, color: 0.8 } },
  { label: "PE-G + Color + DINOv3", weights: { pecore_g: 2.0, color: 1.0, dinov3: 0.5 } },
  { label: "DINOv3", weights: { dinov3: 1.0 } },
];

const WEIGHT_LABELS: { key: keyof Required<WeightConfig>; label: string }[] = [
  { key: "dinov3", label: "DINOv3" },
  { key: "pecore_g", label: "PE-G" },
  { key: "color", label: "Color" },
  { key: "learned_proj", label: "Learned head" },
];

/** Binary search: count how many sorted distances are < threshold */
function countMergesBelow(distances: number[], threshold: number): number {
  let lo = 0,
    hi = distances.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (distances[mid]! < threshold) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

interface Props {
  loading: boolean;
  progress: string;
  nClusters: number;
  totalClusters: number;
  hasError: boolean;
  distanceProfile: DistanceProfile | null;
  weights: WeightConfig;
  usePatches: boolean;
  useRerank: boolean;
  rerankBlend: number;
  onRun: (n?: number) => void;
  onRecut: (n: number) => void;
  onRecutByThreshold: (threshold: number) => void;
  onRecutAdaptive: (minClusterSize: number) => void;
  onWeightsChange: (w: WeightConfig) => void;
  onUsePatchesChange: (v: boolean) => void;
  onUseRerankChange: (v: boolean) => void;
  onRerankBlendChange: (v: number) => void;
  onExpandAll: () => void;
  onCollapseAll: () => void;
  onAcceptAll: (minSize: number) => void;
  onImportClusters: (payload: { clusters: ImportClusterInput[] }) => void;
  onClearImported: () => void;
}

export function ClusterToolbar({
  loading,
  progress,
  nClusters,
  totalClusters,
  hasError,
  distanceProfile,
  weights,
  onRun,
  onRecut,
  onRecutByThreshold,
  onRecutAdaptive,
  onWeightsChange,
  usePatches,
  onUsePatchesChange,
  useRerank,
  rerankBlend,
  onUseRerankChange,
  onRerankBlendChange,
  onExpandAll,
  onCollapseAll,
  onAcceptAll,
  onImportClusters,
  onClearImported,
}: Props) {
  const [customN, setCustomN] = useState(String(nClusters));
  const [sliderPos, setSliderPos] = useState(500);
  const [previewCount, setPreviewCount] = useState<number | null>(null);
  const [showWeights, setShowWeights] = useState(false);
  const [minClusterSize, setMinClusterSize] = useState(5);
  const fileInputRef = useRef<HTMLInputElement>(null);
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

  useEffect(() => {
    setCustomN(String(nClusters));
  }, [nClusters]);

  const sliderToThreshold = useCallback(
    (pos: number): number => {
      if (!distanceProfile || distanceProfile.distances.length === 0) return 0;
      const { distances } = distanceProfile;
      const frac = 1 - pos / 1000;
      const idx = frac * (distances.length - 1);
      const lo = Math.floor(idx);
      const hi = Math.ceil(idx);
      if (lo === hi) return distances[lo]!;
      const t = idx - lo;
      return distances[lo]! * (1 - t) + distances[hi]! * t;
    },
    [distanceProfile],
  );

  const computePreview = useCallback(
    (pos: number) => {
      if (!distanceProfile) return null;
      const threshold = sliderToThreshold(pos);
      const merges = countMergesBelow(distanceProfile.distances, threshold);
      return distanceProfile.nAfterPremerge - merges;
    },
    [distanceProfile, sliderToThreshold],
  );

  useEffect(() => {
    if (!distanceProfile || distanceProfile.distances.length === 0) return;
    const { distances, nAfterPremerge } = distanceProfile;
    const mergesNeeded = nAfterPremerge - nClusters;
    if (mergesNeeded <= 0) {
      setSliderPos(1000);
      return;
    }
    if (mergesNeeded >= distances.length) {
      setSliderPos(0);
      return;
    }
    const pos = Math.round((1 - mergesNeeded / (distances.length - 1)) * 1000);
    setSliderPos(Math.min(1000, Math.max(0, pos)));
  }, [distanceProfile, nClusters]);

  function handleSliderInput(pos: number) {
    setSliderPos(pos);
    setPreviewCount(computePreview(pos));
  }

  function handleSliderCommit(pos: number) {
    setPreviewCount(null);
    if (!distanceProfile) return;
    const threshold = sliderToThreshold(pos);
    onRecutByThreshold(threshold);
  }

  const hasProfile = distanceProfile && distanceProfile.distances.length > 0;
  const displayCount = previewCount ?? totalClusters;

  // Active weights summary
  const activeWeightParts = WEIGHT_LABELS.filter(({ key }) => (weights[key] ?? 0) > 0).map(
    ({ key, label }) => `${label}=${weights[key]}`,
  );

  return (
    <>
      <button
        className="btn btn-primary"
        onClick={() => onRun(parseInt(customN, 10) || DEFAULT_N_CLUSTERS)}
        disabled={loading}
      >
        {loading ? "Clustering..." : "Run Clustering"}
      </button>

      {/* Configuration: weights + patches toggle */}
      <div className="toolbar-group" title="Embedding configuration">
        <div className="cluster-weights-control">
          <button
            className="btn"
            onClick={() => setShowWeights(!showWeights)}
            title="Configure embedding weights"
          >
            {activeWeightParts.join(", ") || "No weights"}
          </button>
          {showWeights && (
            <div className="cluster-weights-dropdown">
              <div className="cluster-weights-presets">
                {WEIGHT_PRESETS.map((p) => (
                  <button
                    key={p.label}
                    className="btn btn-small"
                    onClick={() => {
                      onWeightsChange(p.weights);
                      setShowWeights(false);
                    }}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
              <div className="cluster-weights-sliders">
                {WEIGHT_LABELS.map(({ key, label }) => {
                  // The learned-head slider is a target contribution fraction
                  // vs raw concat-weights for the others
                  const isLearned = key === "learned_proj";
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
            </div>
          )}
        </div>

        <label
          className="cluster-patches-toggle"
          title="k-reciprocal re-ranking — uses kNN graph structure on top of cosine for more accurate clustering (~3s precompute, recommended)"
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

      {/* Tuning: threshold slider(s) or N input */}
      {hasProfile ? (
        <div className="toolbar-group" title="Cut tuning">
          <ThresholdSlider
            label="Cut"
            title="Drag left for coarser (fewer) clusters, right for finer (more) clusters"
            min={0}
            max={1000}
            value={sliderPos}
            disabled={loading}
            onInput={handleSliderInput}
            onCommit={handleSliderCommit}
            countText={String(displayCount)}
            countTitle={`threshold: ${sliderToThreshold(sliderPos).toFixed(4)}`}
          />
          <ThresholdSlider
            label="Min"
            title="Min cluster size — smaller = more granular clusters"
            min={2}
            max={30}
            value={minClusterSize}
            disabled={loading}
            onInput={setMinClusterSize}
            onCommit={onRecutAdaptive}
            countText={String(minClusterSize)}
          />
        </div>
      ) : (
        <label className="cluster-n-selector">
          N=
          <input
            type="number"
            value={customN}
            onChange={(e) => setCustomN(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") onRecut(parseInt(customN, 10) || DEFAULT_N_CLUSTERS);
            }}
            className="cluster-n-input"
            min={2}
            disabled={loading}
          />
          <button
            className="btn btn-small"
            onClick={() => onRecut(parseInt(customN, 10) || DEFAULT_N_CLUSTERS)}
            disabled={loading || !totalClusters}
          >
            Re-cut
          </button>
        </label>
      )}

      {(loading || hasError) && (
        <span className={`cluster-progress ${hasError ? "cluster-error" : ""}`}>{progress}</span>
      )}
      {totalClusters > 0 && !hasProfile && (
        <span className="cluster-count">{totalClusters} clusters</span>
      )}

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
            if (confirm("Clear imported clusters? The linkage-tree cache (if any) will remain.")) {
              onClearImported();
            }
          }}
          disabled={loading}
          title="Delete the imported-clusters cache so the view falls back to the linkage tree"
        >
          Clear import
        </OverflowMenuItem>
      </OverflowMenu>
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

interface ThresholdSliderProps {
  label: string;
  title: string;
  min: number;
  max: number;
  value: number;
  disabled: boolean;
  onInput: (v: number) => void;
  onCommit: (v: number) => void;
  countText: string;
  countTitle?: string;
}

function ThresholdSlider({
  label,
  title,
  min,
  max,
  value,
  disabled,
  onInput,
  onCommit,
  countText,
  countTitle,
}: ThresholdSliderProps) {
  return (
    <label className="cluster-threshold-control" title={title}>
      <span className="cluster-threshold-label">{label}</span>
      <input
        type="range"
        className="cluster-threshold-slider"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onInput(parseInt(e.target.value, 10))}
        onMouseUp={(e) => onCommit(parseInt((e.target as HTMLInputElement).value, 10))}
        onTouchEnd={(e) => onCommit(parseInt((e.target as HTMLInputElement).value, 10))}
        disabled={disabled}
      />
      <span className="cluster-threshold-count" title={countTitle}>
        {countText}
      </span>
    </label>
  );
}
