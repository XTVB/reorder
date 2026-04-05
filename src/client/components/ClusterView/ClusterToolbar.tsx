import React, { useCallback, useEffect, useState } from "react";
import type { DistanceProfile, WeightConfig } from "../../types.ts";

const DEFAULT_N_CLUSTERS = 200;

const WEIGHT_PRESETS: { label: string; weights: WeightConfig }[] = [
  { label: "PE-Core G + color", weights: { pecore_g: 1.0, color: 0.5 } },
  { label: "CLIP + color (legacy)", weights: { clip: 1.0, color: 0.5 } },
  { label: "PE-Core L", weights: { pecore_l: 1.0 } },
  { label: "PE-Core G + CLIP + color", weights: { pecore_g: 1.0, clip: 1.0, color: 0.5 } },
  { label: "PE-Core L + CLIP + color", weights: { pecore_l: 1.0, clip: 1.0, color: 0.5 } },
];

const WEIGHT_LABELS: { key: keyof Required<WeightConfig>; label: string }[] = [
  { key: "clip", label: "CLIP" },
  { key: "dino", label: "DINOv2" },
  { key: "dinov3", label: "DINOv3" },
  { key: "pecore_l", label: "PE-L" },
  { key: "pecore_g", label: "PE-G" },
  { key: "color", label: "Color" },
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
  suggestedCounts: number[];
  totalClusters: number;
  hasError: boolean;
  distanceProfile: DistanceProfile | null;
  weights: WeightConfig;
  usePatches: boolean;
  onRun: (n?: number) => void;
  onRecut: (n: number) => void;
  onRecutByThreshold: (threshold: number) => void;
  onRecutAdaptive: (minClusterSize: number) => void;
  onWeightsChange: (w: WeightConfig) => void;
  onUsePatchesChange: (v: boolean) => void;
  onExpandAll: () => void;
  onCollapseAll: () => void;
  onAcceptAll: (minSize: number) => void;
}

export function ClusterToolbar({
  loading,
  progress,
  nClusters,
  suggestedCounts,
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
  onExpandAll,
  onCollapseAll,
  onAcceptAll,
}: Props) {
  const [customN, setCustomN] = useState(String(nClusters));
  const [sliderPos, setSliderPos] = useState(500);
  const [previewCount, setPreviewCount] = useState<number | null>(null);
  const [showWeights, setShowWeights] = useState(false);
  const [minClusterSize, setMinClusterSize] = useState(5);

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
        onClick={() => onRun(parseInt(customN) || DEFAULT_N_CLUSTERS)}
        disabled={loading}
      >
        {loading ? "Clustering..." : "Run Clustering"}
      </button>

      {/* Weights selector */}
      <div className="cluster-weights-control">
        <button
          className="btn btn-small"
          onClick={() => setShowWeights(!showWeights)}
          title="Configure embedding weights"
        >
          {activeWeightParts.join(", ") || "No weights"}
        </button>
        {showWeights && (
          <div className="cluster-weights-dropdown">
            <div className="cluster-weights-presets">
              {WEIGHT_PRESETS.map((p, i) => (
                <button
                  key={i}
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
              {WEIGHT_LABELS.map(({ key, label }) => (
                <label key={key} className="cluster-weight-row">
                  <span className="cluster-weight-label">{label}</span>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={weights[key] ?? 0}
                    onChange={(e) =>
                      onWeightsChange({ ...weights, [key]: parseFloat(e.target.value) })
                    }
                  />
                  <span className="cluster-weight-value">{(weights[key] ?? 0).toFixed(1)}</span>
                </label>
              ))}
            </div>
          </div>
        )}
      </div>

      <label className="cluster-patches-toggle" title="Use DINOv3 patch-level distances instead of global embeddings (better for distinguishing specific outfits/locations, ~30s extra)">
        <input
          type="checkbox"
          checked={usePatches}
          onChange={(e) => onUsePatchesChange(e.target.checked)}
        />
        Patches
      </label>

      {/* Threshold slider or N input */}
      {hasProfile ? (
        <label className="cluster-threshold-control">
          <span className="cluster-threshold-label">Coarse</span>
          <input
            type="range"
            className="cluster-threshold-slider"
            min={0}
            max={1000}
            value={sliderPos}
            onChange={(e) => handleSliderInput(parseInt(e.target.value))}
            onMouseUp={(e) => handleSliderCommit(parseInt((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => handleSliderCommit(parseInt((e.target as HTMLInputElement).value))}
            disabled={loading}
          />
          <span className="cluster-threshold-label">Fine</span>
          <span
            className="cluster-threshold-count"
            title={`threshold: ${sliderToThreshold(sliderPos).toFixed(4)}`}
          >
            → {displayCount} clusters
          </span>
        </label>
      ) : (
        <label className="cluster-n-selector">
          N=
          <input
            type="number"
            value={customN}
            onChange={(e) => setCustomN(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") onRecut(parseInt(customN) || DEFAULT_N_CLUSTERS);
            }}
            className="cluster-n-input"
            min={2}
            disabled={loading}
          />
          <button
            className="btn btn-small"
            onClick={() => onRecut(parseInt(customN) || DEFAULT_N_CLUSTERS)}
            disabled={loading || !totalClusters}
          >
            Re-cut
          </button>
        </label>
      )}

      {/* HDBSCAN stability-based adaptive cutting */}
      {hasProfile && (
        <label className="cluster-threshold-control">
          <span className="cluster-threshold-label">Stability</span>
          <input
            type="range"
            className="cluster-threshold-slider"
            min={2}
            max={30}
            value={minClusterSize}
            onChange={(e) => setMinClusterSize(parseInt(e.target.value))}
            onMouseUp={(e) => onRecutAdaptive(parseInt((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => onRecutAdaptive(parseInt((e.target as HTMLInputElement).value))}
            disabled={loading}
          />
          <span
            className="cluster-threshold-count"
            title="Min cluster size — smaller = more granular clusters"
          >
            min {minClusterSize}
          </span>
        </label>
      )}

      {(loading || hasError) && (
        <span className={`cluster-progress ${hasError ? "cluster-error" : ""}`}>{progress}</span>
      )}
      {totalClusters > 0 && !hasProfile && (
        <span className="cluster-count">{totalClusters} clusters</span>
      )}
      <button className="btn btn-small" onClick={onExpandAll} disabled={!totalClusters}>
        Expand All
      </button>
      <button className="btn btn-small" onClick={onCollapseAll} disabled={!totalClusters}>
        Collapse All
      </button>
      <button className="btn btn-success" onClick={() => onAcceptAll(minClusterSize)} disabled={!totalClusters}>
        Accept All
      </button>
    </>
  );
}
