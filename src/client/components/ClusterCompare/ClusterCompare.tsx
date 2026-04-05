import { useVirtualizer } from "@tanstack/react-virtual";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useUIStore } from "../../stores/uiStore.ts";
import type { ClusterData, ClusterResultData, WeightConfig } from "../../types.ts";
import { cn, imageUrl, postJson } from "../../utils/helpers.ts";
import { consumeSSE } from "../../utils/sse.ts";
import { Lightbox } from "../Lightbox.tsx";

const EMPTY_CLUSTERS: ClusterResultData[] = [];

// ── Weight keys and labels ──────────────────────────────────────────────────

type Weights = Required<WeightConfig>;

const WEIGHT_KEYS: { key: keyof Weights; label: string; desc: string }[] = [
  { key: "clip", label: "CLIP", desc: "ViT-B/32 (512d, semantic)" },
  { key: "dino", label: "DINOv2", desc: "ViT-L/14 (1024d, visual)" },
  { key: "dinov3", label: "DINOv3", desc: "ViT-B/16 (768d, instance)" },
  { key: "pecore_l", label: "PE-Core L", desc: "L-14-336 (1024d)" },
  { key: "pecore_g", label: "PE-Core G", desc: "bigG-14-448 (1280d)" },
  { key: "color", label: "Color", desc: "HSV+RGB (77d)" },
];

const ZERO_WEIGHTS: Weights = { clip: 0, dino: 0, dinov3: 0, pecore_l: 0, pecore_g: 0, color: 0 };

const PRESETS: { label: string; weights: Weights }[] = [
  { label: "CLIP + color (old baseline)", weights: { ...ZERO_WEIGHTS, clip: 1.0, color: 0.5 } },
  { label: "PE-Core L only", weights: { ...ZERO_WEIGHTS, pecore_l: 1.0 } },
  { label: "PE-Core G only", weights: { ...ZERO_WEIGHTS, pecore_g: 1.0 } },
  { label: "PE-Core L + color", weights: { ...ZERO_WEIGHTS, pecore_l: 1.0, color: 0.5 } },
  { label: "PE-Core G + color", weights: { ...ZERO_WEIGHTS, pecore_g: 1.0, color: 0.5 } },
  { label: "PE-Core G + CLIP", weights: { ...ZERO_WEIGHTS, pecore_g: 1.0, clip: 1.0 } },
  {
    label: "PE-Core G + CLIP + color",
    weights: { ...ZERO_WEIGHTS, pecore_g: 1.0, clip: 1.0, color: 0.5 },
  },
  {
    label: "PE-Core L + CLIP + color",
    weights: { ...ZERO_WEIGHTS, pecore_l: 1.0, clip: 1.0, color: 0.5 },
  },
  { label: "DINOv2 only", weights: { ...ZERO_WEIGHTS, dino: 1.0 } },
  {
    label: "DINOv2 + CLIP + color",
    weights: { ...ZERO_WEIGHTS, dino: 1.0, clip: 1.0, color: 0.5 },
  },
  { label: "PE-Core G + DINOv2", weights: { ...ZERO_WEIGHTS, pecore_g: 1.0, dino: 1.0 } },
  { label: "DINOv3 only", weights: { ...ZERO_WEIGHTS, dinov3: 1.0 } },
  { label: "DINOv3 + color", weights: { ...ZERO_WEIGHTS, dinov3: 1.0, color: 0.5 } },
  {
    label: "DINOv3 + PE-Core G + color",
    weights: { ...ZERO_WEIGHTS, dinov3: 1.0, pecore_g: 1.0, color: 0.5 },
  },
  {
    label: "All models equal",
    weights: { clip: 1.0, dino: 1.0, dinov3: 1.0, pecore_l: 1.0, pecore_g: 1.0, color: 0.5 },
  },
];

// ── Tab state ───────────────────────────────────────────────────────────────

interface Tab {
  id: string;
  label: string;
  weights: Weights;
  data: ClusterData | null;
  loading: boolean;
  error: string | null;
}

function makeTab(label: string, weights: Weights): Tab {
  return { id: crypto.randomUUID(), label, weights, data: null, loading: false, error: null };
}

// ── Component ───────────────────────────────────────────────────────────────

export function ClusterCompare() {
  const [tabs, setTabs] = useState<Tab[]>(() => [
    makeTab("CLIP + color (old)", { ...ZERO_WEIGHTS, clip: 1.0, color: 0.5 }),
    makeTab("PE-Core L only", { ...ZERO_WEIGHTS, pecore_l: 1.0 }),
    makeTab("PE-Core G only", { ...ZERO_WEIGHTS, pecore_g: 1.0 }),
  ]);
  const [activeTabId, setActiveTabId] = useState(tabs[0]!.id);
  const [nClusters, setNClusters] = useState(200);
  const [lightbox, setLightbox] = useState<{ clusterId: string; imageIndex: number } | null>(null);
  const [showAddPanel, setShowAddPanel] = useState(false);
  const [customWeights, setCustomWeights] = useState<Weights>({ ...ZERO_WEIGHTS });
  const [customLabel, setCustomLabel] = useState("Custom");
  const [extracting, setExtracting] = useState(false);
  const [extractProgress, setExtractProgress] = useState("");
  const extractAbortRef = useRef<AbortController | null>(null);
  const setHeaderSubtitle = useUIStore((s) => s.setHeaderSubtitle);

  // Abort extraction SSE on unmount
  useEffect(
    () => () => {
      extractAbortRef.current?.abort();
    },
    [],
  );

  // Reconnect to in-progress extraction on mount/refresh
  useEffect(() => {
    fetch("/api/cluster/status")
      .then((r) => r.json())
      .then(({ running, progress }) => {
        if (!running) return;
        setExtracting(true);
        setExtractProgress(progress || "Extraction in progress...");
        const abort = new AbortController();
        extractAbortRef.current = abort;
        fetch("/api/cluster/progress", { signal: abort.signal })
          .then(async (res) => {
            await consumeSSE(
              res,
              {
                onProgress: (msg) => setExtractProgress(msg),
                onResult: () => {
                  setExtracting(false);
                  setExtractProgress("");
                },
              },
              abort.signal,
            );
            setExtracting(false);
            setExtractProgress("");
          })
          .catch(() => {});
      });
  }, []);

  async function runExtraction(models?: string[]) {
    setExtracting(true);
    setExtractProgress(
      models ? `Extracting ${models.join(", ")}...` : "Extracting all missing models...",
    );

    const abort = new AbortController();
    extractAbortRef.current = abort;

    try {
      const res = await fetch("/api/cluster/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(models ? { models } : {}),
        signal: abort.signal,
      });
      await consumeSSE(
        res,
        {
          onProgress: (msg) => setExtractProgress(msg),
          onResult: () => {
            setExtracting(false);
            setExtractProgress("");
          },
          onError: (err) => {
            setExtracting(false);
            setExtractProgress(`Error: ${err}`);
          },
        },
        abort.signal,
      );
    } catch (err) {
      if (!abort.signal.aborted) {
        setExtracting(false);
        setExtractProgress(`Error: ${err}`);
      }
    }
  }

  const activeTab = tabs.find((t) => t.id === activeTabId) ?? tabs[0]!;
  const clusters = activeTab.data?.clusters ?? EMPTY_CLUSTERS;

  const loadedCount = tabs.filter((t) => t.data).length;
  // biome-ignore lint/correctness/useExhaustiveDependencies: setHeaderSubtitle is a stable Zustand action
  useEffect(() => {
    setHeaderSubtitle(`Compare \u2014 ${loadedCount}/${tabs.length} loaded`);
    return () => setHeaderSubtitle("");
  }, [loadedCount, tabs.length]);

  async function runAll() {
    for (const tab of tabs) {
      await runTab(tab.id);
    }
  }

  async function runTab(tabId: string) {
    const tab = tabs.find((t) => t.id === tabId);
    if (!tab) return;
    setTabs((prev) => prev.map((t) => (t.id === tabId ? { ...t, loading: true, error: null } : t)));
    try {
      const res = await postJson("/api/cluster/test", { nClusters, weights: tab.weights });
      const data: ClusterData = await res.json();
      setTabs((prev) => prev.map((t) => (t.id === tabId ? { ...t, data, loading: false } : t)));
    } catch (err) {
      setTabs((prev) =>
        prev.map((t) =>
          t.id === tabId
            ? {
                ...t,
                loading: false,
                error: err instanceof Error ? err.message : String(err),
              }
            : t,
        ),
      );
    }
  }

  function addPreset(preset: (typeof PRESETS)[0]) {
    const tab = makeTab(preset.label, { ...preset.weights });
    setTabs((prev) => [...prev, tab]);
    setActiveTabId(tab.id);
    setShowAddPanel(false);
  }

  function addCustom() {
    const tab = makeTab(customLabel || "Custom", { ...customWeights });
    setTabs((prev) => [...prev, tab]);
    setActiveTabId(tab.id);
    setShowAddPanel(false);
    setCustomWeights({ ...ZERO_WEIGHTS });
    setCustomLabel("Custom");
  }

  function removeTab(tabId: string) {
    setTabs((prev) => {
      const next = prev.filter((t) => t.id !== tabId);
      if (activeTabId === tabId && next.length > 0) setActiveTabId(next[0]!.id);
      return next;
    });
  }

  // Virtualization
  const scrollRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: clusters.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: (index) => {
      const cluster = clusters[index];
      if (!cluster) return 56;
      const containerWidth = scrollRef.current?.clientWidth ?? 960;
      const cols = Math.max(1, Math.floor(containerWidth / 168));
      const thumbRows = Math.ceil(cluster.images.length / cols);
      return 56 + thumbRows * 176 + 32;
    },
    measureElement: (el) => el.getBoundingClientRect().height,
    overscan: 3,
  });

  // biome-ignore lint/correctness/useExhaustiveDependencies: clusters is the intentional trigger for re-measuring row heights
  useEffect(() => {
    virtualizer.measure();
  }, [clusters]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: activeTabId is the intentional trigger for resetting scroll and re-measuring
  useEffect(() => {
    scrollRef.current?.scrollTo(0, 0);
    virtualizer.measure();
  }, [activeTabId]);

  const virtualItems = virtualizer.getVirtualItems();

  return (
    <div className="cluster-view">
      {/* Controls bar */}
      <div className="compare-controls">
        <label>
          N clusters:
          <input
            type="number"
            value={nClusters}
            onChange={(e) => setNClusters(Number(e.target.value))}
            min={2}
            max={1000}
            className="compare-n-input"
          />
        </label>
        <button className="btn btn-primary" onClick={runAll}>
          Run All Tabs
        </button>
        <button className="btn" onClick={() => setShowAddPanel(!showAddPanel)}>
          + Add Config
        </button>
        <span className="compare-separator" />
        <span className="compare-extract-label">Extract:</span>
        {WEIGHT_KEYS.map(({ key, label }) => (
          <button
            key={key}
            className="btn btn-small"
            disabled={extracting}
            onClick={() => runExtraction([key])}
            title={`Extract ${label} embeddings`}
          >
            {label}
          </button>
        ))}
        <button className="btn btn-small" disabled={extracting} onClick={() => runExtraction()}>
          All Missing
        </button>
      </div>
      {(extracting || extractProgress) && (
        <div className="compare-extract-status">
          {extracting && <span className="compare-spinner" />}
          <span>{extractProgress}</span>
        </div>
      )}

      {/* Add config panel */}
      {showAddPanel && (
        <div className="compare-add-panel">
          <div className="compare-add-section">
            <div className="compare-add-heading">Presets</div>
            <div className="compare-preset-grid">
              {PRESETS.map((p, i) => (
                <button key={i} className="btn btn-small" onClick={() => addPreset(p)}>
                  {p.label}
                </button>
              ))}
            </div>
          </div>
          <div className="compare-add-section">
            <div className="compare-add-heading">Custom</div>
            <div className="compare-custom-form">
              <input
                className="compare-custom-label"
                value={customLabel}
                onChange={(e) => setCustomLabel(e.target.value)}
                placeholder="Label"
              />
              <div className="compare-sliders">
                {WEIGHT_KEYS.map(({ key, label, desc }) => (
                  <div key={key} className="compare-slider-row">
                    <span className="compare-slider-label" title={desc}>
                      {label}
                    </span>
                    <input
                      type="range"
                      min="0"
                      max="2"
                      step="0.1"
                      value={customWeights[key]}
                      onChange={(e) =>
                        setCustomWeights((w) => ({ ...w, [key]: parseFloat(e.target.value) }))
                      }
                    />
                    <span className="compare-slider-value">{customWeights[key].toFixed(1)}</span>
                  </div>
                ))}
              </div>
              <button className="btn btn-primary btn-small" onClick={addCustom}>
                Add
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="compare-tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={cn("compare-tab", tab.id === activeTabId && "active")}
            onClick={() => setActiveTabId(tab.id)}
          >
            <span className="compare-tab-label">{tab.label}</span>
            <span className="compare-tab-status">
              {tab.loading
                ? "..."
                : tab.data
                  ? `${tab.data.clusters.length} clusters`
                  : tab.error
                    ? "error"
                    : "---"}
            </span>
            <span
              className="compare-tab-weights"
              title={WEIGHT_KEYS.map((k) => `${k.label}=${tab.weights[k.key]}`).join(", ")}
            >
              {WEIGHT_KEYS.filter((k) => tab.weights[k.key] > 0)
                .map((k) => `${k.label[0]}${tab.weights[k.key]}`)
                .join(" ")}
            </span>
            {tabs.length > 1 && (
              <span
                className="compare-tab-close"
                onClick={(e) => {
                  e.stopPropagation();
                  removeTab(tab.id);
                }}
              >
                &times;
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Active tab weight summary */}
      <div className="compare-weight-summary">
        {WEIGHT_KEYS.map(({ key, label }) => (
          <span
            key={key}
            className={cn("compare-weight-pill", activeTab.weights[key] > 0 && "active")}
          >
            {label}: {activeTab.weights[key].toFixed(1)}
          </span>
        ))}
        {!activeTab.data && !activeTab.loading && (
          <button className="btn btn-small btn-primary" onClick={() => runTab(activeTab.id)}>
            Run
          </button>
        )}
      </div>

      {/* Content */}
      {activeTab.loading && (
        <div className="cluster-empty-state">
          <div className="cluster-empty-title">Running linkage...</div>
          <div className="cluster-empty-desc">{activeTab.label}</div>
        </div>
      )}

      {activeTab.error && (
        <div className="cluster-empty-state">
          <div className="cluster-empty-title">Error</div>
          <div className="cluster-empty-desc">{activeTab.error}</div>
        </div>
      )}

      {!activeTab.data && !activeTab.loading && !activeTab.error && (
        <div className="cluster-empty-state">
          <div className="cluster-empty-title">No results</div>
          <div className="cluster-empty-desc">Click "Run All Tabs" or the Run button above</div>
        </div>
      )}

      {activeTab.data && (
        <div ref={scrollRef} className="cluster-list">
          <div style={{ height: virtualizer.getTotalSize(), width: "100%", position: "relative" }}>
            {virtualItems.map((vi) => {
              const cluster = clusters[vi.index];
              if (!cluster) return null;
              return (
                <div
                  key={cluster.id}
                  data-index={vi.index}
                  ref={virtualizer.measureElement}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    transform: `translateY(${vi.start}px)`,
                  }}
                >
                  <CompareClusterCard
                    cluster={cluster}
                    rank={vi.index + 1}
                    onOpenLightbox={(idx) =>
                      setLightbox({ clusterId: cluster.id, imageIndex: idx })
                    }
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {lightbox &&
        (() => {
          const cluster = clusters.find((c) => c.id === lightbox.clusterId);
          if (!cluster) return null;
          return (
            <Lightbox
              images={cluster.images.map((f) => ({ filename: f }))}
              initialIndex={lightbox.imageIndex}
              onClose={() => setLightbox(null)}
            />
          );
        })()}
    </div>
  );
}

// ── Cluster card (read-only) ────────────────────────────────────────────────

function CompareClusterCard({
  cluster,
  rank,
  onOpenLightbox,
}: {
  cluster: ClusterResultData;
  rank: number;
  onOpenLightbox: (index: number) => void;
}) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="cluster-card">
      <div className="cluster-header" onClick={() => setCollapsed(!collapsed)}>
        <span
          className={`cluster-chevron ${collapsed ? "" : "cluster-chevron-open"}`}
          aria-hidden
        />
        <span className="cluster-name">
          #{rank} {cluster.autoName}
        </span>
        <span className="cluster-count">{cluster.images.length} images</span>
        {!collapsed &&
          cluster.autoTags.slice(0, 4).map((t) => (
            <span key={t.term} className="cluster-tag" title={`z=${t.z.toFixed(1)}`}>
              {t.term}
            </span>
          ))}
      </div>
      {!collapsed && (
        <div className="cluster-body">
          <div className="cluster-thumbs">
            {cluster.images.map((f, i) => (
              <div key={f} className="cluster-thumb" onClick={() => onOpenLightbox(i)}>
                <img src={imageUrl(f)} loading="lazy" decoding="async" alt={f} draggable={false} />
                <span className="cluster-thumb-name">{f}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
