import { useEffect } from "react";
import { createRoot } from "react-dom/client";
import { ClusterView } from "./components/cluster/ClusterView.tsx";
import {
  AppShellHeader,
  DEFAULT_MODE,
  MODES,
  modeFromPath,
} from "./components/header/AppShellHeader.tsx";
import { HeaderActions } from "./components/header/HeaderActions.tsx";
import { MergeSuggestions } from "./components/merge-suggestions/MergeSuggestions.tsx";
import { ReorderToolbarOverflow } from "./components/reorder/ReorderToolbarOverflow.tsx";
import { ReorderView } from "./components/reorder/ReorderView.tsx";
import { Lightbox } from "./components/shared/Lightbox.tsx";
import { Toast } from "./components/shared/Toast.tsx";
import { useRouter } from "./hooks/useRouter.ts";
import { useConstraintsStore } from "./stores/constraintsStore.ts";
import { useLightboxStore } from "./stores/core/lightboxStore.ts";
import { useModalStore } from "./stores/core/modalStore.ts";
import { useSelectionStore } from "./stores/core/selectionStore.ts";
import { useSessionStore } from "./stores/core/sessionStore.ts";
import { useToastStore } from "./stores/core/toastStore.ts";
import { useDndStore } from "./stores/dndStore.ts";
import { useGroupStore } from "./stores/groupStore.ts";
import { useImageStore } from "./stores/imageStore.ts";
import { useMergeSuggestionsStore } from "./stores/mergeSuggestionsStore.ts";
import {
  useExpandStore,
  useInteractionsStore,
  useListStore,
  useMetricsStore,
  useSplitStore,
} from "./stores/modes/cluster/index.ts";
import { useSortHistoryStore } from "./stores/sortHistoryStore.ts";
import { useTrashStore } from "./stores/trashStore.ts";

// Expose all stores on window for console access / debugging
(window as unknown as Record<string, unknown>).__stores = {
  images: useImageStore,
  groups: useGroupStore,
  selection: useSelectionStore,
  dnd: useDndStore,
  modal: useModalStore,
  lightbox: useLightboxStore,
  toast: useToastStore,
  session: useSessionStore,
  clusterList: useListStore,
  clusterInteractions: useInteractionsStore,
  clusterExpand: useExpandStore,
  clusterSplit: useSplitStore,
  clusterMetrics: useMetricsStore,
  mergeSuggestions: useMergeSuggestionsStore,
  constraints: useConstraintsStore,
  trash: useTrashStore,
  sortHistory: useSortHistoryStore,
};

function AppShell() {
  const { pathname, navigate } = useRouter();
  const mode = modeFromPath(pathname);

  // biome-ignore lint/correctness/useExhaustiveDependencies: navigate is a stable ref from useRouter
  useEffect(() => {
    if (!MODES.some((m) => m.path === pathname)) {
      navigate(MODES.find((m) => m.key === DEFAULT_MODE)!.path);
    }
  }, [pathname]);

  useEffect(() => {
    useConstraintsStore.getState().loadConstraints();
  }, []);

  return (
    <>
      <AppShellHeader
        mode={mode}
        navigate={navigate}
        leftSlot={mode === "reorder" ? <ReorderToolbarOverflow /> : null}
      >
        <HeaderActions mode={mode} />
      </AppShellHeader>
      {mode === "cluster" ? (
        <ClusterView />
      ) : mode === "merge-suggestions" ? (
        <MergeSuggestions />
      ) : (
        <ReorderView />
      )}
      <Lightbox />
      <Toast />
    </>
  );
}

createRoot(document.getElementById("root")!).render(<AppShell />);
