import { useEffect } from "react";
import { useImageStore } from "../stores/imageStore.ts";
import { useGroupStore } from "../stores/groupStore.ts";
import { useUIStore } from "../stores/uiStore.ts";
import { useTagStore } from "../stores/tagStore.ts";
import { getErrorMessage } from "../utils/helpers.ts";

export function useTagViewInit() {
  const fetchImages = useImageStore((s) => s.fetchImages);
  const fetchGroups = useGroupStore((s) => s.fetchGroups);
  const setError = useUIStore((s) => s.setError);
  const fetchTargetDir = useUIStore((s) => s.fetchTargetDir);
  const checkUndo = useUIStore((s) => s.checkUndo);
  const checkDbStatus = useTagStore((s) => s.checkDbStatus);
  const loadAllTags = useTagStore((s) => s.loadAllTags);

  useEffect(() => {
    fetchTargetDir();
    fetchImages().catch((err: unknown) => {
      setError(getErrorMessage(err, "Failed to load images"));
    });
    checkUndo();
    fetchGroups();
  }, [fetchImages, checkUndo, fetchTargetDir, fetchGroups, setError]);

  useEffect(() => {
    checkDbStatus().then(() => {
      const state = useTagStore.getState();
      if (state.hasDb && !state.indexReady) {
        loadAllTags();
      }
    });
  }, [checkDbStatus, loadAllTags]);
}
