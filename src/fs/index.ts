// Public surface of the fs/ module — used by server routes. Aggregated
// so consumers don't need to know the internal layout.

export type { RenameMapping } from "../shared/types.ts";
export { backupPath, readJsonTolerant, writeJsonAtomic } from "./atomic-json.ts";
export type {
  CzkawkaActionEntry,
  CzkawkaSessionData,
  CzkawkaTrashEntry,
} from "./czkawka-session.ts";
export {
  loadCzkawkaSession,
  restoreFromTrash,
  saveCzkawkaSession,
  trashFilesRestorable,
} from "./czkawka-session.ts";
export type { FolderData, FolderSaveRequest } from "./folder-save.ts";
export { executeFolderSave, listFolderData, listSubdirectories } from "./folder-save.ts";
export { loadGroups, writeGroupsFile } from "./groups.ts";
export { assertFilesExist, assertWritable, extractTitle } from "./helpers.ts";
export { isImageFile, listImages } from "./images.ts";
export { withRenameLock } from "./lock.ts";
export type { OrganizeGroup, OrganizeMapping, OrganizeOptions } from "./organize.ts";
export { computeOrganize, executeOrganize } from "./organize.ts";
export {
  cacheDir,
  constraintsPath,
  contactSheetsDir,
  contentHashesPath,
  contentHashesTmpPath,
  czkawkaHashCachePath,
  czkawkaSessionPath,
  groupsBackupPath,
  groupsPath,
  historyPath,
  historyPrevPath,
  importedClustersPath,
  linkageTreePath,
  logPath,
  patchDistMatrixPath,
  pendingFolderSavePath,
  pendingRenamePath,
  rerankDistMatrixPath,
  tagsPath,
} from "./paths.ts";
export type { RecoveryResult } from "./recovery.ts";
export { recoverPendingRename } from "./recovery.ts";
export {
  canUndo,
  computeRenames,
  executeRenames,
  twoPhaseRename,
  undoRenames,
} from "./rename.ts";
export { readTagsJson, writeRemappedTags } from "./tags.ts";
export type { DeleteResult } from "./trash.ts";
export { executeDelete } from "./trash.ts";
