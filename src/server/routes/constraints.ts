// Cannot-link / group-lock / clear constraint routes.

import {
  type Constraints,
  cachedHashMapping,
  loadConstraints,
  mutateConstraints,
} from "../../cluster/index.ts";
import { json } from "../middleware/response.ts";
import type { RouteHandler } from "../types.ts";

function safeHashMaps(targetDir: string): {
  hashToFilename: Map<string, string>;
  filenameToHash: Map<string, string>;
} {
  try {
    const m = cachedHashMapping(targetDir);
    return { hashToFilename: m.hashToFilename, filenameToHash: m.filenameToHash };
  } catch {
    return { hashToFilename: new Map(), filenameToHash: new Map() };
  }
}

function constraintsResponse(targetDir: string, c: Constraints, changed: boolean) {
  const { hashToFilename } = safeHashMaps(targetDir);
  return {
    success: true,
    treeStale: changed,
    ...c,
    imageGroupCannotLinkResolved: c.imageGroupCannotLink.map((e) => ({
      imageHash: e.imageHash,
      groupId: e.groupId,
      currentFilename: hashToFilename.get(e.imageHash) ?? null,
    })),
  };
}

export const constraintsRoutes: RouteHandler = async (req, ctx) => {
  const { path, targetDir } = ctx;

  if (path === "/api/constraints" && req.method === "GET") {
    return json(constraintsResponse(targetDir, loadConstraints(targetDir), false));
  }

  if (path === "/api/constraints/cannot-link" && req.method === "POST") {
    const body = (await req.json()) as {
      imageFilename: string;
      groupId: string;
      action: "add" | "remove";
    };
    const hash = safeHashMaps(targetDir).filenameToHash.get(body.imageFilename);
    if (!hash) {
      return json(
        { error: `No content hash for ${body.imageFilename} — run extraction first` },
        400,
      );
    }
    const { changed, next } = await mutateConstraints(targetDir, (c) => {
      const has = c.imageGroupCannotLink.some(
        (e) => e.imageHash === hash && e.groupId === body.groupId,
      );
      if (body.action === "add" && !has) {
        return {
          ...c,
          imageGroupCannotLink: [
            ...c.imageGroupCannotLink,
            { imageHash: hash, groupId: body.groupId },
          ],
        };
      }
      if (body.action === "remove" && has) {
        return {
          ...c,
          imageGroupCannotLink: c.imageGroupCannotLink.filter(
            (e) => !(e.imageHash === hash && e.groupId === body.groupId),
          ),
        };
      }
      return c;
    });
    return json(constraintsResponse(targetDir, next, changed));
  }

  if (path === "/api/constraints/group-lock" && req.method === "POST") {
    const body = (await req.json()) as { groupId: string; locked: boolean };
    const { changed, next } = await mutateConstraints(targetDir, (c) => {
      const has = c.lockedGroupIds.includes(body.groupId);
      if (body.locked && !has) {
        return { ...c, lockedGroupIds: [...c.lockedGroupIds, body.groupId] };
      }
      if (!body.locked && has) {
        return { ...c, lockedGroupIds: c.lockedGroupIds.filter((id) => id !== body.groupId) };
      }
      return c;
    });
    return json(constraintsResponse(targetDir, next, changed));
  }

  if (path === "/api/constraints/clear" && req.method === "POST") {
    const body = (await req.json().catch(() => ({}))) as { groupId?: string };
    const { changed, next } = await mutateConstraints(targetDir, (c) => {
      if (body.groupId) {
        return {
          version: 1,
          imageGroupCannotLink: c.imageGroupCannotLink.filter((e) => e.groupId !== body.groupId),
          lockedGroupIds: c.lockedGroupIds.filter((id) => id !== body.groupId),
        };
      }
      return { version: 1, imageGroupCannotLink: [], lockedGroupIds: [] };
    });
    return json(constraintsResponse(targetDir, next, changed));
  }

  return null;
};
