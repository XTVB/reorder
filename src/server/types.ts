// Shared types for server route handlers.

export type RouteContext = {
  path: string;
  targetDir: string;
};

export type RouteHandler = (req: Request, ctx: RouteContext) => Promise<Response | null>;
