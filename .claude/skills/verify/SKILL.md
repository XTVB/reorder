# Verify: run the app headlessly and drive it with Playwright

`bun run start.ts <dir>` builds the client but also opens the user's browser
(`open <url>`). For verification, skip it and use the pieces directly:

1. **Fixture**: make a scratch dir of small JPEGs (sharp is a dependency —
   solid-color `sharp({create:...}).jpeg()` images work) plus a
   `.reorder-groups.json` (array of `{id, name, images, tags?}`).
2. **Server without browser-open**: build the client + copy static files the
   way start.ts does, then call `createServer(targetDir, distDir, port)` from
   `src/server/index.ts` in a small script; run it in the background.

   ```ts
   const result = await Bun.build({ entrypoints: [join(repo, "src/client/index.tsx")], outdir: distDir, naming: "[name].[ext]", minify: true });
   // copy src/client/index.html and src/client/styles/*.css into distDir
   createServer(targetDir, distDir, port);
   ```

   Ready when `GET /api/groups` returns 200.
3. **Drive**: Playwright is in devDependencies but bun may resolve a newer
   playwright from its global cache whose browsers aren't installed — import
   from the repo explicitly:

   ```ts
   import { chromium } from "/path/to/reorder/node_modules/playwright/index.mjs";
   ```

   Wire `page.on("pageerror")` / console-error logging; screenshot per step.

Gotchas:
- SPA routes: go straight to `http://localhost:<port>/reorder` (server serves
  index.html for non-API paths).
- Group cards render as `.group-card`; modals as `.modal` / `.review-modal`.
- State on disk: groups persist to `<dir>/.reorder-groups.json` on every
  update (client auto-POSTs), caches under `<dir>/.reorder-cache/`. Reset by
  recreating the fixture dir; restart the server after client-code changes
  (it builds dist only at launch).
- Check flows through the API too: `GET /api/groups` shows applied
  order/tags without scraping the DOM.
