import { access, constants, copyFile, mkdir, readdir, stat } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { listImages } from "./src/rename.ts";
import { createServer } from "./src/server.ts";
import { clearCache, preGenerateThumbnails } from "./src/thumbnails.ts";

const DEFAULT_PORT = 4928;

async function main() {
  const targetDir = process.argv[2];

  if (!targetDir) {
    console.error("Usage: bun run start.ts <directory>");
    console.error("  Reorder images in a directory via drag-and-drop");
    process.exit(1);
  }

  const absDir = resolve(targetDir);

  // Validate directory
  try {
    const dirStat = await stat(absDir);
    if (!dirStat.isDirectory()) {
      console.error(`Error: ${absDir} is not a directory`);
      process.exit(1);
    }
  } catch {
    console.error(`Error: directory does not exist: ${absDir}`);
    process.exit(1);
  }

  try {
    await access(absDir, constants.R_OK | constants.W_OK);
  } catch {
    console.error(`Error: no read/write permission for: ${absDir}`);
    process.exit(1);
  }

  // Build client
  const distDir = join(import.meta.dir, "dist");
  await mkdir(distDir, { recursive: true });

  console.log("Building client...");
  const result = await Bun.build({
    entrypoints: [join(import.meta.dir, "src/client/index.tsx")],
    outdir: distDir,
    naming: "[name].[ext]",
    minify: true,
  });

  if (!result.success) {
    console.error("Build failed:");
    for (const log of result.logs) {
      console.error(log);
    }
    process.exit(1);
  }

  // Copy static files in parallel
  const clientDir = join(import.meta.dir, "src/client");
  const stylesDir = join(clientDir, "styles");
  const cssFiles = (await readdir(stylesDir)).filter((f) => f.endsWith(".css"));
  await Promise.all([
    copyFile(join(clientDir, "index.html"), join(distDir, "index.html")),
    ...cssFiles.map((f) => copyFile(join(stylesDir, f), join(distDir, f))),
  ]);

  // Find available port and start server
  let server;
  let port = DEFAULT_PORT;
  for (; port < DEFAULT_PORT + 100; port++) {
    try {
      server = createServer(absDir, distDir, port);
      break;
    } catch {}
  }
  if (!server) {
    console.error("No available port found");
    process.exit(1);
  }

  const url = `http://localhost:${port}`;
  console.log(`\nReorder → ${basename(absDir)}`);
  console.log(`Server running at ${url}`);
  console.log(`Target directory: ${absDir}\n`);

  // Open browser (macOS)
  Bun.spawn(["open", url], { stdout: "ignore", stderr: "ignore" });

  // Background thumbnail pre-generation (fire-and-forget)
  let imageCount = 0;
  listImages(absDir)
    .then((filenames) => {
      imageCount = filenames.length;
      if (filenames.length > 0) {
        preGenerateThumbnails(absDir, filenames).catch(() => {});
      }
    })
    .catch(() => {});

  // Print alias suggestion
  const scriptPath = join(import.meta.dir, "start.ts");
  console.log("Tip: add an alias for quick access:");
  console.log(`  alias reorder="bun run ${scriptPath}"`);
  console.log(`  # Usage: reorder /path/to/image/directory\n`);

  // Graceful shutdown — clear cache for small directories (fast to regenerate)
  const shutdown = () => {
    console.log("\nShutting down...");
    server.stop(true);
    if (imageCount > 0 && imageCount <= 250) {
      clearCache(absDir).finally(() => process.exit(0));
    } else {
      process.exit(0);
    }
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main();
