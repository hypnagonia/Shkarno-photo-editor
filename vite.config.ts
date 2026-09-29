import { defineConfig, type Plugin } from "vite";
import { appendFileSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Local testing only (dev server and `vite preview`, never in a deployed build):
 *   POST /__debug/save?name=…   writes the body to .samples/out (export validation)
 *   GET  /__samples/<file>      a sample photo from .samples (autotest, scripts/memcheck.mjs)
 *   POST /__debug/report        appends one JSON line to .samples/out/autotest.jsonl
 */
type Mw = (req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse, next: () => void) => void;
const localTesting: Array<[string, Mw]> = [
  ["/__debug/save", (req, res) => {
    const name = new URL(req.url ?? "", "http://x").searchParams.get("name")?.replace(/[^\w.\-]/g, "_") ?? "out.bin";
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      mkdirSync(".samples/out", { recursive: true });
      writeFileSync(`.samples/out/${name}`, Buffer.concat(chunks));
      res.end("ok");
    });
  }],
  ["/__samples", (req, res, next) => {
    const name = decodeURIComponent((req.url ?? "").split("?")[0].replace(/^\//, ""));
    // A file in .samples, or in one folder of it (bench/…).
    if (!/^([\w.\-]+\/)?[\w.\- ]+$/.test(name) || name.includes("..")) return next();
    try {
      const data = readFileSync(join(".samples", name));
      res.setHeader("Content-Type", "application/octet-stream");
      res.setHeader("Cross-Origin-Resource-Policy", "same-origin");
      res.end(data);
    } catch { res.statusCode = 404; res.end(); }
  }],
  ["/__debug/report", (req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      mkdirSync(".samples/out", { recursive: true });
      appendFileSync(".samples/out/autotest.jsonl", Buffer.concat(chunks).toString("utf8").replace(/\n/g, " ") + "\n");
      res.end("ok");
    });
  }],
];
const debugSave: Plugin = {
  name: "local-testing",
  configureServer(server) { for (const [path, mw] of localTesting) server.middlewares.use(path, mw); },
  configurePreviewServer(server) { for (const [path, mw] of localTesting) server.middlewares.use(path, mw); },
};

/**
 * Build-only: writes dist/sw.js from scripts/sw.template.js with this build's
 * app shell (everything in dist except /models and /ort, which are cached at
 * run time) and a version derived from the shell's contents.
 */
const serviceWorker: Plugin = {
  name: "service-worker",
  apply: "build",
  closeBundle() {
    const dist = fileURLToPath(new URL("./dist", import.meta.url));
    const files: string[] = [];
    const walk = (d: string) => {
      for (const f of readdirSync(d)) {
        const abs = join(d, f);
        const rel = "/" + relative(dist, abs).split("\\").join("/");
        if (rel === "/models" || rel === "/ort" || rel === "/sw.js" || rel.endsWith(".txt")) continue;
        if (statSync(abs).isDirectory()) walk(abs); else files.push(rel);
      }
    };
    walk(dist);
    files.sort();
    const hash = createHash("sha256");
    for (const f of files) hash.update(f).update(readFileSync(join(dist, f)));
    const precache = ["/", ...files.filter((f) => f !== "/index.html")];
    const sw = readFileSync(new URL("./scripts/sw.template.js", import.meta.url), "utf8")
      .replace("__VERSION__", JSON.stringify(hash.digest("hex").slice(0, 12)))
      .replace("__PRECACHE__", JSON.stringify(precache));
    writeFileSync(join(dist, "sw.js"), sw);
  },
};

// Cross-origin isolation enables SharedArrayBuffer, i.e. multi-threaded WASM
// for ONNX Runtime's CPU fallback. The same headers are set on Vercel.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig(({ command }) => ({
  plugins: [debugSave, serviceWorker],
  // Use ORT's non-bundled build: it loads its runtime from env.wasm.wasmPaths
  // (public/ort, see scripts/copy-ort.mjs) instead of inlining it.
  // Build only: the dev server serves ORT unbundled and refuses imports from /public.
  resolve: command === "build"
    ? { alias: [
      { find: /^onnxruntime-web$/, replacement: fileURLToPath(new URL("./node_modules/onnxruntime-web/dist/ort.min.mjs", import.meta.url)) },
      { find: /^onnxruntime-web\/wasm$/, replacement: fileURLToPath(new URL("./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs", import.meta.url)) },
    ] }
    : {},
  define: { __ORT_EXTERNAL__: JSON.stringify(command === "build"), __BUILD__: JSON.stringify(Date.now().toString(36)) },
  build: { target: "es2022", assetsInlineLimit: 0, chunkSizeWarningLimit: 4000 },
  worker: { format: "es" },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  server: { headers: isolation },
  preview: { headers: isolation },
}));
