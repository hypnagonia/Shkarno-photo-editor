// Installable app: the service worker (dist/sw.js, generated at build time)
// exists only in production builds.
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(() => { /* private mode, unsupported */ });
  });
}

import { MODEL_CACHE, SELECT_MODELS, SELECT_RUNTIME } from "./neural/modelCache.ts";

let prefetched = false;
/**
 * The photo's analysis is over and nothing runs: the service worker may now keep the
 * ONNX Runtime files it served, and Select's models are fetched ahead of their first use
 * — so everything works offline — without competing for memory (Cache Storage writes
 * the downloads straight to disk). Not on a data-saving connection.
 */
export function keepRuntimeWhenIdle() {
  if (!import.meta.env.PROD || !("serviceWorker" in navigator)) return;
  const saveData = (navigator as { connection?: { saveData?: boolean } }).connection?.saveData === true;
  const run = async () => {
    navigator.serviceWorker.controller?.postMessage({ type: "keep-runtime", also: saveData ? [] : SELECT_RUNTIME });
    if (saveData || prefetched) return;
    prefetched = true;
    try {
      const c = await caches.open(MODEL_CACHE);
      for (const u of SELECT_MODELS) if (!(await c.match(u))) await c.add(u);
    } catch { prefetched = false; /* offline, quota: next time */ }
  };
  const ric = (globalThis as { requestIdleCallback?: (f: () => void, o?: { timeout: number }) => void }).requestIdleCallback;
  setTimeout(() => (ric ? ric(() => void run(), { timeout: 10_000 }) : void run()), 5000);
}
