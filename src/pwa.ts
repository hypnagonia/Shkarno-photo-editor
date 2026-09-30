// Installable app: the service worker (dist/sw.js, generated at build time)
// exists only in production builds.
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(() => { /* private mode, unsupported */ });
  });
}

/**
 * The photo's analysis is over and nothing runs: the service worker may now keep the
 * ONNX Runtime files it served (for offline use), without competing for memory.
 */
export function keepRuntimeWhenIdle() {
  if (!import.meta.env.PROD || !("serviceWorker" in navigator)) return;
  const send = () => navigator.serviceWorker.controller?.postMessage({ type: "keep-runtime" });
  const ric = (globalThis as { requestIdleCallback?: (f: () => void, o?: { timeout: number }) => void }).requestIdleCallback;
  setTimeout(() => (ric ? ric(send, { timeout: 10_000 }) : send()), 5000);
}
