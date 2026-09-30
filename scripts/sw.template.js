// Service worker (generated into dist/sw.js at build time, see vite.config.ts).
// - App shell: precached per build; old builds' caches are removed on activate.
// - Pages: network first, so a deploy shows up on the next load; cache when offline.
// - /ort/ (ONNX Runtime, ~28 MB): served from its own cache once kept there, else from
//   the network untouched. Keeping it while it is in use held a second copy of the
//   28 MB response in memory, on phones at the moment the analysis starts; so the files
//   the page used are stored later, when it says it is idle ("keep-runtime"), from the
//   HTTP cache, streamed to disk — and the app then works offline.
// - /models/: not touched here — src/neural/ort.ts keeps them in its own Cache.
// Everything served is same-origin and keeps its original headers, so the page
// stays cross-origin isolated (COOP/COEP) when it comes from the cache.
const VERSION = __VERSION__;
const PRECACHE = __PRECACHE__;
const SHELL = `shell-${VERSION}`;
const RUNTIME = `runtime-${__RUNTIME__}`;
/** The ONNX Runtime files requested since this worker started (what to keep). */
const seen = new Set();

self.addEventListener("install", (e) => {
  e.waitUntil((async () => {
    const c = await caches.open(SHELL);
    // One by one: a single failed request must not lose the whole install.
    await Promise.all(PRECACHE.map((u) => c.add(new Request(u, { cache: "reload" })).catch(() => {})));
    // No skipWaiting: pages already open keep the version they started with (their lazily
    // loaded chunks have that build's names); the new one takes over on the next start.
  })());
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) {
      if (k.startsWith("ort-") || (k.startsWith("runtime-") && k !== RUNTIME)) await caches.delete(k);
    }
    // Keep the last three app shells: a page opened before a deploy can still load its chunks.
    const shells = (await caches.keys()).filter((k) => k.startsWith("shell-") && k !== SHELL);
    for (const k of shells.slice(0, Math.max(0, shells.length - 2))) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  const p = url.pathname;
  if (p.startsWith("/ort/")) {
    seen.add(p);
    e.respondWith((async () => (await caches.match(req, { cacheName: RUNTIME, ignoreSearch: true, ignoreVary: true })) ?? fetch(req))());
    return;
  }
  if (p.startsWith("/models/") || p.startsWith("/__")) return;

  if (req.mode === "navigate") {
    e.respondWith((async () => {
      try {
        const res = await fetch(req);
        if (res.ok) (await caches.open(SHELL)).put("/", res.clone()).catch(() => {});
        return res;
      } catch {
        return (await caches.match("/", { cacheName: SHELL, ignoreVary: true })) ?? Response.error();
      }
    })());
    return;
  }

  if (p.startsWith("/assets/") || PRECACHE.includes(p)) {
    e.respondWith(cacheFirst(SHELL, req));
  }
});

self.addEventListener("message", (e) => {
  if (e.data?.type !== "keep-runtime") return;
  e.waitUntil((async () => {
    const c = await caches.open(RUNTIME);
    // Also what the page names (a runtime a feature will need, not yet used here).
    for (const p of new Set([...seen, ...(e.data.also ?? []).filter((x) => typeof x === "string" && x.startsWith("/ort/"))])) {
      if (!(await c.match(p))) await c.add(p).catch(() => {});
    }
  })());
});

async function cacheFirst(name, req) {
  const c = await caches.open(name);
  // Any kept shell has it: hashed file names never change content. (Not by Vary: a module
  // script's request carries an Origin header that the precache's request did not.)
  const hit = await caches.match(req, { ignoreSearch: true, ignoreVary: true });
  if (hit) return hit;
  const res = await fetch(req);
  // Only whole, successful same-origin answers; never an HTML fallback for a binary.
  if (res.ok && res.status === 200 && !(res.headers.get("content-type") ?? "").includes("text/html")) {
    c.put(req, res.clone()).catch(() => {});
  }
  return res;
}
