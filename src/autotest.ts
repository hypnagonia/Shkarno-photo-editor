/**
 * Autotest (local only: ?autotest on localhost / a LAN address; never on the site).
 * Opens a sample photo by itself and runs the heavy steps a person would, one after
 * the other, reporting each to the local server (/__debug/report → .samples/out/
 * autotest.jsonl). scripts/memcheck.mjs runs it in the iPhone simulator's Safari and
 * measures the tab's memory per step: the guard that keeps updates from breaking
 * phones.
 *
 *   ?autotest&photo=IMG_1514.DNG&steps=open,select,blur,export,reopen
 *   ?autotest&set=burst/b3&steps=burst,ab,export   (a series: merged, then A/B)
 */
import type { FromWorker, ToWorker } from "./engine/protocol.ts";
import type { Params } from "./decision/params.ts";
import { makeLayer } from "./layers/model.ts";

export interface AutotestApp {
  openFile: (f: File) => void;
  /** Opens several shots as one merged series. */
  openSeries: (files: File[]) => void;
  params: () => Params | undefined;
  pushParams: () => void;
  send: (m: ToWorker) => void;
  /** Every message from the engine; returns an unsubscribe. */
  on: (fn: (m: FromWorker) => void) => () => void;
}

export const autotestAllowed = () =>
  new URLSearchParams(location.search).has("autotest") && /^(localhost|127\.|10\.|192\.168\.|\[::1\])/.test(location.hostname);

export async function runAutotest(app: AutotestApp) {
  const q = new URLSearchParams(location.search);
  const photo = q.get("photo") ?? "IMG_1514.DNG";
  const steps = (q.get("steps") ?? "open,select,blur,export,reopen").split(",").filter(Boolean);
  const t0 = performance.now();
  let lastProfile: unknown;
  const report = (stage: string, extra: Record<string, unknown> = {}) =>
    fetch("/__debug/report", { method: "POST", body: JSON.stringify({ t: Math.round(performance.now() - t0), stage, photo, ...extra }) }).catch(() => undefined);
  app.on((m) => {
    if (m.type === "profile") lastProfile = m.stages;
    if (m.type === "progress") void report(`progress:${m.stage}`, { detail: m.detail });
    if (m.type === "error") void report("error", { message: m.message });
  });
  const waitFor = (pred: (m: FromWorker) => boolean, what: string, ms = 240_000) => new Promise<FromWorker>((resolve, reject) => {
    const timer = setTimeout(() => { off(); reject(new Error(`timeout waiting for ${what}`)); }, ms);
    const off = app.on((m) => {
      if (m.type === "error") { clearTimeout(timer); off(); reject(new Error(m.message)); }
      else if (pred(m)) { clearTimeout(timer); off(); resolve(m); }
    });
  });
  const finalPreview = (what: string) => waitFor((m) => m.type === "preview" && !!m.final, what);
  const settle = (ms = 1500) => new Promise((r) => setTimeout(r, ms));
  /** Chrome's own account of JS + WebAssembly memory, by worker (where the browser has it). */
  const jsMem = async (): Promise<Record<string, number>> => {
    const pm = performance as Performance & { measureUserAgentSpecificMemory?: () => Promise<{ breakdown: Array<{ bytes: number; types: string[]; attribution: Array<{ url?: string; scope?: string }> }> }> };
    if (!pm.measureUserAgentSpecificMemory) return {};
    try {
      const r = await pm.measureUserAgentSpecificMemory();
      const out: Record<string, number> = {};
      for (const b of r.breakdown) {
        const who = b.attribution.map((a) => `${a.scope ?? ""}:${(a.url ?? "").split("/").pop()?.split("?")[0] ?? ""}`).join(",") || "shared";
        const k = `${who} [${b.types.join("+") || "-"}]`;
        out[k] = (out[k] ?? 0) + Math.round(b.bytes / 1048576);
      }
      return out;
    } catch { return {}; }
  };
  /** The engine's GPU memory: now, and the peak since the last ask (asking resets it). */
  const mem = async () => { const r = waitFor((m) => m.type === "mem", "mem", 60_000); app.send({ type: "mem" }); const m = await r; return m.type === "mem" ? { gpuLiveMB: m.liveMB, gpuPeakMB: m.peakMB } : {}; };
  const open = async (label: string) => {
    await mem();
    await report(`${label}:start`);
    const res = await fetch(`/__samples/${encodeURIComponent(photo)}`);
    if (!res.ok) throw new Error(`sample ${photo}: ${res.status}`);
    const f = new File([await res.blob()], photo);
    const done = finalPreview(label);
    app.openFile(f);
    await done;
    await settle(3000); // restoration / quality stages after the first final preview
    await report(`${label}:done`, { profile: lastProfile, ...(await mem()), js: await jsMem() });
  };
  const addLayer = async (label: string, layer: ReturnType<typeof makeLayer>) => {
    const p = app.params();
    if (!p) throw new Error("no photo");
    await mem();
    await report(`${label}:start`);
    const done = finalPreview(label);
    p.layers = [...(p.layers ?? []), layer];
    app.pushParams();
    await done;
    await settle();
    await report(`${label}:done`, { ...(await mem()), js: await jsMem() });
  };
  /** A series from .samples/<set>/ (every file in it), merged. */
  const burst = async () => {
    const set = q.get("set") ?? "burst/b3";
    await mem();
    await report("burst:start", { set });
    const names = await (await fetch(`/__samples/${set}/`)).json() as string[];
    const files = await Promise.all(names.map(async (n) => new File([await (await fetch(`/__samples/${set}/${n}`)).blob()], n)));
    const done = finalPreview("burst");
    app.openSeries(files);
    await done;
    await settle(3000);
    await report("burst:done", { profile: lastProfile, frames: files.length, ...(await mem()), js: await jsMem() });
  };
  /** A/B of a series: the single shot, then back to the merge. */
  const ab = async () => {
    await mem();
    await report("ab:start");
    for (const single of [true, false]) {
      const done = finalPreview("ab");
      app.send({ type: "seriesView", single });
      await done;
      await settle(500);
      if (q.has("save")) await saveExport(`ab-${single ? "single" : "merged"}`);
    }
    await report("ab:done", { ...(await mem()), js: await jsMem() });
  };
  /** Exports and keeps the file in .samples/out (for looking at results). */
  const saveExport = async (name: string) => {
    const done = waitFor((m) => m.type === "exported", "export");
    app.send({ type: "export", format: "jpeg", quality: 0.95, space: "p3" });
    const m = await done;
    if (m.type === "exported") await fetch(`/__debug/save?name=${name}.jpg`, { method: "POST", body: m.blob });
  };
  const center = { kind: "select" as const, points: [[0.5, 0.5, 1]] as Array<[number, number, 0 | 1]>, invert: false, feather: 1, density: 1 };
  try {
    await report("start", { ua: navigator.userAgent, gpu: "gpu" in navigator, isolated: crossOriginIsolated });
    for (const s of steps) {
      if (s === "open") await open("open");
      else if (s === "burst") await burst();
      else if (s === "ab") await ab();
      else if (s === "reopen") await open("reopen");
      else if (s === "select") await addLayer("select", makeLayer("basic", "Autotest select", { mask: center, params: { exposure: 0.4, temp: 0, tint: 0, saturation: 0, vibrance: 0, hue: 0 } }));
      else if (s === "blur") await addLayer("blur", makeLayer("blur", "Autotest blur", { mask: { ...center, invert: true }, params: { amount: 0.5 } }));
      else if (s === "export") {
        await mem();
        await report("export:start");
        const done = waitFor((m) => m.type === "exported", "export");
        app.send({ type: "export", format: "jpeg", quality: 0.92, space: "p3" });
        await done;
        await report("export:done", { ...(await mem()), js: await jsMem() });
      }
    }
    await report("done");
    // ?close: leave the page (the memcheck script's tab gives its memory back).
    if (q.has("close")) setTimeout(() => { location.href = "about:blank"; }, 500);
  } catch (e) {
    await report("failed", { message: e instanceof Error ? e.message : String(e) });
    if (q.has("close")) setTimeout(() => { location.href = "about:blank"; }, 500);
  }
}
