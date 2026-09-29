/**
 * Autotest (local only: ?autotest on localhost / a LAN address; never on the site).
 * Opens a sample photo by itself and runs the heavy steps a person would, one after
 * the other, reporting each to the local server (/__debug/report → .samples/out/
 * autotest.jsonl). scripts/memcheck.mjs runs it in the iPhone simulator's Safari and
 * measures the tab's memory per step: the guard that keeps updates from breaking
 * phones.
 *
 *   ?autotest&photo=IMG_1514.DNG&steps=open,select,blur,export,reopen
 */
import type { FromWorker, ToWorker } from "./engine/protocol.ts";
import type { Params } from "./decision/params.ts";
import { makeLayer } from "./layers/model.ts";

export interface AutotestApp {
  openFile: (f: File) => void;
  params: () => Params | undefined;
  pushParams: () => void;
  /** Develop's "Zero all" (every control neutral). */
  zero: () => void;
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
    // A GPU validation error drops that dispatch silently (the result is just wrong): fail the run.
    if (m.type === "log" && /^GPU error/.test(m.text)) void report("gpuerror", { message: m.text });
    if (m.type === "analysis" && q.has("logs")) for (const d of m.decisions) void report("log", { text: `decision ${d.id}: ${JSON.stringify(d.value)} — ${d.reason}` });
    if (m.type === "log" && (q.has("logs") || /^(Apple mattes|auto focus)/.test(m.text))) void report("log", { text: m.text });
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
    const f = new File([await res.blob()], photo.split("/").pop()!);
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
      else if (s === "snap") await saveExport(q.get("tag") ?? "snap");
      else if (s === "check") {
        // The Check tab's findings on the automatic result (the quality benchmark reads them).
        const r = waitFor((m) => m.type === "check", "check", 120_000);
        app.send({ type: "check" });
        const m = await r;
        if (m.type === "check") await report("check", { items: m.items.map((i) => ({ id: i.id, level: i.level, err: i.err, v: i.v })) });
      }
      else if (s === "dof") {
        // Depth of field on (automatic focus and strength, or 0.6 when auto found none).
        const p = app.params(); if (!p) throw new Error("no photo");
        const done = finalPreview("dof");
        p.enable = { ...p.enable, dof: true };
        if (!(p.dof.strength > 0.3)) p.dof = { ...p.dof, strength: 0.6 };
        app.pushParams(); await done; await settle();
        await report("dof:info", { focus: p.dof.focus, span: p.dof.focusSpan, strength: p.dof.strength, mode: p.dof.mode });
        if (q.has("save")) await saveExport("dof");
      }
      else if (s === "bright") {
        // +2 EV: bright colours into the shoulder (where renderings differ most).
        const p = app.params(); if (!p) throw new Error("no photo");
        const done = finalPreview("bright"); p.exposure += 2; app.pushParams(); await done; await settle();
      }
      else if (s === "classic") {
        // The classic display rendering, for comparison with img.
        const p = app.params(); if (!p) throw new Error("no photo");
        const done = finalPreview("classic");
        p.render = { ...(p.render ?? { purity: 0, strength: 1 }), engine: "classic" };
        app.pushParams(); await done; await settle();
        if (q.has("save")) await saveExport("classic");
      }
      else if (s.startsWith("off-")) {
        // One stage off (off-semantic, off-curves, off-localTone, off-dehaze, off-color…), exported as off-<stage>.
        // Stays off for the following steps, like the others: off-dehaze,off-semantic is both.
        const p = app.params(); if (!p) throw new Error("no photo");
        const k = s.slice(4);
        const done = finalPreview(s);
        const e = p.enable as Record<string, boolean>;
        if (k in e) e[k] = false;
        else if (k === "color") p.color = { saturation: 0, vibrance: 0 };
        else if (k === "layers") p.layers = [];
        app.pushParams(); await done; await settle();
        if (q.has("save")) await saveExport(s);
      }
      else if (s === "zero") { const done = finalPreview("zero"); app.zero(); await done; await settle(); if (q.has("save")) await saveExport("zero"); }
      else if (s === "ceq") {
        // Contrast equalizer: a preset (?ceq=id, default clarity), then an export (strips: seams would show).
        const p = app.params();
        if (!p) throw new Error("no photo");
        const { CONTRAST_EQ_PRESETS, neutralContrastEq } = await import("./tone/contrastEq.ts");
        const pr = CONTRAST_EQ_PRESETS.find((x) => x.id === q.get("ceq")) ?? CONTRAST_EQ_PRESETS[0];
        await mem();
        await report("ceq:start");
        const done = finalPreview("ceq");
        p.contrastEq = { ...neutralContrastEq(), luma: [...pr.luma], chroma: [...pr.chroma] };
        app.pushParams();
        await done;
        await settle();
        if (q.has("save")) await saveExport("ceq");
        await report("ceq:done", { ...(await mem()), js: await jsMem() });
      } // (the photo as it is now, for comparisons)
      else if (s === "toneeq") {
        // Tone equalizer: the "compress" preset on a fitted mask, then its mask view (a read-back of zones).
        const p = app.params();
        if (!p) throw new Error("no photo");
        const { autoFitMask, histQuantile, neutralToneEq, TONE_EQ_PRESETS } = await import("./tone/toneEq.ts");
        await mem();
        await report("toneeq:start");
        const hr = waitFor((m) => m.type === "toneEqHist", "toneEqHist");
        app.send({ type: "toneEqHist" });
        const hm = await hr;
        const h = hm.type === "toneEqHist" ? hm.hist?.balanced : undefined;
        if (!h) throw new Error("no tone EQ histogram");
        const fit = autoFitMask(h);
        await report("toneeq:mask", { p5: histQuantile(h, 0.05), p50: histQuantile(h, 0.5), p95: histQuantile(h, 0.95), ...fit });
        const done = finalPreview("toneeq");
        p.toneEq = { ...neutralToneEq(), gains: [...TONE_EQ_PRESETS[q.get("teq") === "strong" ? 2 : 1].gains], ...fit };
        app.pushParams();
        await done;
        await settle();
        const z = waitFor((m) => m.type === "toneEqZone", "toneEqZone");
        app.send({ type: "toneEqZone", x: 0.5, y: 0.2 });
        const zr = await z;
        if (q.has("save")) await saveExport("toneeq");
        await report("toneeq:done", { zone: zr.type === "toneEqZone" ? zr.zone : undefined, ...(await mem()), js: await jsMem() });
      }
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
