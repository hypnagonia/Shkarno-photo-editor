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
  /** Chrome's own account of JS + WebAssembly memory, by worker (where the browser has it; ?jsmem). */
  const jsMem = async (): Promise<Record<string, number>> => {
    const pm = performance as Performance & { measureUserAgentSpecificMemory?: () => Promise<{ breakdown: Array<{ bytes: number; types: string[]; attribution: Array<{ url?: string; scope?: string }> }> }> };
    // ?gc (memcheck): a full collection now, in the page and the engine's worker — Chrome
    // started with --js-flags=--expose-gc; memcheck's page limits assume it. ?jsmem: also
    // the breakdown by worker, which waits for the browser's own full GC (up to ~20 s a call).
    // (Then a moment for the browser to hand back what was freed — WebAssembly and GPU
    // buffers go back lazily; without it the next step starts on the last one's leftovers.)
    if (q.has("gc")) { app.send({ type: "gc" }); (globalThis as { gc?: () => void }).gc?.(); await new Promise((r) => setTimeout(r, Number(q.get("gcWait") ?? 3000))); }
    if (!q.has("jsmem") || !pm.measureUserAgentSpecificMemory) return {};
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
      else if (s === "retouch") {
        // Magic brush: ?stroke=x,y;x,y (0…1) &r=radius (share of the long side), exported as retouch.
        const p = app.params(); if (!p) throw new Error("no photo");
        const pts = (q.get("stroke") ?? "0.5,0.5").split(";").map((v) => v.split(",").map(Number) as [number, number]);
        await mem();
        await report("retouch:start");
        const done = finalPreview("retouch");
        p.retouch = [...(p.retouch ?? []), { pts, r: Number(q.get("r") ?? 0.02) }];
        app.pushParams(); await done; await settle();
        await report("retouch:done", { ...(await mem()), js: await jsMem() });
        if (q.has("save")) await saveExport("retouch");
      }
      else if (s === "unretouch") {
        // Undo the last magic brush stroke (its pixels put back), exported as unretouch.
        const p = app.params(); if (!p) throw new Error("no photo");
        const done = finalPreview("unretouch");
        p.retouch = (p.retouch ?? []).slice(0, -1);
        app.pushParams(); await done; await settle();
        if (q.has("save")) await saveExport("unretouch");
      }
      else if (s === "fog" || s === "light") {
        // A Fog / Light layer over the whole photo at its defaults, exported as fog / light (then removed).
        await addLayer(s, makeLayer(s, `Autotest ${s}`));
        if (q.has("save")) await saveExport(s);
        const p = app.params(); if (p) { p.layers = p.layers.filter((l) => l.name !== `Autotest ${s}`); app.pushParams(); await settle(); }
      }
      else if (s === "rays") {
        // A Light Rays layer from the brightest spot (?rays=x,y instead; ?rlen, ?rthr, ?ramount), exported as rays.
        const at = (q.get("rays") ?? "").split(",").map(Number);
        const b = at.length === 2 && at.every(Number.isFinite) ? { x: at[0], y: at[1] } : await (async () => {
          const r = waitFor((m) => m.type === "brightest", "brightest"); app.send({ type: "brightest" }); const m = await r;
          return m.type === "brightest" ? { x: m.x, y: m.y } : { x: 0.5, y: 0.2 };
        })();
        report("rays:source", b);
        await addLayer("rays", makeLayer("rays", "Autotest rays", { params: { amount: Number(q.get("ramount") ?? 1), length: Number(q.get("rlen") ?? 0.5), threshold: Number(q.get("rthr") ?? 0.6), ...b } }));
        if (q.has("save")) await saveExport("rays");
      }
      else if (s.startsWith("film-")) {
        // A film character at full strength on 35 mm (film-clean, film-negative, film-cinema), exported as film-<character>.
        const p = app.params(); if (!p) throw new Error("no photo");
        const done = finalPreview(s);
        p.film = { character: s.slice(5) as NonNullable<typeof p.film>["character"], strength: 1, format: 36 };
        app.pushParams(); await done; await settle();
        if (q.has("save")) await saveExport(s);
      }
      else if (s === "frame") {
        // ?frame=quarter;flip;angle;x;y;w;h (a crop well inside: it is not fitted here); exported as frame.
        const p = app.params(); if (!p) throw new Error("no photo");
        const [qt, fl, an, ...c] = (q.get("frame") ?? "1;0;5;0.1;0.1;0.8;0.8").split(";").map(Number);
        const f = { quarter: qt & 3, flip: !!fl, angle: an, crop: c as [number, number, number, number] };
        p.frame = f;
        const done = finalPreview("frame"); app.pushParams(); await done; await settle();
        await report("frame:done", { frame: f });
        if (q.has("save")) await saveExport("frame");
      }
      else if (s.startsWith("set:")) {
        // One setting by path (set:semantic.person.exposure=0), exported as set-<path>.
        const p = app.params(); if (!p) throw new Error("no photo");
        const [path, v] = s.slice(4).split("=");
        const ks = path.split(".");
        const o = ks.slice(0, -1).reduce((a: Record<string, unknown>, k) => a[k] as Record<string, unknown>, p as unknown as Record<string, unknown>);
        const done = finalPreview(s);
        o[ks[ks.length - 1]] = Number(v);
        app.pushParams(); await done; await settle();
        if (q.has("save")) await saveExport(`set-${path}`);
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
      else if (s === "blur") {
        // A lens Blur layer around the centre object (?ball: over the whole photo); ?bamount, ?bokeh, ?blades; exported as blur.
        const all = { kind: "all" as const, invert: false, feather: 1, density: 1 };
        await addLayer("blur", makeLayer("blur", "Autotest blur", { mask: q.has("ball") ? all : { ...center, invert: true }, params: { amount: Number(q.get("bamount") ?? 0.5), bokeh: Number(q.get("bokeh") ?? 0), blades: Number(q.get("blades") ?? 0) } }));
        if (q.has("save") && q.has("bokeh")) await saveExport("blur");
      }
      else if (s === "motion") {
        // Motion blur of everything but the picked subject (?angle=, default horizontal; ?parallax: into the depth; ?through: through the mask), exported as motion.
        // ?pick=x,y: the object tapped there (default the centre); ?keep: blur it, not everything else.
        // (Several taps: ?pick=x,y;x,y — one object built from them.)
        const pts = (q.get("pick") ?? "").split(";").map((p) => p.split(",").map(Number)).filter((p) => p.length === 2 && p.every(Number.isFinite));
        const pk = pts[0] ?? [];
        // ?region=vehicle: a region of the segmentation instead of a tapped object.
        const rg = q.get("region");
        const at = rg ? { kind: "region" as const, region: rg as "vehicle", invert: false, feather: 1, density: 1 } : pk.length === 2 && pk.every(Number.isFinite) ? { ...center, points: pts.map((p) => [p[0], p[1], 1]) as Array<[number, number, 0 | 1]> } : center;
        await addLayer("motion", makeLayer("blur", "Autotest motion", { mask: { ...at, invert: !q.has("keep") }, params: { amount: Number(q.get("amount") ?? 0.6), motion: true, angle: Number(q.get("angle") ?? 0), depth: q.has("parallax"), through: q.has("through"), object: q.has("object"), trail: Number(q.get("trail") ?? 0.6), sharp: Number(q.get("sharp") ?? 0), arriving: q.has("arriving") } }));
        if (q.has("parallax")) {
          // Where its mask recedes, as the layer's panel sets it (the render's fallback is the photo's).
          const p = app.params(), l = p?.layers.at(-1);
          const f = waitFor((m) => m.type === "motionField", "motionField");
          app.send({ type: "motionField", layer: (p?.layers.length ?? 1) - 1 });
          const m = await f;
          if (p && l && m.type === "motionField" && m.vanish) {
            Object.assign(l.params, { vanish: m.vanish, range: m.range, reach: m.reach });
            report("motion:vanish", { vanish: m.vanish, range: m.range });
            const done = finalPreview("motion vanish"); app.pushParams(); await done; await settle();
          }
        }
        if (q.has("save")) await saveExport("motion");
      }
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
