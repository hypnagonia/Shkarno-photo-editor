/// <reference lib="webworker" />
/** Worker entry: all decoding, GPU work and inference happen here, off the UI thread. */
import { Engine } from "./engine.ts";
import { forcePhone } from "../device.ts";
import type { FromWorker, ToWorker } from "./protocol.ts";
import { MAX_FOCUS_POINTS } from "../decision/params.ts";

const post = (m: FromWorker, transfer: Transferable[] = []) => (self as unknown as DedicatedWorkerGlobalScope).postMessage(m, transfer);
const engine = new Engine(post);

// A rejected promise nobody awaited must still reach the page (not leave it waiting).
self.addEventListener("unhandledrejection", (e) => {
  post({ type: "error", message: e.reason instanceof Error ? e.reason.message : String(e.reason) });
});
self.onmessage = (ev: MessageEvent<ToWorker>) => {
  const m = ev.data;
  // Cheap, idempotent messages are handled immediately; heavy ones are serialised.
  if (m.type === "params") { engine.setParams(m.params, m.draft); return; }
  if (m.type === "view") { engine.setView(m.view, m.before, m.region, m.range); return; }
  if (m.type === "preview-size") { engine.setPreviewSize(m.long); return; }
  if (m.type === "canvas") { engine.setCanvas(m.canvas); return; }
  // A new photo stops the open still running (it would otherwise finish first, in the queue).
  if (m.type === "open") engine.cancelOpen();
  // Heavy work shares the engine's serial GPU queue with preview renders.
  engine.exclusive(() => handle(m)).catch((e: unknown) => {
    post({ type: "error", message: e instanceof Error ? e.message : String(e) });
    console.error(e);
  });
};

async function handle(m: ToWorker) {
  // Without a GPU nothing else can run: report why it did not start, not a crash inside the pipeline.
  if (m.type !== "init" && !engine.ready) {
    if (m.type === "open" || m.type === "export") throw new Error(engine.initError ?? "The GPU is not ready yet.");
    return;
  }
  switch (m.type) {
    case "init": {
      if (m.phone) forcePhone(true);
      const caps = await engine.init(m.base, m.forceCpu);
      post({ type: "ready", caps, looks: engine.looks() });
      break;
    }
    case "open":
      await engine.open(m.file, m.resolution, m.autoExposure, m.autoDof, m.upscale, m.safeAnalysis, m.analysis, m.series);
      break;
    case "toneEqHist":
      post({ type: "toneEqHist", hist: await engine.toneEqHistograms() });
      break;
    case "toneEqZone":
      post({ type: "toneEqZone", zone: await engine.toneEqZoneAt(m.x, m.y) });
      break;
    case "seriesView":
      await engine.seriesView(m.single);
      break;
    case "upscale-now":
      engine.forceUpscale();
      break;
    case "preview-zoom":
      await engine.resizePreview(m.long);
      break;
    case "focus": {
      const s = engine.session;
      if (!s) break;
      let points = [...s.params.dof.points];
      const a = s.decision.dofSuggestion;
      if (m.action === "clear") {
        points = [];
      } else if (m.action === "move") {
        // Dragging a ring: the point takes the distance at its new place. With no
        // points yet, the ring dragged is the automatic one, which becomes a point.
        const f = engine.focusRangeAt(m.x, m.y);
        if (!f) break;
        const moved = { x: m.x, y: m.y, dist: f.dist, range: f.range };
        if (m.index >= 0 && m.index < points.length) points[m.index] = moved;
        else if (!points.length) points.push(moved);
        else break;
      } else {
        // Tapping an existing point removes it; anywhere else adds one.
        const hit = points.findIndex((p) => Math.hypot(p.x - m.x, p.y - m.y) < 0.045);
        if (hit >= 0) points.splice(hit, 1);
        else {
          const f = engine.focusRangeAt(m.x, m.y);
          if (!f) break;
          // The automatic subject stays: the first manual point is added to it
          // instead of replacing it (and it can be moved or removed like any other).
          // Only when depth of field is already on: otherwise the automatic ring was
          // never shown, and a tap should make just that one point sharp. Its
          // distance is the current focus (the user may have moved the slider).
          if (!points.length && s.params.enable.dof && a.x !== undefined && a.y !== undefined && Math.hypot(a.x - m.x, a.y - m.y) >= 0.045) {
            const fz = s.params.dof.focus, sp = s.params.dof.focusSpan ?? [0, 0];
            points.push({ x: a.x, y: a.y, dist: fz, auto: true, range: [Math.max(0, fz - sp[0]), Math.min(1, fz + sp[1])] });
          }
          points.push({ x: m.x, y: m.y, dist: f.dist, range: f.range });
          if (points.length > MAX_FOCUS_POINTS) points.shift();
        }
      }
      const dof = { ...s.params.dof, points, strength: s.params.dof.strength > 0 ? s.params.dof.strength : 0.5 };
      s.params = { ...s.params, dof, enable: { ...s.params.enable, dof: s.params.enable.dof || points.length > 0 } };
      post({ type: "params", params: s.params });
      await engine.renderNow(true);
      break;
    }
    case "export": {
      const r = await engine.export(m.format, m.quality, m.space, m.stripRows);
      post({ type: "exported", blob: r.blob, name: r.name, ms: r.ms });
      break;
    }
    case "thumbs":
      post({ type: "thumbs", items: await engine.thumbnails(m.profiles, m.long) });
      break;
    case "pick": {
      // Always answered (the page waits for it before taking the next tap).
      const info = await engine.pickAt(m.x, m.y, m.layer, m.object).catch((e) => { post({ type: "log", text: `pick failed: ${e instanceof Error ? e.message : e}` }); return undefined; });
      post({ type: "pick", info });
      break;
    }
    case "brightest":
      post({ type: "brightest", ...(await engine.brightestPoint()) });
      break;
    case "mem":
      post({ type: "mem", ...engine.memStats() });
      break;
    case "check": {
      const r = await engine.check();
      post({ type: "check", ...r }, [r.rgba.buffer, ...r.items.flatMap((i) => (i.mask ? [i.mask.buffer] : []))]);
      // The fixes: one short GPU job per render (edits are not held up), each sent when found.
      void engine.solveCheckFixes((id, fix, partial) => post({ type: "checkFix", id, fix, partial }))
        .catch(() => undefined).then(() => post({ type: "checkFix", done: true }));
      break;
    }
    case "palette":
      post({ type: "palette", stats: await engine.palette() });
      break;
    case "reference": {
      const r = await engine.reference(m.file, m.mode, m.amount);
      post({ type: "lookProfile", profile: r.profile, reference: r.reference, message: r.message });
      break;
    }
    case "importLook":
      post({ type: "looks", looks: engine.importLook(m.name, m.text) });
      break;
  }
}
