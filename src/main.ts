/**
 * UI thread. Deliberately thin: it owns the DOM and a canvas, and forwards
 * everything else to the engine worker. No pixel processing happens here.
 */
import "./styles.css";
import type { Capabilities, ExportFormat, FromWorker, StageProfile, Summary, ToWorker, UpscaleInfo, UpscaleMode } from "./engine/protocol.ts";
import { createToneEqPanel } from "./ui/toneEqPanel.ts";
import { createContrastEqPanel } from "./ui/contrastEqPanel.ts";
import { createFilmPanel } from "./ui/filmPanel.ts";
import { keepRuntimeWhenIdle } from "./pwa.ts";
import logoSvg from "../public/logo.svg?raw";
import { clearFeedback, feedbackBlob, feedbackCount, recordEdit } from "./ui/autoFeedback.ts";
import { DEPTH_BANDS, defaultParams, type Decision, type DepthBand, type Params } from "./decision/params.ts";
import { createLookPanel } from "./ui/lookPanel.ts";
import { createRegionsPanel } from "./ui/regionsPanel.ts";
import { createLlmPanel } from "./ui/llmPanel.ts";
import { normalizeProfile } from "./looks/profile.ts";
import { createToneCurves } from "./ui/toneCurves.ts";
import { History as EditHistory } from "./layers/history.ts";
import { createLayersPanel, layerName } from "./ui/editor/layersPanel.ts";
import { icon, type IconName } from "./ui/editor/icons.ts";
import { AUTO_LAYERS_VERSION } from "./layers/auto.ts";
import { histogramOf } from "./analysis/previewHist.ts";
import { applyAutoCurves, type AutoCurveBands } from "./decision/autoCurves.ts";
import { isFlat } from "./render/curves.ts";
import { consumeCrash, crashInfo, crashedInAnalysis, crashedWhileProcessing, forgetPendingParams, lastStage, markCompleted, markInflight, noteAnalysis, noteStage, rememberParams, rememberPhoto, restorablePhoto } from "./ui/session.ts";
import { LANGS, LANG_NAMES, lang, setLang, storedLang, t, tOr, type Lang } from "./ui/i18n.ts";
import { el } from "./ui/dom.ts";
import { autotestAllowed } from "./autotest.ts";
import { forcePhone, isPhone } from "./device.ts";
import type { AnalysisLevel } from "./neural/scene.ts";
import type { ColorStats } from "./looks/palette.ts";
import type { CheckItem } from "./analysis/check.ts";
import { installTouchSliders } from "./ui/touchSlider.ts";
import { makeLayer } from "./layers/model.ts";

const worker = new Worker(new URL("./engine/worker.ts", import.meta.url), { type: "module" });
const send = (m: ToWorker) => worker.postMessage(m);
// The local autotest can ask for phone behaviour on a desktop browser (?autotest&phone).
const autotestPhone = autotestAllowed() && new URLSearchParams(location.search).has("phone");
if (autotestPhone) forcePhone(true);

// --------------------------------------------------------------------------- DOM helpers

const app = document.getElementById("app")!;
// The mark: an old rangefinder camera with a lens of paint (public/logo.svg), inline so
// its body takes the ink colour and follows the theme; the paint stays paint.
const logo = el("span", { class: "logo", "aria-hidden": "true" });
logo.innerHTML = logoSvg;
const header = el("header", { class: "top" }, logo, el("h1", { text: "Shkarno" }));
const capsEl = el("div", { class: "caps", text: t("app.starting") });
header.append(capsEl);
// Language: "Auto" follows the browser; a pick is remembered on this device (the page reloads).
// A pill shows the active code; the invisible native select on top of it opens the full list.
const langSel = el("select", { "aria-label": t("app.language") },
  el("option", { value: "", text: t("lang.auto") }), ...LANGS.map((l) => el("option", { value: l, text: LANG_NAMES[l] })));
langSel.value = storedLang() ?? "";
langSel.onchange = () => setLang((langSel.value || undefined) as Lang | undefined);
const langPill = el("label", { class: "btn small lang", title: t("app.language") }, lang.toUpperCase(), langSel);
// Theme: dark unless chosen otherwise; "System" follows the device. index.html applies
// the stored choice before first paint; this keeps it in step when it changes.
type Theme = "dark" | "light" | "system";
const THEMES: Theme[] = ["dark", "light", "system"];
const storedTheme = (): Theme => { try { const v = localStorage.getItem("theme"); return v === "light" || v === "system" ? v : "dark"; } catch { return "dark"; } };
const lightQuery = matchMedia("(prefers-color-scheme: light)");
function applyTheme(th: Theme) {
  const eff = th === "system" ? (lightQuery.matches ? "light" : "dark") : th;
  document.documentElement.dataset.theme = eff;
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", eff === "dark" ? "#0e0e0e" : "#ffffff");
}
lightQuery.addEventListener("change", () => { if (storedTheme() === "system") applyTheme("system"); });
const themeSel = el("select", { "aria-label": t("app.theme") }, ...THEMES.map((k) => el("option", { value: k, text: t(`theme.${k}`) })));
themeSel.value = storedTheme();
const themeText = document.createTextNode(t(`theme.${storedTheme()}`));
themeSel.onchange = () => {
  const th = themeSel.value as Theme;
  try { localStorage.setItem("theme", th); } catch { /* private mode */ }
  themeText.textContent = t(`theme.${th}`);
  applyTheme(th);
};
const themePill = el("label", { class: "btn small lang", title: t("app.theme") }, themeText, themeSel);
const stage = el("div", { class: "stage" });
let canvas = el("canvas");
const badge = el("div", { class: "badge" });
const rings = el("div", { class: "rings" });
const progress = el("div", { class: "progress" }, el("div", { class: "t" }), el("div", { class: "bar indet" }, el("i")));
const fileInput = el("input", { type: "file", accept: ".dng,.DNG,.heic,.HEIC,.heif,.jpg,.jpeg,.png,image/*,image/x-adobe-dng", style: "display:none" });
const empty = el("div", { class: "empty" },
  el("h2", { text: t("empty.title") }),
  el("p", { text: t("empty.body") }),
  el("button", { class: "btn primary", id: "open-btn", text: t("empty.open") }),
  el("p", { class: "muted fine", text: t("empty.fine") }),
);
stage.append(empty, canvas, rings, badge, progress, fileInput);
canvas.style.display = "none";

// The editor (src/ui/editor/layersPanel.ts): a dock with the layer stack (top first,
// Develop last) and the properties of the selected layer. On phones a bottom sheet
// whose grip toggles between compact and tall; on desktops the right-hand panel.
const sheet = el("div", { class: "sheet" });
const grip = el("button", { class: "grip", "aria-label": t("ui.more") });
const dockEl = el("div", { class: "dock" });
const propsEl = el("div", { class: "pane props" });
sheet.append(grip, dockEl, propsEl);
grip.onclick = () => sheet.classList.toggle("tall");
app.append(header, stage, sheet);

const panes: Record<string, HTMLElement> = {};
function addPane(id: string, _label?: string) {
  const p = el("div", { class: "pane-content" });
  panes[id] = p;
  return p;
}
// "More": information, export settings, upscaling, history and debugging, in one overlay.
const MORE = ["check", "auto", "history", "export", "upscale", "debug"] as const;
const moreEl = el("div", { class: "more", hidden: "" });
const moreTabs = el("div", { class: "seg" });
const moreBody = el("div", { class: "pane" });
const moreClose = el("button", { class: "btn small icon ghost", title: t("ui.close"), "aria-label": t("ui.close") });
moreClose.append(icon("close"));
moreClose.onclick = () => (moreEl.hidden = true);
moreEl.onclick = (e) => { if (e.target === moreEl) moreEl.hidden = true; }; // tap outside closes
moreEl.append(el("div", { class: "more-head" }, moreTabs, themePill, langPill, moreClose), moreBody);
app.append(moreEl);
let moreId: (typeof MORE)[number] = "check";
function showPane(id: string) {
  if ((MORE as readonly string[]).includes(id)) {
    moreId = id as (typeof MORE)[number];
    moreEl.hidden = false;
    moreTabs.replaceChildren(...MORE.map((k) => {
      const b = el("button", { class: k === moreId ? "on" : "", text: k === "history" ? t("ui.history") : t(`tab.${k}`) });
      b.onclick = () => showPane(k);
      return b;
    }));
    if (moreId === "history") renderHistory();
    if (moreId === "check" && checkStale) runCheck();
    moreBody.replaceChildren(panes[moreId]);
  }
}
const adjustPane = addPane("adjust");
// Hidden for now: the Look and Ask AI panels still exist, off-screen.
const lookPane = el("div");
// Regions are now layer masks; the old panel stays off-screen.
const regionsPane = el("div");
const depthPane = addPane("depth");
const upscalePane = addPane("upscale");
const llmPane = el("div");
const exportPane = addPane("export");
const debugPane = addPane("debug");
const autoPane = addPane("auto");
const historyPane = addPane("history");
const checkPane = addPane("check");
/** Develop: the RAW development (exposure, tone, colour, detail). Blur: depth of field, its own card. */
const developEl = el("div", { class: "develop" }, adjustPane);
const blurEl = el("div", { class: "develop" }, depthPane);
autoPane.append(el("p", { class: "muted", text: t("auto.hint") }));

// --------------------------------------------------------------------------- state
let caps: Capabilities | undefined;
let params: Params | undefined;
let autoParams: Params | undefined;
let decisions: Decision[] = [];
let summary: Summary | undefined;
/** The automatic 2× stage of the open photo. */
let upscale: UpscaleInfo | undefined;
let profile: StageProfile[] = [];
const logLines: string[] = [];
let looks: Array<{ id: string; name: string; description: string }> = [];
let dofInfo: { justified: boolean; focus: number; strength: number; reason: string; x?: number; y?: number; zones?: Array<{ share: number; label: string; lo: number; hi: number }>; bands?: Array<{ share: number; label: string; lo: number; hi: number }> } | undefined;
let focusMode = false;
/** Taps on the photo pick what a layer's mask selects (the layer's Mask tab turns this on). */
let maskPicking = false;
/** The tone equalizer is open: a tap on the photo finds its zone. */
let toneEqPicking = false;
/** Its mask (zones) shown on the photo (view 9). */
let toneEqMask = false;
let toneEqHist: Extract<FromWorker, { type: "toneEqHist" }>["hist"];
/**
 * A tap that edits a mask. One at a time: taps while one is being worked out (the
 * first selection on a photo takes seconds on a phone), and a second tap of a quick
 * double tap, are ignored — so a tap never adds and then removes by accident. The tap
 * is marked on the photo at once and "Selecting…" shows until the mask is back.
 */
let pickBusy = false, lastPickAt = 0, pickTimer = 0, pickAwaitsPreview = false;
/** The layer and reading the pending tap was for (its answer goes to that layer only). */
let pickFor: { layer: string; as: "object" | "color" | "depth" } | undefined;

/**
 * One request to the engine and its answer — always settles: with `fallback` when the
 * engine reports an error instead (a lost GPU, no photo) or does not answer in `ms`.
 */
function askEngine<T>(req: ToWorker, answer: (m: FromWorker) => T | undefined, fallback: T, ms = 15_000): Promise<T> {
  return new Promise((resolve) => {
    const done = (v: T) => { clearTimeout(timer); engineListeners.delete(f); resolve(v); };
    const f = (m: FromWorker) => { if (m.type === "error") return done(fallback); const v = answer(m); if (v !== undefined) done(v); };
    const timer = setTimeout(() => done(fallback), ms);
    engineListeners.add(f);
    send(req);
  });
}
let pickMark: HTMLElement | undefined;
function maskTap(x: number, y: number, cx: number, cy: number) {
  // A lens flare waiting for its light takes the tap itself (no engine round trip).
  if (layersPanel.placeAt(x, y)) return;
  const now = performance.now();
  if (pickBusy || now - lastPickAt < 400) return;
  const target = layersPanel.pickTarget();
  if (!target) return;
  pickFor = { layer: target.id, as: target.as };
  pickBusy = true;
  lastPickAt = now;
  pickMark?.remove();
  const st = stage.getBoundingClientRect();
  pickMark = el("div", { class: "tap-mark busy" });
  pickMark.style.left = `${cx - st.left}px`;
  pickMark.style.top = `${cy - st.top}px`;
  stage.append(pickMark);
  badge.textContent = t("mask.selecting");
  badge.classList.add("on");
  // Never stuck: if no answer comes, taps work again after a while.
  clearTimeout(pickTimer);
  pickTimer = window.setTimeout(pickDone, 30_000);
  send({ type: "pick", x, y, layer: target.layer, object: target.object });
}
function pickDone() {
  clearTimeout(pickTimer);
  pickAwaitsPreview = false;
  pickBusy = false;
  lastPickAt = performance.now();
  const m = pickMark;
  pickMark = undefined;
  if (m) { m.classList.remove("busy"); m.classList.add("done"); setTimeout(() => m.remove(), 450); }
  if (maskPicking) { badge.textContent = t("mask.tapHint"); badge.classList.add("on"); }
}

/** A ring being dragged: which one, where it started, where it is now. */
let drag: { index: number; x0: number; y0: number; x: number; y: number; moved: boolean; ox: number; oy: number } | undefined;
/** Where the lone automatic ring was dropped, until the engine's reply makes it a point. */
let pendingAuto: { x: number; y: number } | undefined;
let busy = false;

(document.getElementById("open-btn") as HTMLButtonElement).onclick = () => fileInput.click();
fileInput.onchange = () => { const f = fileInput.files?.[0]; if (f) openFile(f); fileInput.value = ""; };
const openBtn = el("button", { class: "btn small ghost", text: t("app.open") });
openBtn.onclick = () => fileInput.click();
header.insertBefore(openBtn, capsEl);

// Fullscreen preview: only the photo (hold still compares with the camera rendering).
const fsBtn = el("button", { class: "btn small icon ghost fs-btn", title: t("app.fullscreen"), "aria-label": t("app.fullscreen") });
fsBtn.append(icon("full"));
const fsExit = el("button", { class: "fs-exit", text: "✕", "aria-label": t("app.exitFullscreen") });
function setFullscreen(on: boolean) {
  document.body.classList.toggle("fs", on);
  // Real fullscreen where the platform allows it (not iPhone Safari; the CSS mode covers that).
  const root = document.documentElement as HTMLElement & { webkitRequestFullscreen?: () => void };
  try {
    if (on && !document.fullscreenElement) (root.requestFullscreen?.() ?? root.webkitRequestFullscreen?.())?.catch?.(() => {});
    if (!on && document.fullscreenElement) void document.exitFullscreen();
  } catch { /* unsupported */ }
  requestAnimationFrame(() => renderRings());
}
fsBtn.onclick = () => setFullscreen(true);
fsExit.onclick = (e) => { e.stopPropagation(); setFullscreen(false); };
document.addEventListener("fullscreenchange", () => { if (!document.fullscreenElement) document.body.classList.remove("fs"); });
header.insertBefore(fsBtn, capsEl);
// Export in one tap, with the Export tab's current settings (format, colour, quality).
const exportTop = el("button", { class: "btn small primary push", text: t("app.export") });
exportTop.onclick = () => exportBtn.click();
exportTop.disabled = true; // until a photo is open
header.insertBefore(exportTop, fsBtn);
/** A small inline icon (stroke = current text colour). */
const iconBtn = (name: IconName, label: string) => { const b = el("button", { class: "btn small icon ghost", title: label, "aria-label": label }); b.append(icon(name)); return b; };
const undoBtn = iconBtn("undo", t("ui.undo"));
const redoBtn = iconBtn("redo", t("ui.redo"));
const moreBtn = iconBtn("more", t("ui.more"));
undoBtn.onclick = () => { flushCommit(); stepHistory(history.undo()); };
redoBtn.onclick = () => { flushCommit(); stepHistory(history.redo()); };
moreBtn.onclick = () => (moreEl.hidden ? showPane(moreId) : (moreEl.hidden = true));
// Check in one tap (the finished photo's technical findings); after a check it shows how many mistakes.
const checkBtn = iconBtn("check", t("tab.check"));
checkBtn.classList.add("check-btn");
const checkBadge = el("span", { class: "check-badge", hidden: "" });
checkBtn.append(checkBadge);
checkBtn.onclick = () => (!moreEl.hidden && moreId === "check" ? (moreEl.hidden = true) : showPane("check"));
header.insertBefore(undoBtn, exportTop);
header.insertBefore(redoBtn, exportTop);
header.append(checkBtn, moreBtn);
undoBtn.disabled = redoBtn.disabled = true;
stage.append(fsExit);

let resolution: "auto" | "full" | "half" = "auto";
let exposureSuggestion = 0;
// On unless the user turned it off (a damped, measured correction; see the decision engine).
let autoExposure = (() => { try { return localStorage.getItem("autoExposure") !== "0"; } catch { return true; } })();
let autoDof = (() => { try { return localStorage.getItem("autoDof") === "1"; } catch { return false; } })();
let upscaleMode: UpscaleMode = (() => { try { const v = localStorage.getItem("upscaleMode"); return v === "always" || v === "off" ? v : "auto"; } catch { return "auto"; } })();
/** The open photo (for "back to original size", which reopens it at 1×). */
let currentFile: File | undefined;
/** The open photo's camera white balance (Develop's neutral). */
let cameraWB: { temp: number; tint: number } | undefined;
/** The page died while processing: don't retry automatically — offer a lighter reopen. */
/** Scene analysis crashed this device before (the page died during segmentation or depth): it runs on the CPU from now on. */
const SAFE_ANALYSIS = "safeAnalysis";
const flag = (store: () => Storage, k: string, v?: boolean) => { try { if (v === undefined) return store().getItem(k) === "1"; if (v) store().setItem(k, "1"); else store().removeItem(k); } catch { /* private mode */ } return false; };
/**
 * Scene analysis this device can take, stepped down for good when the tab died in it:
 * dying during depth → segmentation only; during segmentation → no analysis. Stepping
 * only goes down, so a crash can never repeat in a loop.
 */
const ANALYSIS_LEVEL = "analysisLevel";
/** The step-down holds for a day and for this build only: a deploy (a fix) or time gets a fresh try. */
function analysisLevel(): AnalysisLevel | undefined {
  try {
    const raw = localStorage.getItem(ANALYSIS_LEVEL);
    const v = raw?.startsWith("{") ? (JSON.parse(raw) as { level: string; build: string; at: number }) : undefined;
    if (!v || v.build !== __BUILD__ || Date.now() - v.at > 864e5) { if (raw) localStorage.removeItem(ANALYSIS_LEVEL); return undefined; }
    return v.level === "seg" || v.level === "none" ? v.level : undefined;
  } catch { return undefined; }
}
/** Forget what a crash taught (Try depth again). */
function resetAnalysisLevel() { try { localStorage.removeItem(ANALYSIS_LEVEL); localStorage.removeItem(SAFE_ANALYSIS); } catch { /* private mode */ } }
function stepDownAnalysis() {
  const stage = crashedInAnalysis();
  if (!stage) return;
  const next: AnalysisLevel = stage === "depth" && analysisLevel() !== "none" ? "seg" : "none";
  try { localStorage.setItem(ANALYSIS_LEVEL, JSON.stringify({ level: next, build: __BUILD__, at: Date.now() })); } catch { /* private mode */ }
  logLines.push(`the tab stopped during ${stage}: scene analysis on this device is now "${next}"`);
}
const noteAnalysisStage = (stage?: string) => noteAnalysis(stage);
function offerSafeReopen(file: File, saved?: Params) {
  if (crashedInAnalysis()) flag(() => localStorage, SAFE_ANALYSIS, true);
  stepDownAnalysis();
  const box = el("div", { class: "empty" },
    el("h2", { text: t("reopen.title") }),
    el("p", { text: t("reopen.body", { file: file.name }) }),
    el("p", { class: "muted", text: t("reopen.stopped", { stage: lastStage() ?? t("reopen.unknown") }) }),
  );
  const half = el("button", { class: "btn primary", text: t("reopen.half") });
  half.onclick = () => { box.remove(); resolution = "half"; resSel.value = "half"; openFile(file, saved); };
  const other = el("button", { class: "btn", text: t("reopen.other") });
  other.onclick = () => { box.remove(); empty.style.display = ""; fileInput.click(); };
  box.append(el("div", { class: "actions", style: "justify-content:center" }, half, other));
  empty.style.display = "none";
  stage.append(box);
}

/** This photo's edits were restored from before a reload (automatic additions leave them alone). */
let restored = false;
/** Parameters to re-apply once a restored photo has been analysed. */
let pendingRestore: Params | undefined;

/** The selected layer's mask shown on the photo (its index among the live layers), if any. */
let maskIndex: number | undefined;
/** What the photo shows when nothing temporary is on: the mask view, or the chosen view. */
function baseView(): { type: "view"; view: 0 | 1 | 2 | 6 | 9; region?: number } {
  if (toneEqMask) return { type: "view", view: 9 };
  return maskIndex !== undefined ? { type: "view", view: 6, region: maskIndex } : { type: "view", view: currentView };
}
/** A photo is being opened (its analysis has not arrived): late messages about the previous one are ignored. */
let opening = false;
function openFile(f: File, restore?: Params, upscaleOverride?: UpscaleMode) {
  // What was changed after the automatic development of the photo being left (this device only).
  if (!opening && params && autoParams) { recordEdit("leave", summary, autoParams, params); renderFeedback(); }
  opening = true;
  layersPanel.reset(); // nothing of the previous photo's editing state (tab, picking, a flare waiting) survives
  forgetPendingParams();
  // A mask shown for the previous photo's layer must not colour the new one's first previews.
  if (maskIndex !== undefined) { maskIndex = undefined; send(baseView()); }
  pendingRestore = restore;
  upscale = undefined;
  currentFile = f;
  // A new photo starts unzoomed, with the normal preview resolution.
  zoom = 1; panX = 0; panY = 0;
  canvas.style.transform = "";
  clearTimeout(zoomTimer);
  pendingAuto = undefined;
  // Through the engine's queue, ahead of the open: a zoom resize still queued
  // there must not leave the new photo at the zoomed size.
  if (sentPreviewLong) { sentPreviewLong = 0; send({ type: "preview-zoom", long: basePreviewLong() }); }
  renderUpscale();
  if (!restore) void rememberPhoto(f);
  markInflight();
  empty.style.display = "none";
  canvas.style.display = "block";
  logLines.length = 0;
  setProgress(t("progress.opening", { file: f.name }));
  busy = true;
  send({ type: "open", file: f, resolution, autoExposure, autoDof, upscale: upscaleOverride ?? upscaleMode, safeAnalysis: flag(() => localStorage, SAFE_ANALYSIS), analysis: analysisLevel() });
}

// Drag & drop on desktop.
stage.addEventListener("dragover", (e) => e.preventDefault());
stage.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer?.files?.[0]; if (f) openFile(f); });

function setProgress(text: string | undefined, frac?: number) {
  progress.classList.toggle("on", !!text);
  if (!text) return;
  (progress.querySelector(".t") as HTMLElement).textContent = text;
  const bar = progress.querySelector(".bar") as HTMLElement;
  bar.classList.toggle("indet", frac === undefined);
  (bar.querySelector("i") as HTMLElement).style.width = frac === undefined ? "" : `${Math.round(frac * 100)}%`;
}

/** Worker progress (English stage ids + free-form detail) in the user's language; unknown text passes through. */
function stageText(stage: string, detail?: string): string {
  const ref = /^reference (.+)$/.exec(stage), dl = /^download (.+)$/.exec(stage);
  const name = ref ? t("stage.reference", { what: stageText(ref[1]) }) : dl ? t("stage.download", { what: dl[1] }) : tOr(`stage.${stage}`, stage);
  if (!detail) return name;
  let m: RegExpExecArray | null;
  const d = (m = /^tile (\d+)\/(\d+)$/.exec(detail)) ? t("detail.tile", { n: m[1], total: m[2] })
    : (m = /^rendering (\d+)%$/.exec(detail)) ? t("detail.rendering", { pct: m[1] })
    : (m = /^writing (\S+)$/.exec(detail)) ? t("detail.writing", { format: m[1] })
    : (m = /^encoding (\S+)$/.exec(detail)) ? t("detail.encoding", { format: m[1] })
    : tOr(`detail.${detail}`, detail);
  return `${name} — ${d}`;
}

// --------------------------------------------------------------------------- preview
// Created on first use: a canvas with a 2D context can no longer be handed to the worker.
let ctx2d: CanvasRenderingContext2D | undefined;
const ctx = () => (ctx2d ??= (() => {
  try { return canvas.getContext("2d", { colorSpace: "display-p3" }) as CanvasRenderingContext2D; } catch { return canvas.getContext("2d")!; }
})());
/**
 * GPU display: the canvas belongs to the worker, which draws each preview into it
 * (no readback, transfer or putImageData per frame). Off with ?display=cpu, or
 * where the browser cannot hand a canvas over or set it up for WebGPU.
 */
let gpuDisplay = false;
/** Size of the frame on screen (the canvas's own size cannot be read reliably once it is the worker's). */
let shownW = 0, shownH = 0;
const dispW = () => shownW || canvas.width, dispH = () => shownH || canvas.height;
function handCanvasToWorker() {
  if (new URLSearchParams(location.search).get("display") === "cpu" || caps?.backend !== "webgpu") return;
  if (typeof canvas.transferControlToOffscreen !== "function" || ctx2d) return;
  try {
    const off = canvas.transferControlToOffscreen();
    gpuDisplay = true;
    worker.postMessage({ type: "canvas", canvas: off } satisfies ToWorker, [off]);
  } catch { gpuDisplay = false; }
}
/** The worker could not use the canvas: a fresh one, drawn by the page as before. */
function takeCanvasBack() {
  gpuDisplay = false;
  const fresh = el("canvas");
  fresh.className = canvas.className;
  fresh.style.cssText = canvas.style.cssText;
  canvas.replaceWith(fresh);
  canvas = fresh;
  ctx2d = undefined;
  shownW = shownH = 0;
}
/** Drafts (smaller, same shape) are drawn scaled through this, so the canvas keeps its size during a drag. */
const scratch = document.createElement("canvas");
const scratchCtx = (() => { try { return scratch.getContext("2d", { colorSpace: "display-p3" }); } catch { return null; } })() ?? scratch.getContext("2d")!;
function drawPreview(m: Extract<FromWorker, { type: "preview" }>) {
  if (!m.data) {
    // Already on the canvas (GPU display): only its size is news here.
    shownW = m.width; shownH = m.height;
    renderRings();
    return;
  }
  if (gpuDisplay) return; // the canvas is the worker's (a frame from before the hand-over)
  shownW = shownH = 0;
  let img: ImageData;
  try { img = new ImageData(new Uint8ClampedArray(m.data), m.width, m.height, { colorSpace: "display-p3" }); }
  catch { img = new ImageData(new Uint8ClampedArray(m.data), m.width, m.height); }
  const sameShape = Math.abs(m.width / m.height - canvas.width / Math.max(1, canvas.height)) < 0.01;
  if (!m.final && m.width < canvas.width && sameShape) {
    // Resizing the canvas reallocates its backing store twice per drag (in and out of drafts).
    if (scratch.width !== m.width || scratch.height !== m.height) { scratch.width = m.width; scratch.height = m.height; }
    scratchCtx.putImageData(img, 0, 0);
    ctx().drawImage(scratch, 0, 0, canvas.width, canvas.height);
  } else {
    if (canvas.width !== m.width || canvas.height !== m.height) { canvas.width = m.width; canvas.height = m.height; }
    ctx().putImageData(img, 0, 0);
  }
  renderRings();
}

/** On-screen rectangle of the photo inside the letterboxed canvas (object-fit: contain). */
function imageRect() {
  const c = canvas.getBoundingClientRect();
  const k = Math.min(c.width / (dispW() || 1), c.height / (dispH() || 1));
  const w = dispW() * k, h = dispH() * k;
  return { left: c.left + (c.width - w) / 2, top: c.top + (c.height - h) / 2, width: w, height: h };
}

/**
 * The rings shown while picking focus: the focus points, or — before any are
 * added — the automatic subject, which is kept (and editable) as the first
 * point once manual points are added.
 */
function shownFocusPoints(): Array<{ x: number; y: number; auto?: boolean }> {
  if (!focusMode || !params?.enable.dof) return [];
  // The ring being dragged is drawn where the finger is; params stay untouched
  // until the engine has measured the new place (a reply may arrive mid-drag).
  const d = drag?.moved ? drag : undefined;
  if (params.dof.points.length) {
    return params.dof.points.map((q, i) => (d && i === d.index ? { x: d.x, y: d.y } : q));
  }
  if (d) return [{ x: d.x, y: d.y, auto: true }];
  if (pendingAuto) return [{ ...pendingAuto, auto: true }];
  return dofInfo?.x !== undefined && dofInfo.y !== undefined ? [{ x: dofInfo.x, y: dofInfo.y, auto: true }] : [];
}

/** One ring per focus point; numbered so several subjects can be told apart. */
function renderRings() {
  // Rings are an editing aid: only shown while picking focus points.
  const pts = shownFocusPoints();
  if (!pts.length && !rings.childElementCount) return; // nothing to draw: no layout reads (pinch / pan call this per move)
  const r = imageRect(), st = stage.getBoundingClientRect();
  rings.replaceChildren(...pts.map((p, i) => {
    const d = el("div", { class: "focus-ring" + (p.auto ? " auto" : "") + (drag?.index === i ? " drag" : ""), text: p.auto ? t("view.ringAuto") : String(i + 1) });
    d.style.left = `${r.left - st.left + p.x * r.width}px`;
    d.style.top = `${r.top - st.top + p.y * r.height}px`;
    return d;
  }));
}
window.addEventListener("resize", () => applyZoom());

// Zoom and pan: pinch (or wheel / trackpad) zooms around the fingers, one finger
// pans a zoomed photo, double-tap zooms in and back out. The canvas is only
// transformed with CSS; once the gesture settles the preview is re-rendered at
// a higher resolution so the zoomed view is real detail, not enlarged pixels.
let zoom = 1, panX = 0, panY = 0;
/**
 * Deepest zoom: 4 screen pixels per photo pixel (pixel-peeping), at least 16×.
 * Zoom 1 = the photo fitted to the stage.
 */
function maxZoom() { return Math.max(16, 4 * pixelZoom()); }
/** The zoom at which one photo pixel is one screen pixel (100 %). */
function pixelZoom() {
  const st = stage.getBoundingClientRect();
  // (After the 2× stage the working image is twice the size the analysis reported.)
  const up = upscale?.upscaleApplied && upscale.width && upscale.height ? upscale : undefined;
  const long = up ? Math.max(up.width!, up.height!) : Math.max(summary?.working.width ?? dispW(), summary?.working.height ?? dispH());
  const fit = Math.min(st.width / (dispW() || 1), st.height / (dispH() || 1)) * Math.max(dispW(), dispH()) * (window.devicePixelRatio || 1);
  return long / Math.max(1, fit);
}
const pointers = new Map<number, { x: number; y: number }>();
let pinch: { d0: number; z0: number; cx0: number; cy0: number; px0: number; py0: number } | undefined;
/** The current one-finger gesture: where it started, whether it became a pan. */
let press: { x0: number; y0: number; px0: number; py0: number; moved: boolean; tap?: { x: number; y: number } } | undefined;
let lastTap = { t: 0, x: 0, y: 0 };
let zoomTimer = 0;
let sentPreviewLong = 0;
// (The phone autotest measures an iPhone 16 Pro's screen, 874 pt tall, not the desktop window.)
const basePreviewLong = () => (autotestPhone ? 874 : Math.max(window.innerWidth, window.innerHeight)) * Math.min(2, window.devicePixelRatio || 1);

function applyZoom() {
  const st = stage.getBoundingClientRect();
  // Photo size on screen at zoom 1 (object-fit: contain).
  const k = Math.min(st.width / (dispW() || 1), st.height / (dispH() || 1));
  const maxX = Math.max(0, (dispW() * k * zoom - st.width) / 2), maxY = Math.max(0, (dispH() * k * zoom - st.height) / 2);
  panX = Math.min(maxX, Math.max(-maxX, panX));
  panY = Math.min(maxY, Math.max(-maxY, panY));
  canvas.style.transform = zoom === 1 ? "" : `translate(${panX}px, ${panY}px) scale(${zoom})`;
  renderRings();
  clearTimeout(zoomTimer);
  zoomTimer = window.setTimeout(() => {
    // Deep zoom: the photo's full resolution where memory allows (phones: ≈ 7 MP previews).
    const long = Math.max(summary?.working.width ?? 0, summary?.working.height ?? 0);
    const cap = isPhone() ? 3072 : Math.max(4096, Math.min(8192, long));
    // Two steps only (not one size per zoom level): every size is a new proxy and render targets.
    const want = Math.round(zoom > 2.2 ? cap : zoom > 1.2 ? Math.min(cap, basePreviewLong() * 1.8) : basePreviewLong());
    if (params && Math.abs(want - sentPreviewLong) > 64) { sentPreviewLong = want; send({ type: "preview-zoom", long: want }); }
  }, 300);
}
/** Zooms to `z` keeping the stage point (clientX, clientY) where it is. */
function zoomAt(z: number, clientX: number, clientY: number) {
  const st = stage.getBoundingClientRect();
  const fx = clientX - (st.left + st.width / 2), fy = clientY - (st.top + st.height / 2);
  const z1 = Math.min(maxZoom(), Math.max(1, z));
  panX = fx - (fx - panX) * (z1 / zoom);
  panY = fy - (fy - panY) * (z1 / zoom);
  zoom = z1;
  if (zoom === 1) { panX = 0; panY = 0; }
  applyZoom();
}
function resetZoom() { zoom = 1; panX = 0; panY = 0; applyZoom(); }
stage.addEventListener("wheel", (e) => {
  if (!params) return;
  e.preventDefault();
  // Trackpad pinch arrives as ctrl+wheel; a mouse wheel moves in coarse steps (or by lines);
  // anything else is a two-finger trackpad scroll, which pans a zoomed photo.
  const mouseWheel = e.deltaMode !== 0 || (e.deltaX === 0 && Math.abs(e.deltaY) >= 50 && Number.isInteger(e.deltaY));
  if (e.ctrlKey || mouseWheel) zoomAt(zoom * Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.002)), e.clientX, e.clientY);
  else { panX -= e.deltaX; panY -= e.deltaY; applyZoom(); }
}, { passive: false });

// Press and hold: before (camera rendering). Tap in focus mode: add/remove a focus point.
let holdTimer = 0;
/** Press and hold: "before" (the camera rendering) or, in the tone equalizer, the photo without it. */
let holding: false | "before" | "toneEq" = false;
stage.addEventListener("pointerdown", (e) => {
  if (!params || (e.target !== canvas && !pointers.size) || (e.pointerType === "mouse" && e.button !== 0)) return;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  stage.setPointerCapture(e.pointerId);
  if (pointers.size === 2) {
    // Second finger: a pinch. Whatever the first finger started is abandoned.
    clearTimeout(holdTimer);
    endHold();
    if (drag?.moved) endDragRing(); else drag = undefined;
    press = undefined;
    const [p1, p2] = [...pointers.values()];
    pinch = { d0: Math.hypot(p1.x - p2.x, p1.y - p2.y) || 1, z0: zoom, cx0: (p1.x + p2.x) / 2, cy0: (p1.y + p2.y) / 2, px0: panX, py0: panY };
    renderRings();
    return;
  }
  if (pointers.size > 2) return;
  press = { x0: e.clientX, y0: e.clientY, px0: panX, py0: panY, moved: false };
  if (maskPicking || toneEqPicking) {
    // A pick happens on release (a pan or a pinch picks nothing).
    const r = imageRect();
    const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    if (x >= 0 && y >= 0 && x <= 1 && y <= 1) press.tap = { x, y };
    // Tone equalizer: a tap finds the zone; holding shows the photo without the equalizer.
    if (toneEqPicking && params?.toneEq) {
      holdTimer = window.setTimeout(() => {
        if (!params?.toneEq) return;
        holding = "toneEq"; badge.textContent = t("teq.without"); badge.classList.add("on");
        send({ type: "params", params: { ...params, toneEq: { ...params.toneEq, enabled: false } } });
      }, 180);
    }
    return;
  }
  if (focusMode) {
    const r = imageRect();
    const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    if (x < 0 || y < 0 || x > 1 || y > 1) return;
    // On a ring: drag moves it, a tap removes it. Elsewhere: a new point, on release
    // (so a pan or a pinch does not add one).
    const hit = shownFocusPoints().findIndex((q) => Math.hypot(q.x - x, q.y - y) < 0.045);
    if (hit >= 0) {
      const q = shownFocusPoints()[hit];
      drag = { index: hit, x0: x, y0: y, x, y, moved: false, ox: q.x, oy: q.y };
      renderRings();
      return;
    }
    press.tap = { x, y };
    return;
  }
  const off = holdOff;
  if (off && params) {
    // An equalizer's tab: holding shows the photo without that equalizer.
    holdTimer = window.setTimeout(() => { if (!params) return; holding = "toneEq"; badge.textContent = t("ceq.without"); badge.classList.add("on"); send({ type: "params", params: off(params) }); }, 180);
    return;
  }
  holdTimer = window.setTimeout(() => { holding = "before"; badge.textContent = t("view.before"); badge.classList.add("on"); send({ ...baseView(), before: true }); }, 180);
});
stage.addEventListener("pointermove", (e) => {
  if (!pointers.has(e.pointerId)) return;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pinch && pointers.size >= 2) {
    const [p1, p2] = [...pointers.values()];
    const d = Math.hypot(p1.x - p2.x, p1.y - p2.y);
    const st = stage.getBoundingClientRect();
    const scx = st.left + st.width / 2, scy = st.top + st.height / 2;
    const z1 = Math.min(maxZoom(), Math.max(1, pinch.z0 * d / pinch.d0));
    // The photo point under the fingers' first midpoint follows their current midpoint.
    const c0x = pinch.cx0 - scx, c0y = pinch.cy0 - scy;
    panX = ((p1.x + p2.x) / 2 - scx) - (c0x - pinch.px0) * (z1 / pinch.z0);
    panY = ((p1.y + p2.y) / 2 - scy) - (c0y - pinch.py0) * (z1 / pinch.z0);
    zoom = z1;
    if (zoom === 1) { panX = 0; panY = 0; }
    applyZoom();
    return;
  }
  if (drag && params) {
    const r = imageRect();
    drag.x = clamp01((e.clientX - r.left) / r.width);
    drag.y = clamp01((e.clientY - r.top) / r.height);
    if (Math.hypot(drag.x - drag.x0, drag.y - drag.y0) > 0.01) drag.moved = true;
    if (drag.moved) renderRings();
    return;
  }
  if (!press) return;
  const dx = e.clientX - press.x0, dy = e.clientY - press.y0;
  if (!press.moved && Math.hypot(dx, dy) > 8) {
    press.moved = true;
    press.tap = undefined;
    if (!holding) clearTimeout(holdTimer);
  }
  if (press.moved && zoom > 1) { panX = press.px0 + dx; panY = press.py0 + dy; applyZoom(); }
});
function endDragRing(cancelled = false) {
  if (!drag) return;
  const d = drag;
  drag = undefined;
  const pts = params?.dof.points ?? [];
  if (d.moved) {
    // The point is found again by where it was (the list may have changed meanwhile).
    let index = -1;
    if (pts.length) {
      index = pts.findIndex((q) => Math.hypot(q.x - d.ox, q.y - d.oy) < 1e-3);
      if (index < 0) { renderRings(); return; } // it is gone: nothing to move
    } else pendingAuto = { x: d.x, y: d.y };
    send({ type: "focus", action: "move", index, x: d.x, y: d.y });
  }
  // A tap on a point removes it; a tap on the lone automatic ring changes nothing;
  // a touch the system cancelled changes nothing either.
  else if (!cancelled && pts.length) send({ type: "focus", action: "toggle", x: d.x0, y: d.y0 });
  renderRings();
}
function clamp01(v: number) { return Math.min(1, Math.max(0, v)); }
const endHold = () => {
  clearTimeout(holdTimer);
  if (holding === "before") send({ ...baseView(), before: false });
  if (holding === "toneEq" && params) send({ type: "params", params });
  if (holding) { holding = false; badge.classList.remove("on"); }
};
function pointerEnd(e: PointerEvent) {
  if (!pointers.delete(e.pointerId)) return;
  if (pinch) {
    // The pinch ends with its first lifted finger; the other one may keep panning.
    if (pointers.size < 2) pinch = undefined;
    const rest = [...pointers.values()][0];
    press = rest ? { x0: rest.x, y0: rest.y, px0: panX, py0: panY, moved: true } : undefined;
    return;
  }
  endDragRing(e.type !== "pointerup");
  const held = !!holding;
  endHold();
  const p = press;
  press = undefined;
  if (!p || p.moved || held || e.type === "pointercancel") return;
  if (p.tap) {
    if (maskPicking) maskTap(p.tap.x, p.tap.y, e.clientX, e.clientY);
    else if (toneEqPicking) void askEngine({ type: "toneEqZone", x: p.tap.x, y: p.tap.y }, (m) => (m.type === "toneEqZone" ? m.zone ?? -1 : undefined), -1).then((z) => { if (z >= 0) showToneEqZone(z); });
    else send({ type: "focus", action: "toggle", x: p.tap.x, y: p.tap.y });
    return;
  }
  if (focusMode) return;
  // Double-tap: zoom in to 100 % (one photo pixel per screen pixel, at least 2.5×) there, or back out.
  const now = performance.now();
  if (now - lastTap.t < 320 && Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < 30) {
    lastTap.t = 0;
    if (zoom > 1) resetZoom(); else zoomAt(Math.max(2.5, pixelZoom()), e.clientX, e.clientY);
  } else lastTap = { t: now, x: e.clientX, y: e.clientY };
}
stage.addEventListener("pointerup", pointerEnd);
stage.addEventListener("pointercancel", pointerEnd);
// Capture lost without an up/cancel (e.g. a context menu): treat it as a cancel.
stage.addEventListener("lostpointercapture", (e) => { if (pointers.has(e.pointerId)) pointerEnd(new PointerEvent("pointercancel", { pointerId: e.pointerId })); });
// Leaving the page mid-gesture: forget every finger, so the next touch starts clean.
function resetGestures() {
  pointers.clear();
  pinch = undefined;
  press = undefined;
  if (drag) endDragRing(true);
  endHold();
}
window.addEventListener("blur", resetGestures);
document.addEventListener("visibilitychange", () => { if (document.hidden) resetGestures(); });
let currentView: 0 | 1 | 2 = 0;

// --------------------------------------------------------------------------- params plumbing
let pushTimer = 0;
// While a slider is held, previews render at a quarter of the pixels (drafts);
// releasing it renders the full preview once.
let dragging = false;
/** Something was changed during this drag (a draft went out): the release renders the final preview. */
let draftSent = false;
installTouchSliders(); // a touch anywhere on a slider sets it (not only on its knob)
document.addEventListener("pointerdown", (e) => { if ((e.target as HTMLElement).matches?.('input[type="range"], .curve-editor, .hs-range, .grad-handle, .grad-handle *')) { dragging = true; draftSent = false; } }, true);
// A touch on a curve box that turned into a page scroll changed nothing: no render.
const endDrag = () => { if (!dragging) return; dragging = false; if (draftSent) pushParams(); };
document.addEventListener("pointerup", endDrag, true);
document.addEventListener("pointercancel", endDrag, true);

// History: one step per finished gesture (drafts while a control is held do not count).
const history = new EditHistory();
let nextLabel = "";
let histTimer = 0;
/** The label of the step waiting to be committed ("" = none): a drag's release keeps it. */
let pendingLabel = "";
function scheduleCommit(label: string) {
  clearTimeout(histTimer);
  pendingLabel = label;
  histTimer = window.setTimeout(() => {
    if (!params) return;
    if (dragging) { scheduleCommit(label); return; }
    pendingLabel = "";
    history.commit(params, label);
    updateUndo();
  }, 450);
}
/** Commits a waiting step now (before undo / redo, so a quick edit is not lost). */
function flushCommit() {
  if (!pendingLabel || !params) return;
  clearTimeout(histTimer);
  history.commit(params, pendingLabel);
  pendingLabel = "";
}
function updateUndo() {
  undoBtn.disabled = !history.canUndo;
  redoBtn.disabled = !history.canRedo;
  if (!moreEl.hidden && moreId === "history") renderHistory();
}
function stepHistory(p: Params | undefined) {
  if (!p) return;
  markCheckStale();
  clearTimeout(histTimer);
  pendingLabel = "";
  params = p;
  syncControls();
  send({ type: "params", params: structuredClone(params), draft: false });
  rememberParams(params);
  updateUndo();
}
function renderHistory() {
  historyPane.replaceChildren(el("div", { class: "hist" }, ...history.list().map((h, i) => {
    const b = el("button", { class: "hist-row" + (h.current ? " on" : ""), text: h.label });
    b.onclick = () => { flushCommit(); stepHistory(history.go(i)); };
    return b;
  })));
}
function pushParams() {
  markCheckStale();
  paramsGen++;
  if (!params) return;
  scheduleCommit(nextLabel || pendingLabel || t("hist.develop"));
  nextLabel = "";
  clearTimeout(pushTimer);
  const draft = dragging;
  if (draft) draftSent = true;
  pushTimer = window.setTimeout(() => send({ type: "params", params: params!, draft }), draft ? 0 : 16); // postMessage copies
  rememberParams(params);
}
function getPath(o: unknown, path: string): number {
  return path.split(".").reduce((a: unknown, k) => (a as Record<string, unknown>)[k], o) as number;
}
function setPath(o: unknown, path: string, v: unknown) {
  const ks = path.split(".");
  const last = ks.pop()!;
  const t = ks.reduce((a: unknown, k) => (a as Record<string, unknown>)[k], o) as Record<string, unknown>;
  t[last] = v;
}

interface SliderDef { path: string; label: string; min: number; max: number; step: number; fmt?: (v: number) => string }
const sliders: Array<{ def: SliderDef; input: HTMLInputElement; out: HTMLOutputElement }> = [];
function slider(def: SliderDef): HTMLElement {
  const input = el("input", { type: "range", min: String(def.min), max: String(def.max), step: String(def.step) });
  const out = el("output");
  const row = el("div", { class: "row" }, el("label", { text: def.label }), input, out);
  input.oninput = () => {
    if (!params) return;
    setPath(params, def.path, parseFloat(input.value));
    if (!def.path.startsWith("profile.")) lookPanel.invalidate();
    refreshSlider(sliders.find((s) => s.input === input)!);
    pushParams();
  };
  // Double-tap the label: back to the automatic value.
  row.querySelector("label")!.addEventListener("dblclick", () => {
    if (!params || !autoParams) return;
    setPath(params, def.path, getPath(autoParams, def.path));
    syncControls();
    pushParams();
  });
  sliders.push({ def, input, out });
  return row;
}
function refreshSlider(s: { def: SliderDef; input: HTMLInputElement; out: HTMLOutputElement }) {
  if (!params) return;
  const v = getPath(params, s.def.path);
  s.input.value = String(v);
  s.out.textContent = s.def.fmt ? s.def.fmt(v) : v.toFixed(s.def.step < 0.1 ? 2 : s.def.step < 1 ? 1 : 0);
  const a = autoParams ? getPath(autoParams, s.def.path) : undefined;
  s.out.classList.toggle("auto", a !== undefined && Math.abs(a - v) < 1e-6);
}
const pct = (v: number) => `${v > 0 ? "+" : ""}${Math.round(v * 100)}`;

// Highlight protection (on by default): layers may not clip highlights that were not clipped.
const protectToggle = el("input", { type: "checkbox" });
protectToggle.onchange = () => {
  if (!params) return;
  params.protectHighlights = protectToggle.checked;
  nextLabel = t("adj.protectHighlights");
  pushParams();
};
// Display rendering: img (render_tone.wgsl img_render) or the classic one.
const imgToggle = el("input", { type: "checkbox" });
imgToggle.onchange = () => {
  if (!params) return;
  params.render = { ...(params.render ?? { purity: 0, strength: 1 }), engine: imgToggle.checked ? "img" : "classic" };
  nextLabel = t("adj.imgRender");
  lookPanel.invalidate();
  pushParams();
};
const aeToggle = el("input", { type: "checkbox" });
aeToggle.checked = autoExposure;
const aeNote = el("span", { class: "muted" });
aeToggle.onchange = () => {
  autoExposure = aeToggle.checked;
  try { localStorage.setItem("autoExposure", autoExposure ? "1" : "0"); } catch { /* private mode */ }
  if (params) { params.exposure = autoExposure ? exposureSuggestion : 0; if (autoParams) autoParams.exposure = params.exposure; syncControls(); pushParams(); }
};
adjustPane.append(
  el("div", { class: "group-title", text: t("adj.light") }),
  el("label", { class: "toggle" }, el("span", {}, t("adj.autoExposure") + " ", aeNote), aeToggle),
  slider({ path: "exposure", label: t("adj.exposure"), min: -3, max: 3, step: 0.05, fmt: (v) => `${v > 0 ? "+" : ""}${v.toFixed(2)}` }),
  slider({ path: "tone.highlights", label: t("adj.highlights"), min: -1, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "tone.shadows", label: t("adj.shadows"), min: -1, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "tone.whites", label: t("adj.whites"), min: -1, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "tone.blacks", label: t("adj.blacks"), min: -1, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "tone.contrast", label: t("adj.contrast"), min: -1, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "tone.rolloff", label: t("adj.rolloff"), min: 0, max: 1, step: 0.01, fmt: pct }),
  el("label", { class: "toggle" }, el("span", {}, t("adj.protectHighlights") + " ", el("span", { class: "muted", text: t("adj.protectHighlightsNote") })), protectToggle),
  el("div", { class: "group-title", text: t("adj.rendering") }),
  el("label", { class: "toggle" }, el("span", {}, t("adj.imgRender") + " ", el("span", { class: "muted", text: t("adj.imgRenderNote") })), imgToggle),
  slider({ path: "render.purity", label: t("adj.purity"), min: -1, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "render.strength", label: t("adj.renderStrength"), min: 0, max: 1, step: 0.01, fmt: (v) => `${Math.round(v * 100)}%` }),
  slider({ path: "hdr.headroom", label: t("adj.hdrHeadroom"), min: 0, max: 3, step: 0.25, fmt: (v) => (v ? `+${v.toFixed(2)} EV` : t("adj.hdrOff")) }),
  el("div", { class: "group-title", text: t("adj.wb") }),
  slider({ path: "wb.temp", label: t("adj.temp"), min: 2000, max: 12000, step: 10, fmt: (v) => `${Math.round(v)}K` }),
  slider({ path: "wb.tint", label: t("adj.tint"), min: -60, max: 60, step: 0.5 }),
  el("div", { class: "group-title", text: t("adj.local") }),
  slider({ path: "local.compression", label: t("adj.range"), min: 0, max: 0.8, step: 0.01, fmt: pct }),
  slider({ path: "local.clarity", label: t("adj.clarity"), min: -0.5, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "local.texture", label: t("adj.texture"), min: -0.5, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "dehaze.strength", label: t("adj.dehaze"), min: 0, max: 1, step: 0.01, fmt: pct }),
  el("div", { class: "group-title", text: t("adj.colour") }),
  slider({ path: "color.vibrance", label: t("adj.vibrance"), min: -1, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "color.saturation", label: t("adj.saturation"), min: -1, max: 1, step: 0.01, fmt: pct }),
  el("div", { class: "group-title", text: t("adj.detail") }),
  slider({ path: "denoise.luma", label: t("adj.noiseLuma"), min: 0, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "denoise.chroma", label: t("adj.noiseColour"), min: 0, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "sharpen.amount", label: t("adj.sharpen"), min: 0, max: 1.5, step: 0.01, fmt: pct }),
  slider({ path: "sharpen.radius", label: t("adj.radius"), min: 0.5, max: 2.5, step: 0.05 }),
  el("div", { class: "group-title", text: t("adj.depth") }),
  slider({ path: "depth.near", label: t("adj.nearDetail"), min: 0.5, max: 1.5, step: 0.01 }),
  slider({ path: "depth.far", label: t("adj.farDetail"), min: 0.2, max: 1.5, step: 0.01 }),
  el("div", { class: "group-title", text: t("adj.vignette") }),
  slider({ path: "vignette.amount", label: t("adj.vigAmount"), min: -1, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "vignette.midpoint", label: t("adj.vigMidpoint"), min: 0, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "vignette.feather", label: t("adj.vigFeather"), min: 0, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "vignette.roundness", label: t("adj.vigRoundness"), min: 0, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "vignette.highlights", label: t("adj.vigHighlights"), min: 0, max: 1, step: 0.01, fmt: pct }),
);
// Curves for this photo (L, R, G, B), independent of the look's own curves and of
// the per-region curves (Regions tab): tone-range sliders (src/ui/toneCurves.ts).
let histograms: Float32Array | undefined;
/** Dev builds only: a read-only view of the state for the automated photo checks (scripts, not the UI). */
let finalPreviews = 0;
if (import.meta.env.DEV) (globalThis as unknown as { __shk: unknown }).__shk = () => ({ summary, decisions, params, autoParams, finalPreviews, busy, log: logLines, gpuDisplay, caps,
  /** Test harness: change the parameters and render. */
  apply: (f: (p: Params) => void) => { if (params) { f(params); syncControls(); pushParams(); } } });
/** Share of the frame (%) of each region at each distance. */
let cellCov: Record<string, number> | undefined;
const photoCurves = createToneCurves({
  histogram: (c) => histogramOf(histograms, "photo", c),
  get: () => params?.curves,
  set: (c) => { if (params) params.curves = c; },
  changed: () => { lookPanel.invalidate(); pushParams(); },
  enabled: () => !!params,
});
// Strength of the automatic curves (photo, regions, skin, distance): rescales
// every curve that is still the automatic one; curves edited by hand are kept.
let autoCurveBands: AutoCurveBands | undefined;
const acInput = el("input", { type: "range", min: "0", max: "1", step: "0.01" });
const acOut = el("output");
const acRow = el("div", { class: "row" }, el("label", { text: t("adj.autoStrength") }), acInput, acOut);
function renderAutoCurves() {
  const k = params?.autoCurves ?? 1;
  acInput.value = String(k);
  acOut.textContent = `${Math.round(k * 100)}%`;
  acOut.classList.toggle("auto", Math.abs(k - 1) < 1e-6);
  acRow.hidden = !params?.layers.some((l) => l.auto);
}
// Auto strength: scales the opacity of every automatic layer (0…100 %).
function setAutoCurves(k: number) {
  if (!params) return;
  params.autoCurves = k;
  renderAutoCurves();
  nextLabel = t("hist.autoStrength");
  pushParams();
}
acInput.oninput = () => setAutoCurves(parseFloat(acInput.value));
acRow.querySelector("label")!.addEventListener("dblclick", () => setAutoCurves(1));
adjustPane.prepend(acRow);

const resetBtn = el("button", { class: "btn small", text: t("adj.reset") });
resetBtn.onclick = () => { if (autoParams) { params = structuredClone(autoParams); syncControls(); pushParams(); } };
/**
 * Every Develop control at the value that leaves the photo as developed: exposure,
 * tone, colour, detail, noise, haze, vignette, grain at 0, the camera's own white
 * balance, no automatic grade (Auto strength 0). The look and your own layers stay.
 */
function neutralParams(p: Params): Params {
  const d = defaultParams(), n = structuredClone(p);
  n.exposure = 0;
  n.wb = { ...(cameraWB ?? d.wb) };
  n.tone = { ...d.tone, displayReferred: p.tone.displayReferred };
  n.local = { ...d.local, anchorEV: p.local.anchorEV };
  n.color = { ...d.color };
  n.curves = d.curves;
  n.denoise = { ...d.denoise };
  n.sharpen = { ...p.sharpen, amount: 0 };
  n.dehaze = { ...p.dehaze, strength: 0 };
  n.semantic = d.semantic; n.skin = d.skin; n.distance = d.distance;
  n.cells = {}; n.cellCurves = {}; n.regionCurves = {}; n.depthCurves = {};
  n.autoCurves = 0;
  if (n.vignette) n.vignette = { ...n.vignette, amount: 0 };
  if (n.grain) n.grain = { ...n.grain, amount: 0 };
  if (n.film) n.film = { ...n.film, character: "off" };
  n.layers = (p.layers ?? []).filter((l) => !l.auto);
  return n;
}
const zeroBtn = el("button", { class: "btn small", text: t("adj.zero") });
zeroBtn.title = t("adj.zeroTip");
zeroBtn.onclick = () => { if (params) { params = neutralParams(params); nextLabel = t("adj.zero"); syncControls(); pushParams(); } };
adjustPane.append(el("div", { class: "actions" }, zeroBtn, resetBtn), el("p", { class: "muted", text: t("adj.amberHint") }));

// Look profiles: browser, palette, reference, editor (src/ui/lookPanel.ts)
const lookPanel = createLookPanel(lookPane, {
  send,
  params: () => params,
  changed: () => { syncControls(); pushParams(); },
  download,
  progress: (t) => setProgress(t),
});
function renderLooks() { lookPanel.sync(); }

// Ask an LLM: a prompt to copy out, an answer to paste back (src/ui/llmPanel.ts)
const llmPanel = createLlmPanel(llmPane, {
  params: () => params,
  auto: () => autoParams,
  summary: () => summary,
  decisions: () => decisions,
  looks: () => lookPanel.list(),
  cellCoverage: () => cellCov,
  selectLook: (id) => lookPanel.select(id),
  changed: () => { lookPanel.invalidate(); regionsPanel.render(); syncControls(); pushParams(); },
  canvas,
});

/**
 * A tap on the photo in the tone equalizer: its zone is selected in the panel, named on
 * the photo, and lit up there for a moment (the rest dimmed, view 11).
 */
let zoneFlash = 0;
function showToneEqZone(z: number) {
  toneEqPanel.selectZone(z);
  const g = params?.toneEq?.gains[z] ?? 0;
  badge.textContent = t("teq.tapped", { ev: String(z - 8), gain: `${g > 0 ? "+" : ""}${g.toFixed(2)}` });
  badge.classList.add("on");
  send({ type: "view", view: 11, region: z });
  clearTimeout(zoneFlash);
  zoneFlash = window.setTimeout(() => { badge.classList.remove("on"); send(baseView()); }, 1400);
}

// Tone equalizer (src/ui/toneEqPanel.ts): a card of its own in the dock.
/** The equalizer whose tab is open: holding the photo shows it without that one. */
let holdOff: ((p: Params) => Params) | undefined;
const contrastEqPanel = createContrastEqPanel({
  params: () => params,
  changed: (label) => { nextLabel = label; lookPanel.invalidate(); pushParams(); },
});
const contrastEqTab = {
  el: contrastEqPanel.el,
  render: () => { contrastEqPanel.render(); holdOff = ceqOff; },
  leave: () => { if (holdOff === ceqOff) holdOff = undefined; },
};
function ceqOff(p: Params): Params { return p.contrastEq ? { ...p, contrastEq: { ...p.contrastEq, enabled: false } } : p; }
const toneEqPanel = createToneEqPanel({
  params: () => params,
  changed: (label) => { nextLabel = label; lookPanel.invalidate(); pushParams(); },
  hist: () => toneEqHist,
  // The mask as the photo renders now (it follows exposure and local tone): asked when the panel opens.
  refreshHist: () => { void askEngine({ type: "toneEqHist" }, (m) => (m.type === "toneEqHist" ? { h: m.hist } : undefined), { h: toneEqHist }).then((r) => { toneEqHist = r.h; toneEqPanel.redraw(); }); },
  showMask: (on) => { if (toneEqMask === on) return; toneEqMask = on; send(baseView()); },
  pickMode: (on) => { toneEqPicking = on && !!params; },
});

const filmPanel = createFilmPanel({
  params: () => params,
  changed: (label) => { nextLabel = label; pushParams(); },
});

// Regions: per-segment controls (src/ui/regionsPanel.ts)
const layersPanel = createLayersPanel(dockEl, propsEl, {
  params: () => params,
  auto: () => autoParams,
  coverage: () => summary?.coverage,
  cellCoverage: () => cellCov,
  histogram: (target, c) => histogramOf(histograms, target, c),
  changed: (label) => { nextLabel = label; lookPanel.invalidate(); pushParams(); renderAutoCurves(); },
  // View 6: the layer's mask on the photo (what it does not reach, tinted red).
  showMask: (i) => { maskIndex = i; send(baseView()); },
  develop: developEl,
  blur: blurEl,
  toneEq: toneEqPanel,
  contrastEq: contrastEqTab,
  film: filmPanel,
  photoColors: () => askEngine({ type: "palette" }, (m) => (m.type === "palette" ? m.stats.palette.map((w) => w.hex) : undefined), [] as string[]),
  notice: (text) => { badge.textContent = text; badge.classList.add("on"); },
  brightest: () => askEngine({ type: "brightest" }, (m) => (m.type === "brightest" ? { x: m.x, y: m.y } : undefined), { x: 0.3, y: 0.2 }),
  pickMode: (on, hint) => {
    maskPicking = on;
    if (!on) { pickMark?.remove(); pickMark = undefined; }
    if (on && focusMode) setFocusMode(false);
    badge.textContent = on ? (hint ?? t("mask.pickBadge")) : "";
    badge.classList.toggle("on", on);
  },
  leftBlur: () => {
    if (focusMode) setFocusMode(false);
    if (zoneHighlight !== undefined) setZoneHighlight(undefined);
    if (bandShown) setBandShown(false);
  },
});
setTimeout(() => layersPanel.render(), 0); // after the whole page is built
const regionsPanel = createRegionsPanel(regionsPane, {
  params: () => params,
  auto: () => autoParams,
  coverage: () => summary?.coverage,
  cellCoverage: () => cellCov,
  bandShare: () => dofInfo?.bands?.map((b) => b.share),
  histogram: (target, c) => histogramOf(histograms, target, c),
  changed: () => { lookPanel.invalidate(); pushParams(); },
  // A region (view 4, optionally only at a depth range: a region at a distance) or only a depth range (view 5).
  highlight: (h) => send(h === undefined ? { type: "view", view: currentView }
    : h.region === undefined ? { type: "view", view: 5, range: h.range } : { type: "view", view: 4, region: h.region, range: h.range }),
});

// Depth / DoF
const dofToggle = el("input", { type: "checkbox" });
dofToggle.onchange = () => {
  if (!params) return;
  params.enable.dof = dofToggle.checked;
  if (params.dof.strength <= 0) params.dof.strength = 0.5;
  // With no focus points, focus follows the automatic subject.
  if (dofToggle.checked && !params.dof.points.length && dofInfo) params.dof.focus = dofInfo.focus;
  syncControls();
  pushParams();
};
const autoDofToggle = el("input", { type: "checkbox" });
autoDofToggle.checked = autoDof;
autoDofToggle.onchange = () => {
  autoDof = autoDofToggle.checked;
  try { localStorage.setItem("autoDof", autoDof ? "1" : "0"); } catch { /* private mode */ }
  // Apply to the current photo right away when the scene justifies it.
  if (params && dofInfo) {
    if (autoDof && dofInfo.justified) { params.enable.dof = true; params.dof.focus = dofInfo.focus; if (params.dof.strength <= 0) params.dof.strength = 0.5; }
    else if (!autoDof && !params.dof.points.length) params.enable.dof = false;
    syncControls();
    pushParams();
  }
};
const focusBtn = el("button", { class: "btn small", text: t("dof.pick") });
function setFocusMode(on: boolean) {
  focusMode = on;
  focusBtn.classList.toggle("primary", on);
  badge.textContent = on ? t("dof.pickBadge") : "";
  badge.classList.toggle("on", on);
  renderRings();
}
focusBtn.onclick = () => setFocusMode(!focusMode);
const autoFocusBtn = el("button", { class: "btn small", text: t("dof.automatic") });
autoFocusBtn.onclick = () => {
  if (!params) return;
  // Back to the automatic subject: drop focus points and send the focus with the parameters.
  params.dof.points = [];
  if (dofInfo) params.dof.focus = dofInfo.focus;
  syncControls();
  pushParams();
  renderRings();
};
const dofReason = el("p", { class: "muted" });
// This photo opened without a depth map (analysis failed, or a crash taught this device to skip it).
const noDepthText = el("p", { class: "muted" });
const retryDepth = el("button", { class: "btn small", text: t("dof.retryDepth") });
retryDepth.onclick = () => { resetAnalysisLevel(); if (currentFile) openFile(currentFile, params ? structuredClone(params) : undefined, undefined); };
const noDepthBox = el("div", { class: "no-depth", hidden: "" }, el("div", { class: "group-title", text: t("dof.noDepth") }), noDepthText, el("div", { class: "actions" }, retryDepth));
depthPane.append(noDepthBox);
depthPane.append(
  el("label", { class: "toggle" }, t("dof.autoToggle"), autoDofToggle),
  el("label", { class: "toggle" }, t("dof.toggle"), dofToggle),
  slider({ path: "dof.strength", label: t("dof.blur"), min: 0, max: 1, step: 0.01, fmt: pct }),
  slider({ path: "dof.focus", label: t("dof.autoFocus"), min: 0, max: 1, step: 0.005 }),
  el("p", { class: "muted", text: t("dof.pointsHint") }),
  el("div", { class: "actions" }, focusBtn, autoFocusBtn),
  dofReason,
  el("div", { class: "group-title", text: t("dof.inspect") }),
);
// ---- Depth zones: blur set by hand for each part of the depth range --------
const ZONE_NAMES = [t("zone.0"), t("zone.1"), t("zone.2"), t("zone.3"), t("zone.4")];
const zoneToggle = el("input", { type: "checkbox" });
const zoneRows = el("div");
let zoneHighlight: number | undefined;
function setZoneHighlight(i: number | undefined) {
  zoneHighlight = i;
  if (i !== undefined && bandShown) { bandShown = false; renderBands(); }
  send(i === undefined ? { type: "view", view: currentView } : { type: "view", view: 5, region: i });
  renderZones();
}
zoneToggle.onchange = () => {
  if (!params) return;
  params.dof.mode = zoneToggle.checked ? "zones" : "focus";
  if (zoneToggle.checked) {
    params.enable.dof = true;
    if (params.dof.strength <= 0) params.dof.strength = 0.5;
    if (!params.dof.zones && autoParams?.dof.zones) { params.dof.zones = [...autoParams.dof.zones]; params.dof.zoneBounds = [...(autoParams.dof.zoneBounds ?? [])]; }
  }
  syncControls();
  pushParams();
};
function renderZones() {
  if (!params?.dof.zones) { zoneRows.replaceChildren(el("p", { class: "muted", text: t("dof.zonesEmpty") })); return; }
  zoneToggle.checked = params.dof.mode === "zones";
  const on = params.dof.mode === "zones";
  zoneRows.replaceChildren(...ZONE_NAMES.map((name, i) => {
    const input = el("input", { type: "range", min: "0", max: "1", step: "0.01" });
    input.value = String(params!.dof.zones![i] ?? 0);
    input.disabled = !on;
    const out = el("output", { text: String(Math.round(+input.value * 100)) });
    input.oninput = () => {
      if (!params?.dof.zones) return;
      params.dof.zones[i] = +input.value;
      out.textContent = String(Math.round(+input.value * 100));
      pushParams();
    };
    const eye = el("button", { class: "chip eye" + (zoneHighlight === i ? " on" : ""), text: "◉", title: t("dof.showZone", { zone: name.toLowerCase() }), "aria-label": t("dof.showZone", { zone: name.toLowerCase() }) });
    eye.onclick = () => setZoneHighlight(zoneHighlight === i ? undefined : i);
    const z = dofInfo?.zones?.[i];
    const info = z ? el("span", { class: "zone-info", text: `${Math.round(z.share * 100)}%${z.label ? " · " + tOr(`group.${z.label}`, z.label).toLowerCase() : ""}` }) : "";
    return el("div", { class: "row zone-row" }, el("label", {}, eye, el("span", {}, ` ${name}`, info)), input, out);
  }));
}
const viewChips = el("div", { class: "chips" });
([[t("dof.viewPhoto"), 0], [t("dof.viewRegions"), 1], [t("dof.viewDepth"), 2]] as const).forEach(([name, v]) => {
  const b = el("button", { class: "chip" + (v === 0 ? " on" : ""), text: name });
  b.onclick = () => { currentView = v; for (const c of viewChips.children) c.classList.toggle("on", c === b); send({ type: "view", view: v }); };
  viewChips.append(b);
});
depthPane.append(viewChips, el("p", { class: "muted", text: t("dof.legend") }));
depthPane.append(
  el("div", { class: "group-title", text: t("dof.zones") }),
  el("label", { class: "toggle" }, t("dof.zonesToggle"), zoneToggle),
  el("p", { class: "muted", text: t("dof.zonesHint") }),
  zoneRows,
);

// ---- Curves by distance: near / middle / far, soft thirds of this photo's depth layers
let curveBand: DepthBand = "near";
let bandShown = false;
const bandChips = el("div", { class: "chips" });
/** Depth range of a band for the highlight view (open-ended at the ends). */
function bandRange(b: DepthBand): [number, number] {
  const [b1, b2] = params?.depthBands ?? [0.33, 0.66];
  return b === "near" ? [-1, b1] : b === "middle" ? [b1, b2] : [b2, 2];
}
function setBandShown(on: boolean) {
  bandShown = on;
  if (on && zoneHighlight !== undefined) { zoneHighlight = undefined; renderZones(); }
  send(on ? { type: "view", view: 5, range: bandRange(curveBand) } : { type: "view", view: currentView });
  renderBands();
}
const depthCurves = createToneCurves({
  histogram: (c) => histogramOf(histograms, curveBand, c),
  get: () => params?.depthCurves[curveBand],
  set: (c) => { if (params) params.depthCurves[curveBand] = c; },
  changed: () => { lookPanel.invalidate(); pushParams(); },
  enabled: () => !!params,
});
function renderBands() {
  const eye = el("button", { class: "chip eye" + (bandShown ? " on" : ""), text: "◉", title: t("dof.showBand"), "aria-label": t("dof.showBand") });
  eye.onclick = () => setBandShown(!bandShown);
  bandChips.replaceChildren(eye, ...DEPTH_BANDS.map((b, i) => {
    const z = dofInfo?.bands?.[i];
    const edited = params?.depthCurves[b] && !(["l", "r", "g", "b"] as const).every((k) => isFlat(params!.depthCurves[b]![k] ?? [{ x: 0, y: 0 }, { x: 1, y: 1 }]));
    const info = z ? ` ${Math.round(z.share * 100)}%${z.label ? " · " + tOr(`group.${z.label}`, z.label).toLowerCase() : ""}` : "";
    const c = el("button", { class: "chip" + (b === curveBand ? " on" : ""), text: t(`band.${b}`) + info + (edited ? " •" : "") });
    c.onclick = () => { curveBand = b; depthCurves.render(); if (bandShown) setBandShown(true); else renderBands(); };
    return c;
  }));
  depthCurves.render();
}
// Curves by distance are layers now (a Curves layer with a distance mask).
void bandChips;
renderBands();

// Check: the finished photo looked at for technical mistakes (src/analysis/check.ts).
let checkResult: { items: CheckItem[]; rgba: Uint8Array; w: number; h: number } | undefined;
/** The edit changed since the last check (the tab re-checks when opened). */
let checkStale = true;
let checkShown: CheckItem["id"] | undefined;
/** Fixes are still being worked out (they arrive one by one after the findings). */
let checkSolving = false;
const checkRun = el("button", { class: "btn small primary", text: t("check.run") });
const checkSummary = el("span", { class: "muted check-summary" });
const checkView = el("canvas", { class: "check-view", hidden: "" });
const checkList = el("div", { class: "check-list" });
checkPane.append(el("div", { class: "actions check-head" }, checkRun, checkSummary), checkView, checkList);
checkRun.onclick = () => runCheck();
/** The edit changed since: the check shown is of an older version (the tab re-checks, the badge hides). */
function markCheckStale() { checkStale = true; checkBadge.hidden = true; }
/** Bumped by every edit sent: a check result is fresh only if none happened while it ran. */
let paramsGen = 0, checkGen = 0;
function runCheck() {
  if (!params) { checkSummary.textContent = t("check.noPhoto"); return; }
  checkGen = paramsGen;
  checkRun.disabled = true;
  checkSummary.textContent = t("check.running");
  send({ type: "check" });
}
const LEVEL_ICON = { bad: "✗", warn: "!", ok: "✓" } as const;
function checkText(it: CheckItem): string {
  const v: Record<string, string | number> = { ...it.v };
  v.where = it.v.where ? t("check.where", { name: tOr(`group.${it.v.where}`, String(it.v.where)).toLowerCase() }) : "";
  if (it.id === "skin" && it.v.issue) v.issue = tOr(`check.issue.${it.v.issue}`, String(it.v.issue));
  if (it.id === "cast") v.tint = tOr(`check.tint.${it.v.tint}`, String(it.v.tint));
  if (it.id === "exposure") {
    v.dir = tOr(`check.dir.${it.v.dir}`, String(it.v.dir));
    const name = tOr(`check.scene.${it.v.scene}`, String(it.v.scene));
    v.scene = it.v.ev !== "" && it.v.ev !== undefined ? t("check.sceneEv", { name, ev: it.v.ev }) : name;
    if (it.v.basis === "camera") return tOr(`check.exposure.cam.${it.level}`, "", v);
  }
  if (it.id === "contrast" && it.v.issue) return tOr(`check.contrast.${it.v.issue}`, "", v);
  if (it.id === "saturation") {
    v.cam = it.v.camera !== "" && it.v.camera !== undefined ? t("check.camRef", { v: it.v.camera }) : "";
    return tOr(it.v.issue ? `check.saturation.${it.v.issue}.${it.level}` : "check.saturation.ok", "", v);
  }
  return tOr(`check.${it.id}.${it.level}`, "", v);
}
/** The controls a fix names: label and how the value reads on its slider. */
const FIX_LABEL: Record<string, [Parameters<typeof t>[0], (v: number) => string]> = {
  exposure: ["adj.exposure", (v) => `${v > 0 ? "+" : ""}${v.toFixed(2)}`],
  "tone.highlights": ["adj.highlights", pctFix], "tone.shadows": ["adj.shadows", pctFix], "tone.whites": ["adj.whites", pctFix],
  "tone.blacks": ["adj.blacks", pctFix], "tone.contrast": ["adj.contrast", pctFix],
  "wb.temp": ["adj.temp", (v) => `${Math.round(v)}K`], "wb.tint": ["adj.tint", (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}`],
  "local.compression": ["adj.range", pctFix], "local.clarity": ["adj.clarity", pctFix], "local.texture": ["adj.texture", pctFix],
  "dehaze.strength": ["adj.dehaze", pctFix], "color.vibrance": ["adj.vibrance", pctFix], "color.saturation": ["adj.saturation", pctFix],
  "denoise.luma": ["adj.noiseLuma", pctFix], "sharpen.amount": ["adj.sharpen", pctFix], "vignette.amount": ["adj.vigAmount", pctFix],
  autoCurves: ["adj.autoStrength", (v) => `${Math.round(v * 100)}%`],
};
function pctFix(v: number) { return `${v > 0 ? "+" : ""}${Math.round(v * 100)}`; }
/** A finding's fix: the exact change (worked out by the engine) with Apply, or the general advice. */
function checkFixEl(it: CheckItem): HTMLElement {
  const box = el("div", { class: "check-fix" });
  if (!it.fix?.length || !params) {
    box.textContent = checkSolving && it.err !== undefined ? t("check.solving") : it.id === "saturation" && it.v.issue === "pale" ? t("check.saturation.pale.fix") : t(`check.${it.id}.fix`);
    box.classList.toggle("solving", checkSolving && it.err !== undefined);
    return box;
  }
  const lines = it.fix.map((c) => {
    if ("layer" in c) {
      const l = params!.layers?.find((x) => x.id === c.layer);
      const name = l ? layerName(l) : c.name;
      return t("check.fixLayer", { name, from: `${Math.round(c.from * 100)}%`, to: c.to <= 0 ? t("check.hide") : `${Math.round(c.to * 100)}%` });
    }
    const [key, fmt] = FIX_LABEL[c.path] ?? [undefined, (v: number) => v.toFixed(2)];
    const ev = c.path === "exposure" ? ` (${c.to - c.from > 0 ? "+" : "−"}${Math.abs(c.to - c.from).toFixed(2)} EV)` : "";
    return `${key ? t(key) : c.path}: ${fmt(c.from)} → ${fmt(c.to)}${ev}`;
  });
  const apply = el("button", { class: "btn small", text: t("check.apply") });
  apply.onclick = (e) => {
    e.stopPropagation();
    if (!params) return;
    for (const c of it.fix!) {
      // Only over the value the fix was worked out from: a control moved since is left alone.
      if ("layer" in c) { const l = params.layers?.find((x) => x.id === c.layer); if (l && Math.abs(l.opacity - c.from) < 1e-6) { if (c.to <= 0) l.visible = false; else l.opacity = c.to; } }
      else if (Math.abs(getPath(params, c.path) - c.from) < 1e-6) setPath(params, c.path, c.to);
    }
    nextLabel = t("check.applied", { what: t(`check.t.${it.id}`) });
    syncControls();
    pushParams();
    setTimeout(runCheck, 80); // after the new settings reach the engine
  };
  box.append(el("span", { text: (it.fixPartial ? t("check.fixPartial") : t("check.fixSet")) + " " + lines.join(" · ") }), apply);
  return box;
}

function renderCheck() {
  checkRun.disabled = false;
  const r = checkResult;
  checkView.hidden = true;
  if (!r) { checkList.replaceChildren(); checkBadge.hidden = true; checkSummary.textContent = params ? "" : t("check.noPhoto"); return; }
  const count = (l: string) => r.items.filter((i) => i.level === l).length;
  checkSummary.textContent = t("check.summary", { bad: count("bad"), warn: count("warn"), ok: count("ok") });
  checkBadge.textContent = String(count("bad") + count("warn"));
  checkBadge.hidden = count("bad") + count("warn") === 0;
  checkBadge.classList.toggle("bad", count("bad") > 0);
  const order = { bad: 0, warn: 1, ok: 2 } as const;
  checkList.replaceChildren(...[...r.items].sort((a, b) => order[a.level] - order[b.level]).map((it) => {
    const row = el("div", { class: `check-item ${it.level}` + (checkShown === it.id ? " on" : "") + (it.mask ? " where" : "") },
      el("span", { class: "check-icon", text: LEVEL_ICON[it.level] }),
      el("div", { class: "check-body" },
        el("div", { class: "check-title", text: t(`check.t.${it.id}`) }),
        el("div", { class: "check-text", text: checkText(it) }),
        ...(it.level !== "ok" ? [checkFixEl(it)] : [])));
    if (it.mask) row.onclick = () => { checkShown = checkShown === it.id ? undefined : it.id; renderCheck(); };
    return row;
  }));
  const shown = r.items.find((i) => i.id === checkShown && i.mask);
  if (shown?.mask) {
    // The photo with the finding in red (50 %), everything else a little dimmed.
    checkView.width = r.w; checkView.height = r.h;
    const img = new ImageData(r.w, r.h);
    for (let i = 0, j = 0; i < r.w * r.h; i++, j += 4) {
      const on = shown.mask[i] === 1;
      img.data[j] = on ? (r.rgba[j] + 255) / 2 : r.rgba[j] * 0.7;
      img.data[j + 1] = on ? r.rgba[j + 1] * 0.4 : r.rgba[j + 1] * 0.7;
      img.data[j + 2] = on ? r.rgba[j + 2] * 0.4 : r.rgba[j + 2] * 0.7;
      img.data[j + 3] = 255;
    }
    checkView.getContext("2d")!.putImageData(img, 0, 0);
    checkView.hidden = false;
  }
}

// Export
const fmtSel = el("select", {}, el("option", { value: "jpeg", text: "JPEG" }), el("option", { value: "jpeg-hdr", text: t("exp.jpegHdr") }), el("option", { value: "heic", text: "HEIC" }), el("option", { value: "tiff16", text: t("exp.tiff") }), el("option", { value: "dng", text: t("exp.dng") }));
const spaceSel = el("select", {}, el("option", { value: "p3", text: "Display P3" }), el("option", { value: "srgb", text: "sRGB" }));
const qualitySl = el("input", { type: "range", min: "0.6", max: "1", step: "0.01", value: "0.92" });
const qualityOut = el("output", { text: "92" });
qualitySl.oninput = () => (qualityOut.textContent = String(Math.round(+qualitySl.value * 100)));
const exportBtn = el("button", { class: "btn primary", text: t("exp.button") });
const exportInfo = el("p", { class: "muted" });
/** The Export tab's button and the header's one are busy together. */
function setExportEnabled(on: boolean) { exportBtn.disabled = !on; exportTop.disabled = !on || !params; }
exportBtn.onclick = () => {
  if (!params || busy) return;
  busy = true;
  setExportEnabled(false);
  setProgress(t("progress.exporting"));
  // ?strip=N overrides the export strip height (memory vs speed; testing).
  const stripRows = Number(new URLSearchParams(location.search).get("strip")) || undefined;
  send({ type: "export", format: fmtSel.value as ExportFormat, quality: +qualitySl.value, space: spaceSel.value as "srgb" | "p3", stripRows });
};
const resSel = el("select", {}, el("option", { value: "auto", text: t("exp.resAuto") }), el("option", { value: "full", text: t("exp.resFull") }), el("option", { value: "half", text: t("exp.resHalf") }));
resSel.onchange = () => (resolution = resSel.value as typeof resolution);
const hdrHint = el("p", { class: "muted", text: t("exp.jpegHdrHint") });
hdrHint.hidden = true;
exportPane.append(
  el("div", { class: "row" }, el("label", { text: t("exp.format") }), fmtSel, el("span")),
  el("div", { class: "row" }, el("label", { text: t("exp.colour") }), spaceSel, el("span")),
  el("div", { class: "row" }, el("label", { text: t("exp.quality") }), qualitySl, qualityOut),
  hdrHint,
  el("div", { class: "actions" }, exportBtn),
  exportInfo,
  el("div", { class: "group-title", text: t("exp.next") }),
  resSel,
  el("p", { class: "muted", text: t("exp.hint") }),
);
fmtSel.onchange = () => {
  const f = fmtSel.value;
  qualitySl.disabled = !(f === "jpeg" || f === "jpeg-hdr" || f === "heic");
  hdrHint.hidden = f !== "jpeg-hdr";
  spaceSel.disabled = f === "tiff16" || f === "dng";
};

// Debug
const stageToggles = el("div");
const profTable = el("table", { class: "prof" });
const logPre = el("pre", { class: "log" });
const dlReport = el("button", { class: "btn small", text: t("dbg.download") });
// Auto feedback (src/ui/autoFeedback.ts): what was changed after the automatic development.
const dlFeedback = el("button", { class: "btn small" });
const clearFb = el("button", { class: "btn small ghost", text: t("fb.clear") });
function renderFeedback() { dlFeedback.textContent = t("fb.download", { n: String(feedbackCount()) }); }
dlFeedback.onclick = () => download(feedbackBlob(), `auto-feedback-${new Date().toISOString().slice(0, 10)}.json`);
clearFb.onclick = () => { clearFeedback(); renderFeedback(); };
renderFeedback();
debugPane.append(el("div", { class: "group-title", text: t("dbg.stages") }), stageToggles, el("div", { class: "group-title", text: t("dbg.profile") }), profTable, el("div", { class: "group-title", text: t("dbg.log") }), logPre, el("div", { class: "actions" }, dlReport),
  el("div", { class: "group-title", text: t("fb.title") }), el("p", { class: "muted", text: t("fb.tip") }), el("div", { class: "actions" }, dlFeedback, clearFb));
const STAGES: Array<[keyof Params["enable"], string]> = [
  ["denoise", t("dbg.denoise")], ["wb", t("dbg.wb")], ["exposure", t("dbg.exposure")], ["localTone", t("dbg.localTone")],
  ["curves", t("dbg.curves")], ["lut", t("dbg.lut")], ["semantic", t("dbg.semantic")], ["dehaze", t("dbg.dehaze")], ["sharpen", t("dbg.sharpen")],
];
function renderStageToggles() {
  stageToggles.replaceChildren(...STAGES.map(([k, name]) => {
    const cb = el("input", { type: "checkbox" });
    cb.checked = !!params?.enable[k];
    cb.onchange = () => { if (!params) return; params.enable[k] = cb.checked; pushParams(); };
    return el("label", { class: "toggle" }, name, cb);
  }));
}
dlReport.onclick = () => {
  const blob = new Blob([JSON.stringify({ summary, decisions, auto: autoParams, params, profile, log: logLines }, null, 2)], { type: "application/json;charset=utf-8" });
  download(blob, (summary?.file ?? "photo").replace(/\.[^.]+$/, "") + "-analysis.json");
};

function renderProfile() {
  profTable.replaceChildren(
    el("tr", {}, el("th", { text: "stage" }), el("th", { text: "ms" }), el("th", { text: "GPU MB" }), el("th", { text: "peak" })),
    ...profile.flatMap((p) => {
      const rows = [el("tr", {}, el("td", { text: p.stage }), el("td", { text: String(p.ms) }), el("td", { text: String(p.gpuLiveMB) }), el("td", { text: String(p.gpuPeakMB) }))];
      if (p.note) rows.push(el("tr", {}, el("td", { class: "note", colspan: "4", text: p.note })));
      return rows;
    }),
  );
}

/** One line on the automatic 2× stage: what happened, and one short reason. */
function upscaleText(u: UpscaleInfo): string {
  const reason = tOr(`up.r.${u.code}`, u.upscaleReason, u.vars);
  switch (u.state) {
    case "applied": return `${t("up.applied")} — ${reason}` + (u.width ? ` (${t("up.size", { w: u.width, h: u.height ?? 0 })})` : "");
    case "running": return `${t("up.running")} ${reason}`;
    case "pending": return t("up.pending");
    case "failed": return t("up.failed");
    case "cancelled": return t("up.cancelled");
    default:
      if (u.code === "off") return t("up.offSkip");
      if (u.code === "forced") return t("up.pending");
      return ["sufficient", "sharp-12", "adequate"].includes(u.code) ? `${t("up.skipSufficient")} (${reason})` : `${t("up.skipped")} — ${reason}`;
  }
}

// --------------------------------------------------------------------------- upscale tab
const upMode = el("select", {},
  el("option", { value: "auto", text: t("upt.modeAuto") }),
  el("option", { value: "always", text: t("upt.modeAlways") }),
  el("option", { value: "off", text: t("upt.modeOff") }));
upMode.value = upscaleMode;
upMode.onchange = () => {
  upscaleMode = upMode.value as UpscaleMode;
  try { localStorage.setItem("upscaleMode", upscaleMode); } catch { /* private mode */ }
  renderUpscale();
};
const upStatus = el("p", { class: "up-status" });
const upFacts = el("dl", { class: "kv" });
const upNow = el("button", { class: "btn primary", text: t("upt.runNow") });
upNow.onclick = () => { if (upscale) { markInflight(); send({ type: "upscale-now" }); } };
const upRevert = el("button", { class: "btn", text: t("upt.revert") });
upRevert.onclick = () => { if (currentFile && params) openFile(currentFile, structuredClone(params), "off"); };
const upModeNote = el("p", { class: "muted" });
upscalePane.append(
  upStatus,
  el("div", { class: "actions" }, upNow, upRevert),
  upFacts,
  el("div", { class: "group-title", text: t("upt.mode") }),
  el("div", { class: "row" }, el("label", { text: t("upt.forNext") }), upMode, el("span")),
  upModeNote,
  el("p", { class: "muted", text: t("upt.about") }),
);

function renderUpscale() {
  const u = upscale;
  upStatus.textContent = u ? upscaleText(u) : currentFile ? t("up.pending") : t("upt.none");
  upModeNote.textContent = t(upscaleMode === "auto" ? "upt.noteAuto" : upscaleMode === "always" ? "upt.noteAlways" : "upt.noteOff");
  const idle = !!u && (u.state === "skipped" || u.state === "failed" || u.state === "cancelled");
  upNow.hidden = !idle || u!.code === "memory";
  upRevert.hidden = u?.state !== "applied";
  upFacts.replaceChildren();
  if (!u) return;
  const r = u.report, m = r.metrics;
  const add = (k: string, v: string) => upFacts.append(el("dt", { text: k }), el("dd", { text: v }));
  add(t("upt.source"), `${r.width}×${r.height} · ${r.megapixels.toFixed(1)} MP`);
  if (u.state === "applied" && u.width) add(t("upt.result"), `${u.width}×${u.height} · ${((u.width * (u.height ?? 0)) / 1e6).toFixed(1)} MP`);
  add(t("upt.sharpness"), `${Math.round(r.sharpnessScore * 100)}% · ${t("upt.edge", { px: Number.isFinite(m.edgeSigma) ? m.edgeSigma.toFixed(2) : "—" })}`);
  add(t("upt.noise"), `${(m.noiseSigma * 255).toFixed(2)} / 255`);
  add(t("upt.detail"), `${(m.detailDensity * 100).toFixed(1)}%`);
  add(t("upt.blur"), r.severeBlur ? t("upt.yes") : t("upt.no"));
}

/** "WebGPU · fp16 · 11 threads": what this device runs on (Info). */
let capsText = "";
function renderAuto() {
  const s = summary;
  const kv = el("dl", { class: "kv" });
  const add = (k: string, v: string | number) => kv.append(el("dt", { text: k }), el("dd", { text: String(v) }));
  if (capsText) add(t("auto.engine"), capsText);
  if (s) {
    add(t("auto.source"), s.source);
    add(t("auto.size"), `${s.width}×${s.height}` + (s.working.factor > 1 ? ` (${t("auto.working", { w: s.working.width, h: s.working.height })})` : ""));
    for (const [k, v] of Object.entries(s.meta)) add(k, v);
    add(t("auto.regions"), Object.entries(s.coverage).filter(([, v]) => v >= 1).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${tOr(`group.${k}`, k).toLowerCase()} ${v}%`).join(", "));
    if (upscale) add(t("auto.detail"), upscaleText(upscale));
  }
  const list = el("div", { class: "decisions" }, ...decisions.map((d) => {
    const v = Array.isArray(d.value) ? d.value.join(" / ") : String(d.value);
    const card = el("div", { class: "decision" },
      el("div", { class: "h" }, el("span", { text: d.id }), el("span", { text: v.length > 60 ? "…" : v })),
      el("div", { class: "r", text: d.reason }));
    if (Object.keys(d.inputs).length) card.append(el("details", {}, el("summary", { text: t("auto.inputs") }), el("code", { text: JSON.stringify(d.inputs) })));
    return card;
  }));
  autoPane.replaceChildren(kv, el("div", { class: "group-title", text: t("auto.decisions") }), list);
}

function syncControls() {
  sliders.forEach(refreshSlider);
  layersPanel.render();
  renderLooks();
  renderStageToggles();
  if (params) { dofToggle.checked = params.enable.dof; protectToggle.checked = params.protectHighlights !== false; imgToggle.checked = params.render?.engine === "img"; }
  renderRings();
  renderZones();
  renderBands();
  renderAutoCurves();
}

function download(blob: Blob, name: string) {
  const a = el("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 10000);
}

async function share(blob: Blob, name: string) {
  // iOS: the share sheet lets the user save to Photos; fall back to a download.
  const file = new File([blob], name, { type: blob.type });
  const nav = navigator as Navigator & { canShare?: (d: ShareData) => boolean };
  if (nav.canShare?.({ files: [file] }) && /iPhone|iPad/.test(navigator.userAgent)) {
    try { await navigator.share({ files: [file] }); return; } catch { /* cancelled */ }
  }
  download(blob, name);
}

/** Persistent, visible error notice (GPU problems would otherwise just render black). */
const errorBox = el("div", { class: "error-box" });
errorBox.hidden = true;
stage.append(errorBox);
function showError(text: string) {
  // No WebGPU at all: not a failure of this photo — say what to do, in the UI language.
  const noGpu = text.startsWith("WebGPU is not available");
  if (noGpu) text = t("err.noWebgpuHow");
  errorBox.hidden = false;
  errorBox.replaceChildren(
    el("strong", { text: noGpu ? t("err.noWebgpu") : t("err.gpu") }),
    el("div", { text: text.slice(0, 600) }),
    el("div", { class: "muted", text: t("err.seeLog") }),
  );
  const close = el("button", { class: "btn small", text: t("err.dismiss") });
  close.onclick = () => (errorBox.hidden = true);
  errorBox.append(close);
}

// --------------------------------------------------------------------------- worker messages
/** Listeners of every engine message (the local autotest only). */
const engineListeners = new Set<(m: FromWorker) => void>();
worker.onmessage = (ev: MessageEvent<FromWorker>) => {
  const m = ev.data;
  for (const f of engineListeners) f(m);
  switch (m.type) {
    case "display":
      if (!m.ok) { logLines.push(`GPU display unavailable (${m.message ?? "?"}): previews drawn by the page`); takeCanvasBack(); }
      else logLines.push("GPU display: previews drawn by the worker");
      break;
    case "ready":
      caps = m.caps;
      looks = m.looks;
      lookPanel.onLuts(looks);
      capsEl.textContent = ""; // "Starting…" done; the header keeps only errors
      handCanvasToWorker();
      // Shown in Info.
      capsText = `${caps.backend === "webgpu" ? "WebGPU" : "WASM"}${caps.f16 ? " · fp16" : ""}${caps.crossOriginIsolated ? ` · ${t("app.threads", { n: caps.threads })}` : ""}`;
      renderAuto();
      if (!caps.heicEncode) (fmtSel.querySelector('option[value="heic"]') as HTMLOptionElement).disabled = true;
      renderLooks();
      // After Safari evicted this tab (memory) or the GPU was reset: reopen where we were —
      // unless that very photo was being processed when the page died (would loop forever).
      // A crash during analysis steps this device down whether or not the photo can be
      // offered again (private mode has no stored photo; one may be opened meanwhile).
      void crashInfo().then((c) => {
        if (c?.analysis) { flag(() => localStorage, SAFE_ANALYSIS, true); stepDownAnalysis(); }
      });
      void restorablePhoto().then((r) => {
        if (crashedWhileProcessing()) consumeCrash();
        if (!r || params) return;
        if (crashedWhileProcessing()) {
          offerSafeReopen(r.file, r.params);
          return;
        }
        logLines.push(`restoring ${r.file.name} after a reload of this tab`);
        openFile(r.file, r.params);
      });
      break;
    case "upscale":
      upscale = m.info;
      // While the 2× stage runs the tab holds its largest buffers: keep the crash
      // guard armed so a memory kill leads to the safe reopen, not a loop.
      if (m.info.state === "running") markInflight();
      else if (!busy) markCompleted();
      if (m.info.state !== "running" && m.info.state !== "pending" && !busy) setProgress(undefined);
      renderAuto();
      renderUpscale();
      break;
    case "gpu-lost":
      // A lost GPU device cannot be revived in place; restart the page — the session restores itself.
      setProgress(t("progress.gpuReset"));
      setTimeout(() => location.reload(), 600);
      break;
    case "progress":
      setProgress(stageText(m.stage, m.detail), m.frac);
      noteStage(stageText(m.stage, m.detail));
      // Only a crash *inside* segmentation or depth marks the device for CPU analysis.
      noteAnalysisStage(m.stage);
      break;
    case "preview":
      if (m.final) finalPreviews++;
      drawPreview(m);
      if (m.final && !holding) setProgress(undefined);
      if (m.final) markCompleted();
      if (m.final && pickAwaitsPreview) pickDone();
      if (m.final) exportTop.disabled = exportBtn.disabled || !params;
      void 0; // look thumbnails: the Look panel is hidden for now
      busy = false;
      break;
    case "analysis":
      keepRuntimeWhenIdle(); // the analysis used ONNX Runtime: keep it for offline use
      checkResult = undefined; checkStale = true; renderCheck();
      noteAnalysisStage();
      opening = false;
      restored = false;
      summary = m.summary; decisions = m.decisions; autoParams = m.auto; params = m.params; dofInfo = m.dof; cameraWB = m.cameraWB; toneEqHist = undefined;
      autoCurveBands = m.autoCurves;
      exportTop.disabled = busy;
      cellCov = m.cellCoverage;
      histograms = undefined; // the previous photo's; the first final preview brings new ones
      exposureSuggestion = m.exposureSuggestion;
      aeNote.textContent = exposureSuggestion ? t("adj.suggests", { ev: `${exposureSuggestion > 0 ? "+" : ""}${exposureSuggestion.toFixed(2)}` }) : t("adj.noCorrection");
      // The reason itself comes from the decision engine and stays in English.
      noDepthBox.hidden = !m.noDepth;
      noDepthText.textContent = m.noDepth ?? "";
      dofReason.textContent = t("dof.reason", { d: m.dof.focus.toFixed(2) }) + (m.dof.justified ? t("dof.suggested") : t("dof.notSuggested")) + m.dof.reason;
      syncControls();
      renderAuto();
      llmPanel.reset();
      lookPanel.invalidate();
      regionsPanel.render();
      if (pendingRestore) {
        // Edits saved before layers existed: their curves and region colour would now be
        // ignored; the automatic layers of today stand in for them.
        if (!pendingRestore.layers) {
          for (const k of ["curves", "regionCurves", "depthCurves", "cellCurves", "cells", "semantic", "skin", "distance"] as const) delete (pendingRestore as Partial<Params>)[k];
        }
        // Automatic layers saved by an older conversion: today's stand in for them (user layers stay).
        else if ((pendingRestore.autoLayersVersion ?? 1) < AUTO_LAYERS_VERSION) {
          // Today's automatic layers in today's order (ones the user deleted stay deleted), the user's own above.
          const kept = new Set(pendingRestore.layers.filter((l) => l.auto).map((l) => l.auto));
          pendingRestore.layers = [...params!.layers.filter((f) => f.auto && kept.has(f.auto)), ...pendingRestore.layers.filter((l) => !l.auto)];
          pendingRestore.autoLayersVersion = AUTO_LAYERS_VERSION;
        }
        // Edits saved before the img rendering existed keep the rendering they were made with.
        if (!pendingRestore.render) pendingRestore.render = { engine: "classic", purity: 0, strength: 1 };
        // Same photo, same analysis: bring back the edits made before the reload.
        params = { ...params!, ...pendingRestore, enable: { ...params!.enable, ...pendingRestore.enable } };
        params.profile = normalizeProfile(params.profile);
        // Zones are recomputed for this photo; keep the user's blur values only if they fit.
        if (!params.dof.zones || params.dof.zones.length !== 5) params.dof = { ...params.dof, zones: autoParams!.dof.zones, mode: params.dof.mode ?? "focus" };
        params.dof.zoneBounds = autoParams!.dof.zoneBounds;
        // Regions saved by older versions lack newer fields: fill from today's automatic values.
        for (const g of Object.keys(params.semantic) as Array<keyof Params["semantic"]>) params.semantic[g] = { ...autoParams!.semantic[g], ...params.semantic[g] };
        pendingRestore = undefined;
        restored = true;
        syncControls();
        pushParams();
      }
      history.reset(params!, t("hist.open"));
      updateUndo();
      break;
    case "histograms":
      histograms = m.data;
      layersPanel.refreshHistogram();
      break;
    case "blackPointMatched": {
      // The automatic "Black point" layer: just above the photo's own tone curve.
      // Not over a restored session (the user's layers, perhaps without it on purpose),
      // nor from the previous photo after another was opened.
      if (restored || opening) break;
      markCheckStale();
      const flat = () => [{ x: 0, y: 0 }, { x: 1, y: 1 }];
      for (const p of [params, autoParams]) {
        if (!p) continue;
        const layer = makeLayer("curves", "Black point", { auto: "curves.black", params: { l: m.points, r: flat(), g: flat(), b: flat() } });
        const i = p.layers.findIndex((l) => l.auto === "curves.black");
        if (i >= 0) p.layers[i] = { ...layer, id: p.layers[i].id };
        else p.layers.splice(p.layers.findIndex((l) => l.auto === "curves.photo") + 1, 0, layer);
      }
      if (params) {
        if (!history.canUndo) history.reset(params, t("hist.open"));
        send({ type: "params", params: structuredClone(params), draft: false });
        rememberParams(params);
      }
      logLines.push(m.note);
      syncControls();
      break;
    }
    case "autoAdjusted": {
      // Part of opening the photo: the automatic values change, and the current ones where not edited meanwhile.
      if (opening) break;
      markCheckStale();
      let changed = false, differs = false;
      for (const [k, v] of Object.entries(m.changes)) {
        if (autoParams) setPath(autoParams, k, v);
        const take = !!params && getPath(params, k) === m.from[k];
        if (take && v !== m.from[k]) { setPath(params!, k, v); changed = true; }
        if (take !== m.applied.includes(k)) differs = true;
      }
      if (params && changed) {
        if (!history.canUndo) history.reset(params, t("hist.open"));
        rememberParams(params);
        syncControls();
      }
      // An edit of ours was still on its way to the engine when it decided: ours is what counts.
      if (params && differs) send({ type: "params", params: structuredClone(params), draft: false });
      logLines.push(m.note);
      break;
    }
    case "colorCalibrated": {
      // Part of opening the photo, like the exposure calibration.
      if (opening) break;
      markCheckStale();
      if (autoParams) { autoParams.color.saturation = m.saturation; autoParams.color.vibrance = m.vibrance; }
      if (params && params.color.saturation === m.from[0] && params.color.vibrance === m.from[1]) {
        params.color.saturation = m.saturation; params.color.vibrance = m.vibrance;
        if (!history.canUndo) history.reset(params, t("hist.open"));
        rememberParams(params);
        syncControls();
      }
      logLines.push(m.note);
      break;
    }
    case "exposureCalibrated": {
      // Part of opening the photo, not an edit: the history's first step takes it too.
      if (opening) break; // the previous photo's, arriving after another was opened
      markCheckStale();
      const before = autoParams?.exposure;
      exposureSuggestion = m.exposure;
      if (autoParams) autoParams.exposure = m.exposure;
      // Not over an exposure the user set meanwhile.
      if (params && params.exposure === before) {
        params.exposure = m.exposure;
        if (!history.canUndo) history.reset(params, t("hist.open"));
        rememberParams(params);
      }
      aeNote.textContent = t("adj.suggests", { ev: `${m.exposure > 0 ? "+" : ""}${m.exposure.toFixed(2)}` });
      logLines.push(m.note);
      syncControls();
      break;
    }
    case "params":
      markCheckStale();
      params = m.params;
      pendingAuto = undefined;
      syncControls();
      scheduleCommit(t("hist.develop")); // focus points placed on the photo are a step too
      break;
    case "log":
      logLines.push(m.text);
      logPre.textContent = logLines.join("\n");
      if (m.text.startsWith("GPU error")) showError(m.text);
      break;
    case "profile":
      profile = m.stages;
      { const l = m.stages[m.stages.length - 1]; if (l) noteStage(`after “${l.stage}” (GPU ${l.gpuLiveMB} MB, peak ${l.gpuPeakMB} MB)`); }
      renderProfile();
      break;
    case "exported":
      busy = false;
      setExportEnabled(true);
      setProgress(undefined);
      exportInfo.textContent = t("exp.done", { file: m.name, mb: (m.blob.size / 1e6).toFixed(1), s: (m.ms / 1000).toFixed(1) });
      recordEdit("export", summary, autoParams, params); renderFeedback();
      void share(m.blob, m.name);
      break;
    case "looks":
      looks = m.looks;
      lookPanel.onLuts(looks);
      break;
    case "thumbs":
      lookPanel.onThumbs(m.items);
      break;
    case "palette":
      lookPanel.onPalette(m.stats);

      break;
    case "check":
      checkResult = m;
      checkStale = paramsGen !== checkGen; // edited while it ran: shown, but re-checked next time
      queueMicrotask(() => { if (checkStale) checkBadge.hidden = true; }); // its counts are of an older version
      checkShown = undefined;
      checkSolving = true;
      renderCheck();
      break;
    case "checkFix":
      if (m.done) checkSolving = false;
      else if (checkResult && m.id) { const it = checkResult.items.find((i) => i.id === m.id); if (it) { it.fix = m.fix; it.fixPartial = m.partial; } }
      renderCheck();
      break;
    case "pick":
      // A changed mask is "done" when the photo shows it (a first selection is worked
      // out while rendering: seconds on a phone); otherwise now.
      if (m.info && layersPanel.onPick(m.info, pickFor)) pickAwaitsPreview = true;
      else pickDone();
      break;
    case "lookProfile":
      lookPanel.onProfile(m.profile, m.reference, m.message);
      break;
    case "error":
      if (pickBusy) pickDone();
      checkRun.disabled = false;
      noteAnalysisStage();
      opening = false;
      showError(m.message);
      busy = false;
      setExportEnabled(true);
      setProgress(undefined);
      logLines.push("ERROR: " + m.message);
      logPre.textContent = logLines.join("\n");
      capsEl.innerHTML = "";
      {
        const shown = m.message.startsWith("WebGPU is not available") ? t("err.noWebgpuHow") : m.message;
        capsEl.append(el("span", { class: "error", text: m.message.startsWith("WebGPU is not available") ? t("err.noWebgpu") : shown.slice(0, 140) }));
        if (!params) { empty.style.display = ""; canvas.style.display = "none"; (empty.querySelector("p") as HTMLElement).textContent = shown; }
      }
      break;
  }
};
// The worker itself failed: say so and let the page be used again (no endless progress).
worker.onerror = (e) => {
  capsEl.textContent = t("err.worker", { msg: e.message });
  noteAnalysisStage();
  opening = false;
  busy = false;
  setProgress(undefined);
  setExportEnabled(true);
  markCompleted();
};

const long = Math.max(window.innerWidth, window.innerHeight) * Math.min(2, window.devicePixelRatio || 1);
send({ type: "preview-size", long });
send({ type: "init", base: import.meta.env.BASE_URL, phone: autotestPhone || undefined });
fmtSel.onchange?.(new Event("change"));

// Local autotest (scripts/memcheck.mjs): never on the site.
if (autotestAllowed()) {
  // Reported before anything else, so a page that never gets ready still says why.
  void (async () => {
    const adapter = await (navigator as Navigator & { gpu?: GPU }).gpu?.requestAdapter().catch(() => null);
    void fetch("/__debug/report", { method: "POST", body: JSON.stringify({ t: 0, stage: "boot", gpu: !!(navigator as Navigator & { gpu?: GPU }).gpu, adapter: !!adapter, isolated: crossOriginIsolated, ua: navigator.userAgent }) }).catch(() => undefined);
  })();
  engineListeners.add((m) => { if (m.type === "error" && !caps) void fetch("/__debug/report", { method: "POST", body: JSON.stringify({ stage: "failed", message: `engine did not start: ${m.message}` }) }).catch(() => undefined); });
  void import("./autotest.ts").then(({ runAutotest }) => {
    const start = () => runAutotest({
      openFile: (f) => openFile(f), params: () => params, pushParams, send,
      zero: () => zeroBtn.click(),
      on: (fn) => { engineListeners.add(fn); return () => engineListeners.delete(fn); },
    });
    // After the engine is ready (the first "ready" message).
    if (caps) void start(); else { const f = (m: FromWorker) => { if (m.type === "ready") { engineListeners.delete(f); setTimeout(() => void start(), 500); } }; engineListeners.add(f); }
  });
}
