/**
 * The processing engine (runs inside a Web Worker). Wires the independent
 * modules together in the documented order and owns every GPU resource:
 *
 *   Input/Decoder → RAW Development → Scene Analysis (reduced image) →
 *   Semantic Segmentation → Depth Estimation → Mask/Depth Refinement →
 *   Image statistics → Automatic Decision Engine → [preview] →
 *   Denoise (GPU, noise-adaptive) →
 *   Image quality analysis → optional 2× upscale (Swin2SR, only when needed) →
 *   Exposure/Tone → Camera Color → Semantic/Depth → Depth of Field → Output
 *
 * Deviation from the reference order (documented in docs/PIPELINE.md): the
 * denoise and the tiled restoration run after the first preview. The analysis
 * networks see a ≤768 px area-averaged image in which sensor noise is already
 * averaged away, so running them on the denoised image would not change
 * their output — but it would delay the first preview.
 *
 * The class holds the state and the render loop; each area lives beside it as
 * functions of the engine (`fn(eng, …)`), reached through one-line methods here:
 *
 *   session.ts      the photo's state (Session) and helpers shared by the parts
 *   open.ts         opening: decode, develop, analysis, decisions, first preview
 *   calibration.ts  the automatic development matched to the camera's rendering
 *   upscale.ts      restoration after the first preview: denoise, 2× upscale
 *   export.ts       full-resolution export in strips
 *   selection.ts    taps on the photo, tap-to-select masks
 *   retouch.ts      the magic brush's fills and their undo
 *   check.ts        the Check and its fixes
 *   queries.ts      what the page asks about the photo (depth, focus, zones…)
 *   looks.ts        look thumbnails, looks from a reference photo
 */
import { Gpu } from "../gpu/gpu.ts";
import { Neural } from "../neural/ort.ts";
import type { AnalysisLevel } from "../neural/scene.ts";
import type { ImageQualityReport, UpscaleMode } from "../analysis/quality.ts";
import { downsample, releaseRefined } from "../refine/refine.ts";
import { filmOf } from "../film/film.ts";
import { HI_QS, chromaStats, renderedQuantiles } from "../decode/preview.ts";
import { previewHistograms } from "../analysis/previewHist.ts";
import type { Params, RetouchStroke } from "../decision/params.ts";
import { Renderer, type RenderSource } from "../render/renderer.ts";
import { wbMatrix } from "../color/wb.ts";
import { neutralProfile, normalizeProfile, type LookProfile } from "../looks/profile.ts";
import type { ColorStats } from "../looks/palette.ts";
import { canEncodeHeic } from "../output/encoders.ts";
import { Profiler } from "./profiler.ts";
import type { Capabilities, ExportFormat, PickInfo, Summary } from "./protocol.ts";
import type { CameraColor } from "../color/dng.ts";
import type { Rect } from "../retouch/geometry.ts";
import { selectionMask } from "../refine/selection.ts";
import type { MaskShape } from "../layers/model.ts";
import type { MaskHist, ToneEqDetail } from "../tone/toneEq.ts";
import type { CheckItem, CheckInput } from "../analysis/check.ts";
import type { FixChange } from "../analysis/checkFix.ts";
import { type Post, type Session, type Selections, isMobile } from "./session.ts";
import { calibrateToCamera } from "./calibration.ts";
import { dropMotionPlate, ensureMotionPlate } from "./objectMotion.ts";
import * as openMod from "./open.ts";
import * as upscaleMod from "./upscale.ts";
import * as exportMod from "./export.ts";
import * as selectionMod from "./selection.ts";
import * as retouchMod from "./retouch.ts";
import * as checkMod from "./check.ts";
import * as queriesMod from "./queries.ts";
import * as looksMod from "./looks.ts";

export class Engine {
  gpu!: Gpu;
  neural!: Neural;
  renderer!: Renderer;
  post: Post;
  s?: Session;
  /** An explicit depth range to highlight in view 5 (a distance band). */
  viewRange?: [number, number];
  previewLong = isMobile() ? 1600 : 2048;
  view: 0 | 1 | 2 | 4 | 5 | 6 | 9 | 11 = 0;
  region = 0;
  before = false;
  generation = 0;
  profiler = new Profiler();

  constructor(post: Post) { this.post = post; }

  /**
   * One serial queue for all GPU work on the session. Opening a photo,
   * restoration, export, previews and look thumbnails create and destroy
   * textures; running any two at once let a render submit a texture another
   * job had just destroyed ("Destroyed texture used in a submit").
   */
  chain: Promise<unknown> = Promise.resolve();
  exclusive<T>(f: () => Promise<T>): Promise<T> {
    const r = this.chain.then(f, f);
    this.chain = r.catch(() => {});
    return r;
  }
  renderQueued = false;
  queuedDraft = false;
  queuedFinal = false;
  /** Schedules a preview render; bursts of slider changes coalesce into one
   * render with the latest parameters (a final request overrides a draft). */
  requestRender(final = true, draft = false) {
    if (!this.s) return;
    this.queuedDraft = draft;
    this.queuedFinal ||= final; // a release arriving while a draft waits makes it final
    if (this.renderQueued) return;
    this.renderQueued = true;
    void this.exclusive(async () => {
      this.renderQueued = false;
      const fin = this.queuedFinal;
      this.queuedFinal = false;
      await this.renderNow(fin, this.queuedDraft);
    }).catch((e) => this.post({ type: "error", message: e instanceof Error ? e.message : String(e) }));
  }

  /** Depth range of the highlighted zone (view 5; `region` holds the zone index, or `viewRange` an explicit range). */
  zoneRange(): [number, number] | undefined {
    if ((this.view === 5 || this.view === 4) && this.viewRange) return this.viewRange; // view 4: a region at a distance
    const e = this.s?.decision.dofSuggestion.zoneEdges;
    if (this.view !== 5 || !e) return undefined;
    const i = Math.min(4, Math.max(0, this.region));
    return [i === 0 ? -1 : e[i], i === 4 ? 2 : e[i + 1]];
  }

  /** The page's canvas, when previews are drawn straight into it on the GPU. */
  display?: { canvas: OffscreenCanvas; ctx: GPUCanvasContext };
  setCanvas(canvas: OffscreenCanvas) {
    try {
      if (!this.gpu) throw new Error("no GPU");
      const ctx = canvas.getContext("webgpu") as GPUCanvasContext | null;
      if (!ctx) throw new Error("no WebGPU canvas context");
      // The preview is display-encoded Display P3 in rgba8unorm: copied as is.
      ctx.configure({ device: this.gpu.device, format: "rgba8unorm", colorSpace: "display-p3", alphaMode: "opaque", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
      this.display = { canvas, ctx };
      this.post({ type: "display", ok: true });
    } catch (e) {
      this.display = undefined;
      this.post({ type: "display", ok: false, message: e instanceof Error ? e.message : String(e) });
    }
  }

  draftIdle = 0;
  /** The draft copy is freed a few seconds after the last drag (it comes back on the next one). */
  scheduleDraftRelease(s: Session) {
    clearTimeout(this.draftIdle);
    this.draftIdle = setTimeout(() => void this.exclusive(async () => {
      if (this.s !== s || !s.draft) return;
      this.gpu.release(s.draft.base, s.draft.denoised === s.draft.base ? undefined : s.draft.denoised);
      s.draft = undefined;
    }), 8000) as unknown as number;
  }

  /** Half-size (quarter-pixel) copy of the preview proxy, created on first drag. */
  async draftSource(): Promise<RenderSource> {
    const s = this.s!;
    const px = s.proxy!;
    if (!s.draft) {
      const w = Math.max(1, Math.round(px.w / 2)), h = Math.max(1, Math.round(px.h / 2));
      const base = await downsample(this.gpu, px.base, px.w, px.h, w, h, false, 1, "draft.base");
      const dn = px.denoised === px.base ? base : await downsample(this.gpu, px.denoised, px.w, px.h, w, h, false, 1, "draft.denoised");
      s.draft = { base, denoised: dn, w, h };
    }
    const d = s.draft;
    return { base: d.base, denoised: d.denoised, width: d.w, height: d.h, fullWidth: s.work.width };
  }

  /** Why the engine could not start (kept, so every later request reports the real reason). */
  initError?: string;
  get ready(): boolean { return !!this.gpu; }

  /** Where the app's files are served from (models, ORT runtime): for the analysis worker too. */
  base = "";
  async init(base: string, forceCpu = false): Promise<Capabilities> {
    let gpu: Gpu | undefined;
    try { gpu = forceCpu ? undefined : await Gpu.create(); }
    catch (e) { this.initError = e instanceof Error ? e.message : String(e); throw e; }
    if (!gpu) {
      this.initError = "WebGPU is not available in this browser. On iPhone, use Safari on iOS 26 or later (Settings → Apps → Safari → Advanced → Feature Flags → WebGPU on older versions).";
      throw new Error(this.initError);
    }
    this.gpu = gpu;
    if (isMobile()) { gpu.stagingLimitMB = 16; this.previewLong = 1600; }
    gpu.onError = (m) => this.log("GPU error: " + m);
    gpu.onLost = (m) => this.post({ type: "gpu-lost", reason: m });
    this.renderer = new Renderer(gpu);
    this.base = base;
    this.neural = await Neural.create(gpu, base);
    this.neural.onProgress = (id, loaded, total) => this.post({ type: "progress", stage: `download ${id}`, frac: loaded / total, detail: `${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB` });
    const heic = await canEncodeHeic();
    return {
      webgpu: true,
      f16: gpu.info.f16,
      backend: this.neural.backend,
      gpu: `${gpu.info.vendor} ${gpu.info.architecture}`.trim(),
      heicEncode: heic,
      crossOriginIsolated: (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true,
      threads: globalThis.navigator?.hardwareConcurrency ?? 1,
    };
  }

  looks() { return this.renderer.looks().map((l) => ({ id: l.id, name: l.name, description: l.description })); }

  log(text: string) { this.post({ type: "log", text }); }
  progress(stage: string, detail?: string, frac?: number) { this.post({ type: "progress", stage, detail, frac }); }

  closeSession() {
    const s = this.s;
    if (!s) return;
    const g = this.gpu;
    if (s.denoised !== s.work.tex) g.release(s.denoised);
    g.release(s.work.tex, s.skin);
    if (s.sel) { s.sel.sam.dispose(); g.release(s.sel.tex); this.renderer.selection = undefined; }
    s.retouch?.painter?.dispose();
    dropMotionPlate(this, s);
    this.releaseProxy(s);
    releaseRefined(g, s.maps);
    this.renderer.releaseTargets();
    this.dropThumb();
    try { s.decoded.close(); } catch { /* already closed */ }
    this.s = undefined;
  }

  /**
   * Opens a photo. Everything an open allocates is registered; if it does not get
   * as far as becoming the session (a newer open replaced it, or it failed), all of
   * it is freed here — a stopped 48 MP open otherwise kept ≈ 600 MB (the decoder
   * worker, the working image, masks) and the next attempt started that far behind.
   */
  async open(file: File, resolution: "auto" | "full" | "half", autoExposure = false, autoDof = false, upscaleMode: UpscaleMode = "auto", safeAnalysis = false, level?: AnalysisLevel) {
    const cleanup: Array<() => void> = [];
    let committed = false;
    try {
      await this.openInner(file, resolution, autoExposure, autoDof, upscaleMode, safeAnalysis, level, (f) => cleanup.push(f), () => { committed = true; });
    } finally {
      if (!committed) for (const f of cleanup.reverse()) { try { f(); } catch { /* already freed */ } }
    }
  }

  openInner(file: File, resolution: "auto" | "full" | "half", autoExposure: boolean, autoDof: boolean, upscaleMode: UpscaleMode, safeAnalysis: boolean, level: AnalysisLevel | undefined,
    track: (free: () => void) => void, commit: () => void) { return openMod.openInner(this, file, resolution, autoExposure, autoDof, upscaleMode, safeAnalysis, level, track, commit); }

  analyseQuality(reducedByUser: boolean, mode: UpscaleMode) : Promise<ImageQualityReport> { return openMod.analyseQuality(this, reducedByUser, mode); }

  forceUpscale() { return upscaleMod.forceUpscale(this); }

  runUpscale(gen: number) { return upscaleMod.runUpscale(this, gen); }

  restore(_gen: number) { return upscaleMod.restore(this, _gen); }

  protectGroupAt(x: number, y: number) : number | undefined { return queriesMod.protectGroupAt(this, x, y); }

  toneEqZoneAt(x: number, y: number) : Promise<number | undefined> { return queriesMod.toneEqZoneAt(this, x, y); }

  /**
   * A new open is on its way (called outside the queue, when the message arrives):
   * the open running now stops at its next check instead of finishing first.
   */
  cancelOpen() { this.generation++; }

  async makeProxy() {
    const s = this.s!;
    const gpu = this.gpu;
    const { width: W, height: H } = s.work;
    const k = Math.min(1, this.previewLong / Math.max(W, H));
    const w = Math.max(1, Math.round(W * k)), h = Math.max(1, Math.round(H * k));
    this.releaseProxy(s);
    // Image no larger than the preview: render the working textures directly (never freed here).
    if (k === 1) { s.proxy = { base: s.work.tex, denoised: s.denoised, w, h, owned: false }; return; }
    const base = await downsample(gpu, s.work.tex, W, H, w, h, false, 1, "proxy.base");
    const dn = s.denoised === s.work.tex ? base : await downsample(gpu, s.denoised, W, H, w, h, false, 1, "proxy.denoised");
    s.proxy = { base, denoised: dn, w, h, owned: true };
  }

  releaseProxy(s: Session) {
    const d = s.draft;
    s.draft = undefined;
    if (d) { if (d.denoised !== d.base) this.gpu.release(d.denoised); this.gpu.release(d.base); }
    const p = s.proxy;
    s.proxy = undefined;
    if (!p || !p.owned) return;
    if (p.denoised !== p.base) this.gpu.release(p.denoised);
    this.gpu.release(p.base);
  }

  renderSource(full: boolean): RenderSource {
    const s = this.s!;
    if (full || !s.proxy) return { base: s.work.tex, denoised: s.denoised, width: s.work.width, height: s.work.height, fullWidth: s.work.width, skin: s.skin };
    return { base: s.proxy.base, denoised: s.proxy.denoised, width: s.proxy.w, height: s.proxy.h, fullWidth: s.work.width, skin: s.skin };
  }

  effectiveParams(): Params {
    const s = this.s!;
    return this.before ? this.cameraParams() : s.params;
  }

  /** "Before": camera rendering only — exposure/WB from the camera, tone curve, nothing adaptive, no edits. */
  cameraParams(): Params {
    const s = this.s!;
    const p = structuredClone(s.params);
    const e = p.enable;
    e.denoise = false; e.localTone = false; e.semantic = false; e.dehaze = false; e.sharpen = false; e.dof = false; e.curves = false;
    p.exposure = 0;
    p.wb = { temp: s.work.camera?.temp ?? 6504, tint: s.work.camera?.tint ?? 0 };
    p.tone = { highlights: 0, shadows: 0, whites: 0, blacks: 0, contrast: 0, rolloff: 0.5, displayReferred: s.params.tone.displayReferred };
    p.color = { saturation: 0, vibrance: 0 };
    if (p.vignette) p.vignette = { ...p.vignette, amount: 0 };
    if (p.grain) p.grain = { ...p.grain, amount: 0 };
    if (p.film) p.film = { ...p.film, character: "off" };
    p.profile = neutralProfile();
    return p;
  }

  wbFor(p: Params): number[] {
    const s = this.s!;
    const src = s.decoded.source;
    return wbMatrix(src.kind !== "rgb" ? src.color : undefined, s.work.camera, p.wb.temp, p.wb.tint);
  }

  /** Renders the preview now. Call only from inside `exclusive` (or via requestRender). */
  async renderNow(final: boolean, draft = false) {
    const s = this.s;
    if (!s) return;
    const t0 = performance.now();
    const p = this.effectiveParams();
    await this.ensureSelections(s, p);
    await ensureMotionPlate(this, s, p);
    const src = draft && s.proxy ? await this.draftSource() : this.renderSource(false);
    if (s.draft) this.scheduleDraftRelease(s);
    const dof = p.enable.dof && p.dof.strength > 0;
    const r = await this.renderer.render(src, s.maps, p, { wb: this.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38", debugView: this.view, region: this.region, zoneRange: this.zoneRange(), draft }, dof);
    // Histograms for the curve boxes (the edit as rendered; not for "before" or debug views),
    // computed after the preview is on its way so they never delay it.
    const wantHist = final && !draft && !this.before && this.view === 0;
    const wantCalib = final && !draft && !this.before && this.view === 0 && !!s.calib && (s.calib.rounds < 2 || !s.calib.black || !s.calib.color) && Math.abs(p.exposure - s.decision.params.exposure) < 1e-6;
    if (this.display) {
      // Straight onto the page's canvas: no readback, transfer or drawing on the page.
      const { canvas, ctx } = this.display;
      if (canvas.width !== src.width || canvas.height !== src.height) { canvas.width = src.width; canvas.height = src.height; }
      await this.gpu.run("present", (enc) => {
        enc.copyTextureToTexture({ texture: r.tex, origin: { x: 0, y: r.top } }, { texture: ctx.getCurrentTexture() }, { width: src.width, height: src.height });
      });
      if (!wantHist && !wantCalib) {
        this.post({ type: "preview", width: src.width, height: src.height, space: "p3", final, ms: performance.now() - t0 });
        return;
      }
    }
    const data = await this.gpu.readTexture(r.tex, 0, 0, src.width, src.height, 4);
    // Only the pixels the histograms read (every 3rd in each direction: ~1/9 of the frame), not a full copy.
    const HS = 3;
    const hw = Math.ceil(src.width / HS), hh = Math.ceil(src.height / HS);
    let pixels: Uint8Array | undefined;
    if (wantHist) {
      const all = new Uint32Array(data, 0, src.width * src.height);
      const sub = new Uint32Array(hw * hh);
      for (let y = 0, k = 0; y < src.height; y += HS) for (let x = 0; x < src.width; x += HS) sub[k++] = all[y * src.width + x];
      pixels = new Uint8Array(sub.buffer);
    }
    // Calibration measures the development, not the film over it (its glow lifts the shadows,
    // its shoulder lowers the whites): with a film on, a render of its own without it.
    let cdata = data;
    if (wantCalib && filmOf(p)) {
      const bare = { ...p, film: { ...p.film!, character: "off" as const } };
      const rc = await this.renderer.render(src, s.maps, bare, { wb: this.wbFor(bare), gain: s.gain, lightLinear: s.lightLinear, output: "p38", debugView: this.view, region: this.region, zoneRange: this.zoneRange(), draft }, dof);
      cdata = await this.gpu.readTexture(rc.tex, 0, 0, src.width, src.height, 4);
    }
    const calib = wantCalib ? renderedQuantiles(new Uint8Array(cdata)) : undefined; // read before `data` is transferred
    const calibHi = wantCalib ? renderedQuantiles(new Uint8Array(cdata), HI_QS) : undefined;
    const oursC = wantCalib && s.calib?.black && !s.calib.color ? chromaStats(new Uint8Array(cdata)) : undefined;
    if (this.display) this.post({ type: "preview", width: src.width, height: src.height, space: "p3", final, ms: performance.now() - t0 });
    else this.post({ type: "preview", width: src.width, height: src.height, data, space: "p3", final, ms: performance.now() - t0 }, [data]);
    // Calibration to the camera's rendering (calibration.ts).
    if (calib !== undefined && s.calib) calibrateToCamera(this, s, p, calib, calibHi, oursC);
    if (pixels) {
      const hist = previewHistograms(pixels, hw, hh, s.scene.seg, s.distCPU, p.depthBands ?? [0.33, 0.66], 1);
      this.post({ type: "histograms", data: hist }, [hist.buffer]);
    }
  }

  setParams(p: Params, draft = false) {
    if (!this.s) return;
    // Profiles can come from older saved sessions or imports: fill any missing fields.
    this.s.params = { ...p, profile: normalizeProfile(p.profile) };
    this.requestRender(!draft, draft);
  }

  setView(view: 0 | 1 | 2 | 4 | 5 | 6 | 9 | 11, before = false, region = 0, range?: [number, number]) {
    this.view = view;
    this.region = region;
    this.viewRange = range;
    this.before = before;
    this.requestRender(true);
  }

  setPreviewSize(long: number) {
    this.previewLong = Math.max(512, Math.min(isMobile() ? 3072 : Math.min(8192, this.gpu?.info.maxTextureDimension2D ?? 8192), Math.round(long)));
  }

  /** Zooming into the preview: rebuild it at a higher resolution (or back down). Call inside `exclusive`. */
  async resizePreview(long: number) {
    const prev = this.previewLong;
    this.setPreviewSize(long);
    if (!this.s || this.previewLong === prev) return;
    // Render targets are cached per size: the old size's would otherwise stay allocated.
    this.renderer.releaseTargets();
    await this.makeProxy();
    await this.renderNow(true);
  }

  toneEqHistograms() : Promise<Record<ToneEqDetail, MaskHist> | undefined> { return queriesMod.toneEqHistograms(this); }

  depthBandHistograms(s: Session, b1: number, b2: number) { return queriesMod.depthBandHistograms(this, s, b1, b2); }

  cacheDistance() { return openMod.cacheDistance(this); }

  depthField(cols = 24) : { w: number; h: number; data: number[]; vanish: [number, number] } { return queriesMod.depthField(this, cols); }

  motionField(layer: number) : Promise<{ w: number; h: number; data: number[]; mask: number[]; vanish: [number, number]; range: [number, number]; reach: number }> { return queriesMod.motionField(this, layer); }

  focusAt(x: number, y: number) : number | undefined { return queriesMod.focusAt(this, x, y); }

  focusRangeAt(x: number, y: number) : { dist: number; range: [number, number] } | undefined { return queriesMod.focusRangeAt(this, x, y); }

  importLook(name: string, text: string) { return looksMod.importLook(this, name, text); }

  export(format: ExportFormat, quality: number, space: "srgb" | "p3", stripRows = 512) : Promise<{ blob: Blob; name: string; ms: number }> { return exportMod.exportPhoto(this, format, quality, space, stripRows); }

  readHalfRows(tex: GPUTexture, top: number, W: number, rows: number, out: Float32Array<ArrayBuffer>) { return exportMod.readHalfRows(this, tex, top, W, rows, out); }



  summary(name: string) : Summary { return openMod.summary(this, name); }

  profile() { return this.profiler.stages; }

  // ------------------------------------------------------------------ looks

  thumb?: { base: GPUTexture; denoised: GPUTexture; w: number; h: number; long: number };

  dropThumb() {
    const t = this.thumb;
    this.thumb = undefined;
    if (t) this.gpu.release(t.base, t.denoised === t.base ? undefined : t.denoised);
  }

  async ensureThumb(long: number) {
    const s = this.s!;
    if (this.thumb && this.thumb.long === long) return this.thumb;
    this.dropThumb();
    const { width: W, height: H } = s.work;
    const k = Math.min(1, long / Math.max(W, H));
    const w = Math.max(8, Math.round(W * k)), h = Math.max(8, Math.round(H * k));
    const base = await downsample(this.gpu, s.work.tex, W, H, w, h, false, 1, "thumb.base");
    const dn = s.denoised === s.work.tex ? base : await downsample(this.gpu, s.denoised, W, H, w, h, false, 1, "thumb.dn");
    this.thumb = { base, denoised: dn, w, h, long };
    return this.thumb;
  }

  thumbnails(profiles: LookProfile[], long: number) { return looksMod.thumbnails(this, profiles, long); }

  technicalPixels(long = 384) { return looksMod.technicalPixels(this, long); }




  pickAt(x: number, y: number, layer?: number, object = false) : Promise<PickInfo | undefined> { return selectionMod.pickAt(this, x, y, layer, object); }

  maskUnderTap(s: Session, x: number, y: number, layer: number, object: boolean) : Promise<{ inMask: number; sameAs?: string }> { return selectionMod.maskUnderTap(this, s, x, y, layer, object); }

  ensureSelections(s: Session, p: Params) { return selectionMod.ensureSelections(this, s, p); }

  ensureRetouch(s: Session, p: Params) { return retouchMod.ensureRetouch(this, s, p); }

  inpaintStroke(s: Session, st: NonNullable<Session["retouch"]>, stroke: RetouchStroke) : Promise<{ rect: Rect; before: Uint16Array[] }> { return retouchMod.inpaintStroke(this, s, st, stroke); }

  writeCrop(s: Session, rect: Rect, before: Uint16Array[]) { return retouchMod.writeCrop(this, s, rect, before); }

  writeTexels(t: GPUTexture, rect: Rect, data: Uint16Array) { return retouchMod.writeTexels(this, t, rect, data); }

  scaleRetouch(s: Session, k: number) { return retouchMod.scaleRetouch(this, s, k); }

  selectionMask(s: Session, st: Selections, m: MaskShape) : Promise<Uint8Array> { return selectionMod.selectionMaskFor(this, s, st, m); }

  check() : Promise<{ items: CheckItem[]; rgba: Uint8Array; w: number; h: number }> { return checkMod.check(this); }
  lastCheck?: { s: Session; items: CheckItem[]; seg: NonNullable<CheckInput["seg"]>; scene: CheckInput["scene"]; camera?: CheckInput["camera"] };

  solveCheckFixes(onFix: (id: CheckItem["id"], fix: FixChange[], partial: boolean) => void) { return checkMod.solveCheckFixes(this, onFix); }

  solveFixes(s: Session, items: CheckItem[], seg: NonNullable<CheckInput["seg"]>, scene: CheckInput["scene"], camera: CheckInput["camera"], onFix: (it: CheckItem) => void) { return checkMod.solveFixes(this, s, items, seg, scene, camera, onFix); }

  brightestPoint() : Promise<{ x: number; y: number }> { return queriesMod.brightestPoint(this); }

  palette() : Promise<ColorStats> { return queriesMod.palette(this); }

  reference(file: File, mode: "create" | "match", amount: number) : Promise<{ profile: LookProfile; reference: ColorStats; message: string }> { return looksMod.reference(this, file, mode, amount); }

  get session() { return this.s; }

  /** GPU memory the engine holds now and at most since the last call (MB): the memory guard's per-step numbers. */
  memStats(): { liveMB: number; peakMB: number } {
    const MB = 1048576;
    return { liveMB: +(this.gpu.liveBytes() / MB).toFixed(1), peakMB: +(this.gpu.takeStepPeak() / MB).toFixed(1) };
  }
  get wbCamera(): CameraColor | undefined { return this.s?.work.camera; }
}
