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
 */
import { Gpu } from "../gpu/gpu.ts";
import { halvesToFloats } from "../gpu/half.ts";
import { decodeFile } from "../decode/decode.ts";
import type { DecodedImage } from "../decode/types.ts";
import { develop, type WorkingImage } from "../raw/develop.ts";
import { Neural, MODELS } from "../neural/ort.ts";
import { analyseScene, analyseSceneIsolated, neutralScene, applyAppleMattes, type AnalysisLevel, type SceneMaps } from "../neural/scene.ts";
import { denoiseGPU } from "../restore/denoise.ts";
import { UpscaleJob, probeUpscaler } from "../restore/upscale.ts";
import { decideUpscale, measureQuality, type ImageQualityReport, type UpscaleMode } from "../analysis/quality.ts";
import { downsample, guideSize, refine, releaseRefined, type RefinedMaps } from "../refine/refine.ts";
import { blurReport, lumPercentiles, measureBlocks, measureRegions, noiseProfile } from "../analysis/analysis.ts";
import type { AnalysisReport } from "../analysis/types.ts";
import { decide, type DecisionResult } from "../decision/engine.ts";
import { autoFocus, objectDepthRange } from "../decision/focus.ts";
import { buildAutoLayers } from "../layers/auto.ts";
import { embeddedPreviewPixels, embeddedPreviewStats, MEDIAN, REF_QS, renderedQuantiles, shadowMatch } from "../decode/preview.ts";
import { displayQuantiles } from "../decision/autoCurves.ts";
import { allMask, makeLayer } from "../layers/model.ts";
import { depthZones } from "../decision/zones.ts";
import { cellCoverage, previewHistograms } from "../analysis/previewHist.ts";
import { applyAutoCurves, autoCurves } from "../decision/autoCurves.ts";
import { measureSkin, naturalSkin } from "../decision/skinTone.ts";
import type { Params } from "../decision/params.ts";
import { Renderer, type RenderSource } from "../render/renderer.ts";
import { wbMatrix, neutralToTempTint } from "../color/wb.ts";
import { parseCube } from "../render/looks.ts";
import { neutralProfile, normalizeProfile, type LookProfile } from "../looks/profile.ts";
import { analyseColors, type ColorStats } from "../looks/palette.ts";
import { matchProfile, profileFromReference, type RegionColors } from "../looks/reference.ts";
import { GROUPS, type Group } from "../neural/scene.ts";
import { canEncodeHeic, encodeGainMapJpeg, encodeHeic, encodeJpeg, encodeLinearDng, encodeTiff16 } from "../output/encoders.ts";
import { Profiler } from "./profiler.ts";
import type { Capabilities, ExportFormat, FromWorker, PickInfo, Summary, UpscaleInfo } from "./protocol.ts";
import type { CameraColor } from "../color/dng.ts";
import { srgbEotf, srgbOetf } from "../color/transfer.ts";
import { linSrgbToOklab } from "../color/oklab.ts";
import { SamSelector } from "../neural/sam.ts";
import { isPhone, phoneForced } from "../device.ts";
import { levelsByArea, selectionMask } from "../refine/selection.ts";
import { selectKey, type MaskShape } from "../layers/model.ts";
import { liveLayers } from "../layers/gpu.ts";
import type { CheckItem, CheckInput } from "../analysis/check.ts";
import type { FixChange } from "../analysis/checkFix.ts";
// The check's code is loaded on the first check (src/analysis/check*.ts): not part of opening a photo.
const checkCode = () => Promise.all([import("../analysis/check.ts"), import("../analysis/checkFix.ts")]);

/** The scene's light level (EV at ISO 100) from the photo's exposure settings, when they are physically sensible (resized or re-saved files can carry junk). */
function sceneEV(m: { fNumber?: number; exposureTime?: number; iso?: number }): number | undefined {
  const ok = m.fNumber && m.exposureTime && m.iso && m.fNumber >= 0.9 && m.fNumber <= 32 && m.exposureTime >= 1 / 32000 && m.exposureTime <= 60 && m.iso >= 12 && m.iso <= 409600;
  const ev = ok ? Math.log2((m.fNumber! * m.fNumber!) / m.exposureTime!) - Math.log2(m.iso! / 100) : NaN;
  return ev >= -6 && ev <= 21 ? ev : undefined;
}
import { inverse, mul, mulVec } from "../color/mat3.ts";
import { P3_D65, SRGB, rgbToXYZ } from "../color/spaces.ts";
import type { Region } from "../decision/params.ts";

/** Linear Display P3 → linear sRGB (as P3_TO_SRGB in common.wgsl). */
const P3_TO_SRGB = mul(inverse(rgbToXYZ(SRGB)), rgbToXYZ(P3_D65));

type Post = (m: FromWorker, transfer?: Transferable[]) => void;

interface Session {
  name: string;
  /** The opened file (a reference, not a copy): Check reads the camera's own rendering from it. */
  file: File;
  /** The camera's own rendering (the JPEG inside a DNG) at 256 px, read on the first check; null = none. */
  cameraRef?: { rgba: Uint8Array; w: number; h: number } | null;
  decoded: DecodedImage;
  work: WorkingImage;
  denoised: GPUTexture; // === work.tex until neural restoration ran
  gain: number;
  scene: SceneMaps;
  maps: RefinedMaps;
  report: AnalysisReport;
  decision: DecisionResult;
  params: Params;
  /**
   * Exposure calibration against the camera's rendering (DNG preview): the first
   * final previews are measured and the automatic exposure corrected (≤ 2 rounds),
   * unless the exposure was changed by then.
   */
  calib?: { ref: number[]; rounds: number; black: boolean };
  /** Preview proxy. `owned` is false when it aliases the working textures (image ≤ preview size). */
  proxy?: { base: GPUTexture; denoised: GPUTexture; w: number; h: number; owned: boolean };
  /** Quarter-pixel proxy used while a slider is being dragged. */
  draft?: { base: GPUTexture; denoised: GPUTexture; w: number; h: number };
  distCPU?: { w: number; h: number; data: Float32Array };
  lightLinear: [number, number, number];
  /** Working pixels per original working pixel along each axis: 2 after upscaling. */
  scale: 1 | 2;
  /** Apple's skin matte (ProRAW), uploaded once and sampled while rendering. */
  skin?: GPUTexture;
  /** The quality analysis and what the upscale stage did with it. */
  upscale?: UpscaleInfo;
  /** Tap-to-select: the photo's selector, masks by selection, and the texture the renderer samples. */
  sel?: Selections;
  /** Opened from a series of shots (src/burst): the merge, and the reference shot alone for A/B. */
  series?: SeriesState;
}

interface SeriesState {
  files: File[];
  refIndex: number;
  used: number;
  factor: number;
  noise: { single: number; merged: number };
  /** The merged image (and its denoised version) while the single shot is shown. */
  merged: { tex: GPUTexture; denoised: GPUTexture };
  /** The reference shot alone, developed on first request. */
  single?: { tex: GPUTexture; denoised: GPUTexture };
  showing: "merged" | "single";
}

interface Selections {
  sam: SamSelector;
  /** Masks at the guide resolution, by selectKey (most recently used last). */
  cache: Map<string, Uint8Array>;
  /** Selections that failed (not retried on every render). */
  failed: Set<string>;
  /** The selections in the texture, one array layer each, in order. */
  keys: string[];
  tex?: GPUTexture;
  version: number;
  /** The photo's luminance at the guide resolution (edge snapping). */
  guide?: Float32Array;
}

/** Default blur strength whenever depth of field is switched on (scene-independent, by preference). */
const DEFAULT_DOF_STRENGTH = 0.5;

/** Largest working image (MP) the 2× stage may produce: a 2× texture must fit next to everything else. */
const UPSCALE_MAX_MP = () => (isMobile() ? 16 : 48);

/** Long edge of the image the analysis networks see. */
const ANALYSIS_LONG = 1036;
/** Phones: exactly one 512 px segmentation window, no sliding. */
const ANALYSIS_LONG_LIGHT = 512;

const isMobile = isPhone;

export class Engine {
  private gpu!: Gpu;
  private neural!: Neural;
  private renderer!: Renderer;
  private post: Post;
  private s?: Session;
  /** An explicit depth range to highlight in view 5 (a distance band). */
  private viewRange?: [number, number];
  private previewLong = isMobile() ? 1600 : 2048;
  private view: 0 | 1 | 2 | 4 | 5 | 6 = 0;
  private region = 0;
  private before = false;
  private generation = 0;
  private profiler = new Profiler();

  constructor(post: Post) { this.post = post; }

  /**
   * One serial queue for all GPU work on the session. Opening a photo,
   * restoration, export, previews and look thumbnails create and destroy
   * textures; running any two at once let a render submit a texture another
   * job had just destroyed ("Destroyed texture used in a submit").
   */
  private chain: Promise<unknown> = Promise.resolve();
  exclusive<T>(f: () => Promise<T>): Promise<T> {
    const r = this.chain.then(f, f);
    this.chain = r.catch(() => {});
    return r;
  }
  private renderQueued = false;
  private queuedDraft = false;
  private queuedFinal = false;
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
  private zoneRange(): [number, number] | undefined {
    if ((this.view === 5 || this.view === 4) && this.viewRange) return this.viewRange; // view 4: a region at a distance
    const e = this.s?.decision.dofSuggestion.zoneEdges;
    if (this.view !== 5 || !e) return undefined;
    const i = Math.min(4, Math.max(0, this.region));
    return [i === 0 ? -1 : e[i], i === 4 ? 2 : e[i + 1]];
  }

  /** The page's canvas, when previews are drawn straight into it on the GPU. */
  private display?: { canvas: OffscreenCanvas; ctx: GPUCanvasContext };
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

  private draftIdle = 0;
  /** The draft copy is freed a few seconds after the last drag (it comes back on the next one). */
  private scheduleDraftRelease(s: Session) {
    clearTimeout(this.draftIdle);
    this.draftIdle = setTimeout(() => void this.exclusive(async () => {
      if (this.s !== s || !s.draft) return;
      this.gpu.release(s.draft.base, s.draft.denoised === s.draft.base ? undefined : s.draft.denoised);
      s.draft = undefined;
    }), 8000) as unknown as number;
  }

  /** Half-size (quarter-pixel) copy of the preview proxy, created on first drag. */
  private async draftSource(): Promise<RenderSource> {
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
  private base = "";
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

  private log(text: string) { this.post({ type: "log", text }); }
  private progress(stage: string, detail?: string, frac?: number) { this.post({ type: "progress", stage, detail, frac }); }

  private closeSession() {
    const s = this.s;
    if (!s) return;
    const g = this.gpu;
    if (s.denoised !== s.work.tex) g.release(s.denoised);
    g.release(s.work.tex, s.skin);
    // A series keeps the image not shown (merge or single shot) too.
    for (const t of [s.series?.merged, s.series?.single]) if (t) g.release(t.tex, t.denoised);
    if (s.sel) { s.sel.sam.dispose(); g.release(s.sel.tex); this.renderer.selection = undefined; }
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
  async open(file: File, resolution: "auto" | "full" | "half", autoExposure = false, autoDof = false, upscaleMode: UpscaleMode = "auto", safeAnalysis = false, level?: AnalysisLevel, series?: { files: File[]; ref?: number }) {
    const cleanup: Array<() => void> = [];
    let committed = false;
    try {
      await this.openInner(file, resolution, autoExposure, autoDof, upscaleMode, safeAnalysis, level, (f) => cleanup.push(f), () => { committed = true; }, series);
    } finally {
      if (!committed) for (const f of cleanup.reverse()) { try { f(); } catch { /* already freed */ } }
    }
  }

  private async openInner(file: File, resolution: "auto" | "full" | "half", autoExposure: boolean, autoDof: boolean, upscaleMode: UpscaleMode, safeAnalysis: boolean, level: AnalysisLevel | undefined,
    track: (free: () => void) => void, commit: () => void, series?: { files: File[]; ref?: number }) {
    const gen = ++this.generation;
    this.closeSession();
    this.gpu.flushStaging(); // the previous photo's readback sizes
    const P = new Profiler(this.gpu);
    this.profiler = P;
    const gpu = this.gpu;
    // Working resolution: iPhone memory decides, not desktop assumptions.
    const factorFor = (w: number, h: number) => {
      const mp = (w * h) / 1e6;
      let f = 1;
      if (resolution === "half") f = 2;
      else if (resolution === "auto") f = isMobile() && mp > 16 ? 2 : 1;
      while (Math.max(w, h) / f > gpu.info.maxTextureDimension2D) f++;
      return f;
    };
    let decoded: DecodedImage, work: WorkingImage, seriesInfo: Omit<SeriesState, "merged" | "showing"> | undefined;
    if (series && series.files.length > 1) {
      // Several shots of one scene: merged into one working image (src/burst), then opened like a photo.
      const { mergeSeries } = await import("../burst/burst.ts");
      const r = await P.time("series merge", () => mergeSeries(gpu, series.files, {
        factorFor, mode: "clean", ref: series.ref, exposureGain: exposureGainY,
        progress: (st, d, f) => this.progress(st, d, f), log: (t) => this.log(t), track, cancelled: () => gen !== this.generation,
      }), (r) => `${r.used} frames ${r.work.width}×${r.work.height}`);
      decoded = r.decoded; work = r.work; file = r.file;
      track(() => gpu.release(work.tex));
      seriesInfo = { files: series.files, refIndex: r.refIndex, used: r.used, factor: r.factor, noise: r.noise };
      work.log.forEach((l) => this.log(l));
      upscaleMode = "off"; // the merge is the detail stage
    } else {
      this.progress("decode", file.name);
      // The file bytes are only needed by the decoder (which copies them): no
      // reference is kept here, so 30–80 MB can be collected during development.
      decoded = await P.time("decode", async () => decodeFile(new Uint8Array(await file.arrayBuffer()), file.name, file.type), (d) => `${d.format} via ${d.source.kind === "rgb" ? d.source.decoder : "LibRaw"}`);
      const dec = decoded;
      track(() => dec.close());
      for (const [k, v] of Object.entries(decoded.timings)) this.log(`  ${k}: ${v.toFixed(0)} ms`);
      const src0 = decoded.source;
      const factor = factorFor(src0.width, src0.height);
      this.progress("develop", `${src0.width}×${src0.height}${factor > 1 ? ` → 1/${factor}` : ""}`);
      const w0 = await P.time("raw development", () => develop(gpu, dec, { factor }), (w) => `${w.width}×${w.height}`);
      work = w0;
      track(() => gpu.release(w0.tex));
      work.log.forEach((l) => this.log(l));
      // The sensor data now lives on the GPU; free the decoder's wasm heap. The
      // raw view points into that heap, so it must be dropped as well — otherwise
      // the whole LibRaw memory (≈150 MB for 12 MP, ≈460 MB for 48 MP) stays
      // alive for as long as the photo is open.
      decoded.close();
      decoded.close = () => {};
      if (decoded.source.kind !== "rgb") decoded.source.data = new Uint16Array(0);
    }
    const src = decoded.source;
    if (gen !== this.generation) return;

    // --- reduced analysis image + exposure normalisation gain ------------------
    this.progress("analysis image");
    // The networks see a ~1036 px image: segmentation slides 512 px windows over
    // it and depth adds high-resolution detail tiles to a global pass. Still a
    // reduced image — never the 12/48 MP original.
    // Phones: "light" at most — one 512 px segmentation pass and a 392 px depth pass
    // (a sixth of the segmentation work, about half the depth memory). The page
    // may ask for less after the tab died during analysis on this device.
    const lvl: AnalysisLevel = isMobile() && (!level || level === "full") ? "light" : (level ?? "full");
    const aScale = Math.min(1, (lvl === "full" ? ANALYSIS_LONG : ANALYSIS_LONG_LIGHT) / Math.max(work.width, work.height));
    const gw = Math.max(16, Math.round(work.width * aScale)), gh = Math.max(16, Math.round(work.height * aScale));
    const lin = await downsample(gpu, work.tex, work.width, work.height, gw, gh, false, 1, "analysis.lin");
    const linHalf = new Uint16Array(await gpu.readTexture(lin, 0, 0, gw, gh, 8));
    gpu.release(lin);
    const linF = halvesToFloats(linHalf);
    const gain = exposureGain(linF);
    const analysisRgba = new Float32Array(gw * gh * 4);
    for (let i = 0; i < gw * gh * 4; i += 4) {
      for (let c = 0; c < 3; c++) {
        const v = Math.min(1, Math.max(0, linF[i + c] * gain));
        analysisRgba[i + c] = srgbOetf(v);
      }
      analysisRgba[i + 3] = 1;
    }
    this.log(`analysis image ${gw}×${gh}; normalisation gain ${gain.toFixed(3)} (${Math.log2(gain).toFixed(2)} EV)`);

    // --- semantic segmentation + depth (reduced image only) ------------------------
    // Phones (and devices where analysis crashed before): on the CPU in a worker of its
    // own that is terminated afterwards — the model runtime's memory is returned at
    // once instead of staying for the tab's life. Elsewhere: WebGPU, in this worker.
    const isolated = isMobile() || safeAnalysis;
    const withDepth = lvl !== "seg";
    const depthLong = lvl === "full" ? 518 : 392;
    const detailTiles = lvl === "full" && !isMobile(); // 4 more depth passes, too heavy for phones
    const inHere = () => analyseScene(this.neural, { rgba: analysisRgba, width: gw, height: gh }, (s) => this.progress(s), withDepth, detailTiles, safeAnalysis || isMobile() ? "wasm" : this.neural.backend, depthLong);
    const img = { rgba: analysisRgba, width: gw, height: gh };
    const inWorker = () => analyseSceneIsolated(img, this.base, detailTiles, (s) => this.progress(s), withDepth, depthLong, phoneForced());
    if (lvl !== "full") this.log(`scene analysis level: ${lvl}${lvl === "none" ? " (the tab stopped during segmentation before on this device)" : lvl === "seg" ? " (the tab stopped during depth before on this device)" : ""}`);
    const scene = await P.time("segmentation + depth", async () => {
      if (lvl === "none") return neutralScene(img, "skipped on this device");
      if (!isolated) return inHere();
      try { return await inWorker(); }
      catch (e) {
        this.log(`analysis worker failed (${e instanceof Error ? e.message : e}); trying once more`);
        try { return await inWorker(); }
        catch (e2) {
          // On a phone a second failure is memory: analysing in this worker would keep
          // the model runtime's memory for good. Open without the analysis instead.
          if (isMobile()) return neutralScene(img, e2 instanceof Error ? e2.message : String(e2));
          return inHere();
        }
      }
    }, (s) => Object.entries(s.timings).map(([k, v]) => `${k} ${v.toFixed(0)}ms`).join(", "));
    if (import.meta.env.DEV) {
      // Dev only: dump the analysis image and distance map (PGM) for offline inspection.
      const pgm = (w: number, h: number, v: (i: number) => number) => {
        const head = new TextEncoder().encode(`P5 ${w} ${h} 255\n`);
        const out = new Uint8Array(head.length + w * h);
        out.set(head);
        for (let k = 0; k < w * h; k++) out[head.length + k] = Math.max(0, Math.min(255, Math.round(v(k) * 255)));
        return out;
      };
      const d = scene.depth;
      fetch("/__debug/save?name=depth.pgm", { method: "POST", body: pgm(d.width, d.height, (k) => d.dist[k]) }).catch(() => undefined);
      fetch("/__debug/save?name=analysis.pgm", { method: "POST", body: pgm(gw, gh, (k) => analysisRgba[k * 4 + 1]) }).catch(() => undefined);
    }
    // A ProRAW file carries Apple's own sky / skin mattes: sharper edges than
    // the network can produce on a reduced image, and already computed.
    if (decoded.masks?.length) scene.log.push(...applyAppleMattes(scene.seg, decoded.masks));
    // The skin matte also goes to the GPU: the look's skin protection uses it
    // directly instead of guessing skin from the person mask and its colour.
    const skinMatte = decoded.masks?.find((m) => m.kind === "skin");
    let skinTex: GPUTexture | undefined;
    if (skinMatte) {
      skinTex = gpu.tex("apple.skin", skinMatte.width, skinMatte.height, "r8unorm", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST);
      const t = skinTex;
      track(() => gpu.release(t));
      gpu.device.queue.writeTexture({ texture: skinTex }, skinMatte.data as Uint8Array<ArrayBuffer>, { bytesPerRow: skinMatte.width, rowsPerImage: skinMatte.height }, { width: skinMatte.width, height: skinMatte.height });
      this.log(`Apple skin matte ${skinMatte.width}×${skinMatte.height} drives the look's skin protection`);
    }
    scene.log.forEach((l) => this.log(l));
    if (gen !== this.generation) return; // (freed by open())

    // --- refinement ---------------------------------------------------------------
    this.progress("refine masks");
    const maps = await P.time("mask/depth refinement", () => refine(gpu, work.tex, work.width, work.height, gain, scene), (m) => `guide ${m.w}×${m.h}, r=${m.params.maskRadius}`);
    track(() => releaseRefined(gpu, maps));

    // --- statistics -------------------------------------------------------------------
    this.progress("statistics");
    const report = await P.time("image statistics", async () => {
      const b0 = await measureBlocks(gpu, work.tex, work.width, work.height, gain, 0.02);
      const noise = noiseProfile(b0);
      const thr = Math.max(0.02, 8 * noise.mid);
      const blocks = thr > 0.0201 ? await measureBlocks(gpu, work.tex, work.width, work.height, gain, thr) : b0;
      const blur = blurReport(blocks, noise);
      const reg = await measureRegions(gpu, maps, gain);
      const r: AnalysisReport = {
        width: work.width, height: work.height, gain, referred: work.referred,
        isProRaw: src.kind !== "rgb" && src.isProRaw, iso: decoded.meta.iso,
        global: reg.regions.global, groups: reg.regions as AnalysisReport["groups"],
        histLum: reg.histLum, histRGB: reg.histRGB, atmosphere: reg.atmosphere, noise, blur, blocks,
        lum: lumPercentiles(reg.histLum), timings: {},
      };
      return r;
    }, (r) => `noise σ ${(r.noise.mid * 255).toFixed(2)}/255, blur ${r.blur.median.toFixed(2)}px`);
    this.log(`noise bins (y: σ/255 luma, chroma, blocks): ` + report.noise.bins.map((b) => `${b.y.toFixed(2)}: ${(b.sigma * 255).toFixed(2)}, ${(b.sigmaC * 255).toFixed(2)}, ${b.blocks}`).join(" | "));
    this.log(`regions: ` + Object.entries(report.groups).filter(([, g]) => g.area > 0.01).map(([k, g]) => `${k} ${(g.area * 100).toFixed(0)}% ${g.meanEV.toFixed(2)}EV C${g.chroma.toFixed(3)} d${g.dist.toFixed(2)}`).join("; "));
    this.log(`noise σ (encoded, /255): mid ${(report.noise.mid * 255).toFixed(2)}, shadows ${(report.noise.shadow * 255).toFixed(2)}, chroma ${(report.noise.chroma * 255).toFixed(2)}; blur median ${report.blur.median.toFixed(2)} px over ${report.blur.edgeBlocks} edge blocks`);

    // --- decisions ----------------------------------------------------------------------
    const colorInput = src.kind !== "rgb" ? src.color : undefined;
    // The camera's own rendering (the JPEG inside a DNG): the brightness reference.
    const reference = src.kind !== "rgb" && /\.dng$/i.test(file.name) ? await P.time("camera rendering", () => embeddedPreviewStats(file, REF_QS, isMobile() ? 24 : Infinity)) : undefined;
    if (reference) this.log(`camera rendering: ${reference.width}×${reference.height} embedded JPEG, median ${(reference.q[MEDIAN] * 255).toFixed(0)}/255, shadows ${reference.q.slice(0, MEDIAN).map((v) => Math.round(v * 255)).join("/")}`);
    const decideWith = (referenceExposure?: { ev: number; note: string }) => decide({
      report,
      camera: work.camera,
      referred: work.referred,
      isProRaw: report.isProRaw,
      iso: decoded.meta.iso,
      sceneEV: sceneEV(decoded.meta),
      autoExposure,
      solveNeutral: colorInput ? (n) => neutralToTempTint(colorInput, n) : undefined,
      referenceExposure,
    });
    const decision = await P.time("decision engine", () => {
      let d = decideWith();
      if (!reference || !autoExposure) return d;
      // Exposure at which our rendering's median matches the camera's: solved on the
      // display model, then decided again (tone and local settings follow exposure).
      for (let k = 0; k < 2; k++) {
        const m = (e: number) => displayQuantiles(report.global.hist, { tone: d.params.tone, exposure: e, local: { ...d.params.local, anchorEV: d.params.local.anchorEV + d.params.exposure - e } }, [0.5])[0];
        let lo = -1.5, hi = 2.5;
        for (let it = 0; it < 30; it++) { const mid = (lo + hi) / 2; if (m(mid) < reference.q[MEDIAN]) lo = mid; else hi = mid; }
        const ev = Math.round(((lo + hi) / 2) * 100) / 100;
        d = decideWith({ ev, note: `matched to the camera's rendering (median ${(reference.q[MEDIAN] * 255).toFixed(0)}/255)` });
      }
      return d;
    });
    const params = structuredClone(decision.params);
    // Atmospheric light: dark-channel estimate is in the analysis encoding → linear working.
    const A = decision.params.dehaze.light.map((v) => srgbEotf(v) / gain) as [number, number, number];

    const s: Session = { name: seriesInfo ? `${file.name} + ${seriesInfo.used - 1}` : file.name, file, decoded, work, denoised: work.tex, gain, scene, maps, report, decision, params, lightLinear: A, scale: 1, skin: skinTex };
    if (seriesInfo) s.series = { ...seriesInfo, merged: { tex: work.tex, denoised: work.tex }, showing: "merged" };
    if (reference && autoExposure) s.calib = { ref: reference.q, rounds: 0, black: false };
    this.s = s;
    commit(); // from here closeSession() frees it
    await this.cacheDistance();
    // Automatic focus: subject from refined depth + segmentation + composition.
    const af = autoFocus(s.distCPU!, scene.seg, { blur: { bw: report.blocks.bw, bh: report.blocks.bh, data: report.blur.perBlock }, longPx: Math.max(work.width, work.height) });
    // Without a depth map (it failed on this device) nothing can be separated by distance.
    const flatDepth = !!scene.depth.flat;
    decision.dofSuggestion = { justified: af.justified && !flatDepth, focus: af.focus, strength: af.strength, reason: flatDepth ? "no depth map on this device — no automatic blur" : af.reason, x: af.x, y: af.y };
    // Depth zones by natural breaks of this photo's depth (boundaries fall in the
    // gaps between layers). Initial blur per zone follows the automatic focus
    // curve at the zone's mean distance, so switching to zones is seamless.
    {
      const zones = depthZones(s.distCPU!, scene.seg, 5);
      const smooth = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
      const curve = (c: number) => Math.max(smooth(0, 0.55, c - af.focus), 0.6 * smooth(0, 0.4, af.focus - c));
      decision.dofSuggestion.zoneEdges = [0, ...zones.slice(1).map((z) => z.lo), 1];
      decision.dofSuggestion.zones = zones.map((z) => ({ share: z.share, label: z.label, lo: z.lo, hi: z.hi }));
      for (const p of [decision.params, params]) {
        p.dof.zoneBounds = zones.slice(1).map((z) => z.lo);
        p.dof.zones = zones.map((z) => Math.round(curve(z.center) * 100) / 100);
        p.dof.mode = "focus";
      }
      this.log("depth zones (natural breaks): " + zones.map((z, i) => `${i + 1}: ${z.lo.toFixed(2)}–${z.hi.toFixed(2)} ${Math.round(z.share * 100)}% ${z.label}`).join(" | "));
      // Distance bands for curves (near / middle / far): the same natural breaks, in three.
      const z3 = depthZones(s.distCPU!, scene.seg, 3);
      const b1 = z3[1].lo, b2 = Math.max(z3[2].lo, b1 + 0.02);
      for (const p of [decision.params, params]) p.depthBands = [b1, b2];
      decision.dofSuggestion.bands = z3.map((z) => ({ share: z.share, label: z.label, lo: z.lo, hi: z.hi }));
      decision.cellCoverage = cellCoverage(scene.seg, s.distCPU!, [b1, b2]);

      // Atmospheric perspective, only in outdoor scenes with real depth and clear air
      // (hazy scenes already have it): the distance a little quieter and cooler.
      const nearShare = z3[0].share, farShare = z3[2].share;
      // Outdoors only (visible sky): the far side of a room is not atmosphere.
      if (!flatDepth && nearShare >= 0.1 && farShare >= 0.2 && params.dehaze.strength < 0.15 && report.groups.sky.area >= 0.03) {
        for (const p of [decision.params, params]) p.distance = { ...p.distance, far: { ...p.distance.far, saturation: -0.06, warmth: -0.04, clarity: 0.9, texture: 0.9 } };
        decision.decisions.push({ id: "distance.far", value: "atmospheric perspective", reason: `depth: ${Math.round(nearShare * 100)}% near, ${Math.round(farShare * 100)}% far, clear air → the distance slightly less saturated, textured and cooler`, inputs: { near: nearShare, far: farShare } });
      }
      this.log("distance bands for curves: " + z3.map((z, i) => `${["near", "middle", "far"][i]} ${z.lo.toFixed(2)}–${z.hi.toFixed(2)} ${Math.round(z.share * 100)}% ${z.label}`).join(" | "));

      // Natural skin: measured on the skin itself; pulled back only when overdriven.
      {
        const m = s.maps;
        const lin = new Float32Array(m.w * m.h * 4);
        halvesToFloats(new Uint16Array(await this.gpu.readTexture(m.lin, 0, 0, m.w, m.h, 8)), lin);
        const apple = decoded.masks?.find((k) => k.kind === "skin");
        const st = measureSkin(lin, m.w, m.h, { ...scene.seg, plane: GROUPS.indexOf("person") * scene.seg.width * scene.seg.height }, apple);
        if (st) {
          const fix = naturalSkin(st, params, params.skin);
          if (fix.reasons.length) {
            for (const p of [decision.params, params]) p.skin = { ...p.skin, saturation: Math.round((p.skin.saturation + fix.saturation) * 100) / 100, hue: Math.round((p.skin.hue + fix.hue) * 10) / 10 };
            decision.decisions.push({ id: "skin", value: [params.skin.saturation, params.skin.hue], reason: fix.reasons.join("; "), inputs: { share: Math.round(st.share * 1000) / 10, L: Math.round(st.L * 100) / 100, C: Math.round(st.C * 1000) / 1000, hue: Math.round(st.hue) } });
          } else decision.decisions.push({ id: "skin", value: "natural", reason: `skin already natural (chroma ${st.C.toFixed(3)}, hue ${st.hue.toFixed(0)}°) — left alone`, inputs: {} });
        }
      }
      // Automatic curves: the photo, regions, skin and distance, measured through the rendering.
      const bandHist = await this.depthBandHistograms(s, b1, b2);
      const st = (g: Group) => ({ hist: report.groups[g].hist, area: report.groups[g].area, localContrast: report.groups[g].localContrast });
      const ac = autoCurves({
        tone: params.tone, exposure: params.exposure, local: params.local, clipHi: report.global.clipHi,
        chroma: report.global.chroma, haze: params.dehaze.strength,
        photo: { hist: report.global.hist, area: 1 },
        regions: { sky: st("sky"), vegetation: st("vegetation"), water: st("water"), building: st("building"), person: st("person"), ground: st("ground") },
        bands: flatDepth ? undefined : bandHist, // one distance everywhere: no near / far curves
      });
      decision.autoCurves = { photo: ac.photo, regions: ac.regions, depth: ac.depth };
      for (const p of [decision.params, params]) applyAutoCurves(p, decision.autoCurves, p.autoCurves ?? 1);
      decision.decisions.push(...ac.notes);
      if (!ac.notes.length) decision.decisions.push({ id: "curves", value: "flat", reason: "every region, skin and distance already renders within its comfortable range — no automatic curves", inputs: {} });
      this.log(`automatic curves: ${ac.notes.map((n) => n.id).join(", ") || "none"}`);
    }
    // Subject priority: the main subject may get +0.1…+0.25 EV — only when it does
    // not already stand out from its surroundings.
    {
      const g = af.kind === "person" || af.kind === "animal" || af.kind === "vehicle" || af.kind === "building" ? af.kind : undefined;
      const st = g ? report.groups[g] : undefined;
      if (g && st && st.area >= 0.01 && st.area <= 0.45) {
        const sep = st.meanEV - report.global.meanEV;
        if (sep < 0.3) {
          const add = Math.round(Math.min(0.25, Math.max(0.1, 0.1 + (0.3 - sep) * 0.25)) * 100) / 100;
          // A visible, editable layer: "Subject priority" (Exposure on the subject's region).
          for (const p of [decision.params, params]) {
            p.layers = [...(p.layers ?? []), makeLayer("exposure", "Subject priority", { auto: "subject", mask: { ...allMask(), kind: "region", region: g }, params: { exposure: add, offset: 0, gamma: 1 } })];
          }
          decision.decisions.push({ id: "subject", value: add, reason: `main subject (${g}) only ${sep.toFixed(2)} EV above the scene → +${add} EV luminance priority`, inputs: { separationEV: Math.round(sep * 100) / 100, area: st.area } });
        } else decision.decisions.push({ id: "subject", value: 0, reason: `main subject (${g}) already stands out (${sep.toFixed(2)} EV above the scene) — not brightened`, inputs: {} });
      }
    }
    // Keep the decision trace consistent with what auto focus found.
    const dofNote = decision.decisions.find((d) => d.id === "dof");
    if (dofNote) { dofNote.value = af.justified ? "justified" : "not justified"; dofNote.reason = `auto focus at distance ${af.focus.toFixed(2)}: ${af.reason}`; }
    for (const p of [decision.params, params]) {
      // Blur amount from the lens model (auto focus) when the photo calls for it; the
      // default is only a starting point for turning depth of field on by hand.
      p.dof = { ...p.dof, focus: af.focus, focusSpan: af.span, strength: af.justified ? af.strength : DEFAULT_DOF_STRENGTH, points: [] };
      if (autoDof && af.justified) p.enable = { ...p.enable, dof: true };
    }
    this.log(`auto focus: distance ${af.focus.toFixed(2)} at (${af.x.toFixed(2)}, ${af.y.toFixed(2)}) — ${af.justified ? "blur justified" : "no blur"}: ${af.reason}${autoDof && af.justified ? " — applied (Auto depth of field)" : ""}`);
    // The automatic grade becomes layers (src/layers/auto.ts): visible, editable, removable.
    Object.assign(decision.params, buildAutoLayers(decision.params));
    Object.assign(params, buildAutoLayers(params));
    this.post({ type: "analysis", summary: this.summary(file.name), decisions: decision.decisions, auto: decision.params, params, dof: decision.dofSuggestion, exposureSuggestion: decision.exposureSuggestion, autoCurves: decision.autoCurves, cellCoverage: decision.cellCoverage, noDepth: flatDepth ? (scene.log.find((l) => /Depth skipped|Depth unavailable|Scene analysis unavailable/.test(l)) ?? "no depth map") : undefined });
    this.post({ type: "profile", stages: P.stages });
    this.postSeries();

    // --- first preview (before neural restoration) --------------------------------------
    await P.time("preview proxy", () => this.makeProxy());
    await this.renderNow(false);
    this.post({ type: "profile", stages: P.stages });
    if (gen !== this.generation) return;

    // --- neural restoration (tiled, only where needed) -------------------------------------
    await this.restore(gen);
    if (gen !== this.generation) return;

    // --- image quality → optional 2× upscale ----------------------------------------------
    // Measured on the restored image (after denoise), before any tone or
    // look. When the source already has enough detail this is the only cost:
    // the upscaling model is never downloaded or loaded.
    const q = await P.time("quality analysis", () => this.analyseQuality(resolution === "half", upscaleMode), (r) => `${r.megapixels} MP, edge σ ${r.metrics.edgeSigma.toFixed(2)} px, noise ${(r.metrics.noiseSigma * 255).toFixed(2)}/255 → ${r.needsUpscale ? "2×" : "skip"}`);
    await P.time("preview proxy (restored)", () => this.makeProxy());
    await this.renderNow(true);
    this.post({ type: "profile", stages: P.stages });
    // The upscale runs as its own queued job in short slices (see runUpscale), so
    // the photo is fully editable while it works.
    if (q.needsUpscale) void this.runUpscale(gen);
  }

  private async analyseQuality(reducedByUser: boolean, mode: UpscaleMode): Promise<ImageQualityReport> {
    const s = this.s!;
    const { width: W, height: H } = s.work;
    const metrics = await measureQuality(this.gpu, s.denoised, W, H, s.gain);
    const q = decideUpscale(metrics, {
      width: W, height: H, iso: s.decoded.meta.iso, reducedByUser, mode,
      maxOutputMP: UPSCALE_MAX_MP(), maxTextureDimension: this.gpu.info.maxTextureDimension2D, mobile: isMobile(),
    });
    s.upscale = { state: q.needsUpscale ? "pending" : "skipped", upscaleApplied: false, upscaleFactor: 1, upscaleReason: q.reason, code: q.code, vars: q.vars, report: q };
    const m = q.metrics;
    this.log(`quality: ${q.megapixels} MP; sharpness ${q.sharpnessScore.toFixed(2)} (edge σ ${m.edgeSigma.toFixed(2)} px sharpest quartile, ${m.edgeSigmaMedian.toFixed(2)} px median, ${m.edgeBlocks} edge blocks in ${m.patches} patches); ` +
      `noise ${(m.noiseSigma * 255).toFixed(2)}/255 (score ${q.noiseScore.toFixed(2)}); detail ${(m.detailDensity * 100).toFixed(1)}%; Laplacian var ${m.laplacianVar.toExponential(2)}; Tenengrad ${m.tenengrad.toExponential(2)}`);
    this.log(`upscale: ${q.needsUpscale ? "2× planned" : "skipped"} — ${q.reason}`);
    this.post({ type: "upscale", info: s.upscale });
    return q;
  }

  /** "Upscale 2× now" from the Upscale tab: overrides the decision (never the memory budget). */
  forceUpscale() {
    const s = this.s;
    const info = s?.upscale;
    if (!s || !info || s.scale !== 1 || info.state === "running" || info.state === "pending") return;
    const q = decideUpscale(info.report.metrics, {
      width: s.work.width, height: s.work.height, iso: s.decoded.meta.iso, reducedByUser: false, mode: "always",
      maxOutputMP: UPSCALE_MAX_MP(), maxTextureDimension: this.gpu.info.maxTextureDimension2D,
    });
    s.upscale = { state: q.needsUpscale ? "pending" : "skipped", upscaleApplied: false, upscaleFactor: 1, upscaleReason: q.reason, code: q.code, vars: q.vars, report: q };
    this.log(`upscale: requested — ${q.needsUpscale ? "2× planned" : "not possible"} (${q.reason})`);
    this.post({ type: "upscale", info: s.upscale });
    if (q.needsUpscale) void this.runUpscale(this.generation);
  }

  /**
   * 2× upscale of the restored working image, as a chain of short exclusive
   * jobs: each slice runs tiles for ~250 ms, then preview renders and other
   * requests queued meanwhile get their turn. The result replaces the working
   * image atomically at the end, so every later stage (tone, look, semantic,
   * sharpening, depth of field, export) runs at the new resolution. Any
   * failure leaves the photo exactly as it was.
   */
  private async runUpscale(gen: number) {
    const s = this.s;
    if (!s?.upscale) return;
    const src = s.denoised;
    const { width: W, height: H } = s.work;
    const P = this.profiler;
    const info = s.upscale;
    const post = () => this.post({ type: "upscale", info });
    info.state = "running";
    post();
    let session: Awaited<ReturnType<Neural["session"]>> | undefined;
    let job: UpscaleJob | undefined;
    let backend = this.neural.backend;
    const t0 = performance.now();
    try {
      this.progress("detail enhancement", "loading model");
      // Load, then self-test on one probe tile: WebGPU first, WASM if either fails.
      const open = async (b: typeof backend) => {
        const ses = await this.neural.session(MODELS.swin2sr, false, b);
        const bad = await probeUpscaler(ses);
        if (bad) { await ses.release(); throw new Error(bad); }
        return ses;
      };
      try {
        session = await open(backend);
      } catch (e) {
        if (backend !== "webgpu") throw e;
        this.log(`Swin2SR on WebGPU failed (${e instanceof Error ? e.message : e}); retrying on WASM`);
        backend = "wasm";
        session = await open("wasm");
      }
      const live = () => gen === this.generation && this.s === s && s.denoised === src;
      for (let first = true; ; first = false) {
        const done = await this.exclusive(async () => {
          if (!live()) throw new Error("cancelled");
          if (!job) job = new UpscaleJob(this.gpu, session!, src, W, H, s.gain);
          try {
            return await job.step(first ? 0 : 250);
          } catch (e) {
            // A WebGPU failure mid-run: redo the whole image on WASM rather than give up.
            if (backend !== "webgpu") throw e;
            this.log(`Swin2SR WebGPU inference failed (${e instanceof Error ? e.message : e}); retrying on WASM`);
            job.release();
            await session!.release();
            backend = "wasm";
            session = await this.neural.session(MODELS.swin2sr, false, "wasm");
            job = new UpscaleJob(this.gpu, session, src, W, H, s.gain);
            return false;
          }
        });
        const pr = job!.progress;
        this.progress("detail enhancement", `tile ${pr.done}/${pr.total}`, pr.done / pr.total);
        if (done) break;
      }
      await this.exclusive(async () => {
        if (!live()) throw new Error("cancelled");
        const out = job!.out;
        const tiles = job!.total;
        const netMs = job!.progress.msPerTile;
        job!.finish();
        job = undefined;
        // Swap in the 2× image: it is both base and restored image (restoration is baked in).
        if (s.denoised !== s.work.tex) this.gpu.release(s.denoised);
        this.gpu.release(s.work.tex);
        s.work = { ...s.work, tex: out, width: 2 * W, height: 2 * H };
        s.denoised = out;
        s.scale = 2;
        this.dropThumb();
        P.add("2× upscale (Swin2SR)", performance.now() - t0, `${W}×${H} → ${2 * W}×${2 * H} on ${backend}, ${tiles} tiles`);
        Object.assign(info, { state: "applied", upscaleApplied: true, upscaleFactor: 2, width: 2 * W, height: 2 * H });
        this.log(`upscale: 2× applied on ${backend} in ${((performance.now() - t0) / 1000).toFixed(1)} s — ${W}×${H} → ${2 * W}×${2 * H}, ${tiles} tiles, network ${netMs.toFixed(0)} ms/tile`);
        await this.makeProxy();
        await this.renderNow(true);
        post();
        this.post({ type: "profile", stages: P.stages });
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      job?.release();
      if (msg === "cancelled") {
        this.log("upscale: cancelled (photo changed or restoration re-run)");
        Object.assign(info, { state: "cancelled" });
      } else {
        // Never fail the photo because of the optional stage: keep the 1× image.
        this.log(`upscale: failed (${msg}) — continuing without it`);
        Object.assign(info, { state: "failed", upscaleReason: `${info.upscaleReason}; model could not run: ${msg}` });
      }
      if (this.s === s) post();
    } finally {
      await session?.release().catch(() => {});
    }
  }

  /** Noise-adaptive GPU denoise of the working image, when the decision engine asked for it. */
  private async restore(_gen: number) {
    const s = this.s!;
    const gpu = this.gpu;
    const P = this.profiler;
    const { width: W, height: H } = s.work;
    if (!s.decision.plan.denoise) { this.log("denoise: noise below visibility — not needed"); return; }
    this.progress("denoise", "GPU");
    s.denoised = await P.time("GPU denoise", () => denoiseGPU(gpu, s.work.tex, W, H, s.gain, s.report.noise), () => `${W}×${H}`);
    this.log(`GPU denoise: full frame, noise-adaptive (σ mid ${(s.report.noise.mid * 255).toFixed(2)}/255)`);
    this.dropThumb(); // look previews must see the restored image
  }

  private postSeries() {
    const r = this.s?.series;
    this.post({ type: "series", info: r ? { frames: r.used, ref: r.files[r.refIndex].name, noise: r.noise, showing: r.showing } : undefined });
  }

  /**
   * A/B for a merged series: the reference shot alone (developed the same way, with
   * the app's usual denoise when the photo calls for it) or the merge, under the
   * same edits. Swaps the working textures; the analysis is shared.
   */
  async seriesView(single: boolean) {
    const s = this.s, r = s?.series;
    if (!s || !r || (r.showing === "single") === single) return;
    if (single && !r.single) {
      this.progress("series single");
      const { developSingle } = await import("../burst/burst.ts");
      const w = await developSingle(this.gpu, r.files[r.refIndex], r.factor);
      if (this.s !== s) { this.gpu.release(w.tex); return; }
      let dn = w.tex;
      if (s.decision.plan.denoise) {
        const noise = noiseProfile(await measureBlocks(this.gpu, w.tex, w.width, w.height, s.gain, 0.02));
        dn = await denoiseGPU(this.gpu, w.tex, w.width, w.height, s.gain, noise);
        const m = noise.mid / Math.max(1e-6, s.report.noise.mid);
        this.log(`series A/B: one shot noise ${(noise.mid * 255).toFixed(2)}/255 vs merge ${(s.report.noise.mid * 255).toFixed(2)}/255 (×${m.toFixed(2)})`);
      }
      if (this.s !== s) { this.gpu.release(w.tex, dn); return; }
      r.single = { tex: w.tex, denoised: dn };
    }
    if (r.showing === "merged") r.merged = { tex: s.work.tex, denoised: s.denoised };
    const use = single ? r.single! : r.merged;
    s.work = { ...s.work, tex: use.tex };
    s.denoised = use.denoised;
    r.showing = single ? "single" : "merged";
    this.dropThumb();
    await this.makeProxy();
    this.postSeries();
    this.requestRender(true);
  }

  private async makeProxy() {
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

  private releaseProxy(s: Session) {
    const d = s.draft;
    s.draft = undefined;
    if (d) { if (d.denoised !== d.base) this.gpu.release(d.denoised); this.gpu.release(d.base); }
    const p = s.proxy;
    s.proxy = undefined;
    if (!p || !p.owned) return;
    if (p.denoised !== p.base) this.gpu.release(p.denoised);
    this.gpu.release(p.base);
  }

  private renderSource(full: boolean): RenderSource {
    const s = this.s!;
    if (full || !s.proxy) return { base: s.work.tex, denoised: s.denoised, width: s.work.width, height: s.work.height, fullWidth: s.work.width, skin: s.skin };
    return { base: s.proxy.base, denoised: s.proxy.denoised, width: s.proxy.w, height: s.proxy.h, fullWidth: s.work.width, skin: s.skin };
  }

  private effectiveParams(): Params {
    const s = this.s!;
    return this.before ? this.cameraParams() : s.params;
  }

  /** "Before": camera rendering only — exposure/WB from the camera, tone curve, nothing adaptive, no edits. */
  private cameraParams(): Params {
    const s = this.s!;
    const p = structuredClone(s.params);
    const e = p.enable;
    e.denoise = false; e.localTone = false; e.semantic = false; e.dehaze = false; e.sharpen = false; e.dof = false; e.curves = false;
    p.exposure = 0;
    p.wb = { temp: s.work.camera?.temp ?? 6504, tint: s.work.camera?.tint ?? 0 };
    p.tone = { highlights: 0, shadows: 0, whites: 0, blacks: 0, contrast: 0, rolloff: 0.5 };
    p.color = { saturation: 0, vibrance: 0 };
    if (p.vignette) p.vignette = { ...p.vignette, amount: 0 };
    if (p.grain) p.grain = { ...p.grain, amount: 0 };
    p.profile = neutralProfile();
    return p;
  }

  private wbFor(p: Params): number[] {
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
    const src = draft && s.proxy ? await this.draftSource() : this.renderSource(false);
    if (s.draft) this.scheduleDraftRelease(s);
    const dof = p.enable.dof && p.dof.strength > 0;
    const r = await this.renderer.render(src, s.maps, p, { wb: this.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38", debugView: this.view, region: this.region, zoneRange: this.zoneRange(), draft }, dof);
    // Histograms for the curve boxes (the edit as rendered; not for "before" or debug views),
    // computed after the preview is on its way so they never delay it.
    const wantHist = final && !draft && !this.before && this.view === 0;
    const wantCalib = final && !draft && !this.before && this.view === 0 && !!s.calib && (s.calib.rounds < 2 || !s.calib.black) && Math.abs(p.exposure - s.decision.params.exposure) < 1e-6;
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
    const calib = wantCalib ? renderedQuantiles(new Uint8Array(data)) : undefined; // read before `data` is transferred
    if (this.display) this.post({ type: "preview", width: src.width, height: src.height, space: "p3", final, ms: performance.now() - t0 });
    else this.post({ type: "preview", width: src.width, height: src.height, data, space: "p3", final, ms: performance.now() - t0 }, [data]);
    if (calib !== undefined && s.calib) {
      // The display model misjudges some scenes (backlight, night): correct on what was rendered.
      const ours = calib[MEDIAN], ref = s.calib.ref[MEDIAN];
      if (s.calib.rounds < 2 && Math.abs(ours - ref) > 6 / 255) {
        s.calib.rounds++;
        const d = Math.log2(Math.max(srgbEotf(ref), 1e-4) / Math.max(srgbEotf(ours), 1e-4));
        const ev = Math.round(Math.min(2.5, Math.max(-1.5, p.exposure + 0.9 * d)) * 100) / 100;
        const note = `measured on the preview: median ${Math.round(ours * 255)}/255 vs the camera's ${Math.round(ref * 255)}/255 → ${ev > 0 ? "+" : ""}${ev} EV`;
        this.log(`exposure calibration: ${note}`);
        s.decision.params.exposure = ev;
        // Not over an edit that arrived while this frame rendered.
        if (s.params.exposure === p.exposure) s.params = { ...s.params, exposure: ev };
        this.post({ type: "exposureCalibrated", exposure: ev, note });
        this.requestRender(true);
      } else {
        // Exposure settled: now the black point, on the same rendering.
        s.calib.rounds = 2;
        s.calib.black = true;
        const points = shadowMatch(calib, s.calib.ref);
        if (points) {
          const note = `black point matched to the camera's rendering: shadows ${calib.slice(0, MEDIAN).map((v) => Math.round(v * 255)).join("/")} → ${s.calib.ref.slice(0, MEDIAN).map((v) => Math.round(v * 255)).join("/")}`;
          this.log(note);
          this.post({ type: "blackPointMatched", points, note });
        }
      }
    }
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

  setView(view: 0 | 1 | 2 | 4 | 5 | 6, before = false, region = 0, range?: [number, number]) {
    this.view = view;
    this.region = region;
    this.viewRange = range;
    this.before = before;
    this.requestRender(true);
  }

  setPreviewSize(long: number) {
    this.previewLong = Math.max(512, Math.min(4096, Math.round(long)));
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

  /**
   * Luminance histograms (32 bins of log2 scene luminance, −14 … +4 EV, the same
   * binning as the region statistics) for the near / middle / far bands, soft-
   * weighted as the renderer blends them. From the guide-resolution image the
   * refinement keeps, so it costs one small readback.
   */
  private async depthBandHistograms(s: Session, b1: number, b2: number) {
    const m = s.maps, d = s.distCPU!;
    const lin = new Float32Array(m.w * m.h * 4);
    halvesToFloats(new Uint16Array(await this.gpu.readTexture(m.lin, 0, 0, m.w, m.h, 8)), lin);
    const smooth = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
    const f = 0.06;
    const H = [new Array(32).fill(0), new Array(32).fill(0), new Array(32).fill(0)];
    const W = [0, 0, 0];
    for (let i = 0; i < m.w * m.h; i++) {
      const Y = 0.2627 * lin[i * 4] + 0.678 * lin[i * 4 + 1] + 0.0593 * lin[i * 4 + 2];
      const bin = Math.min(31, Math.max(0, Math.floor(((Math.log2(Math.max(Y, 1e-7)) + 14) / 18) * 32)));
      const dd = d.data[i];
      const wn = 1 - smooth(b1 - f, b1 + f, dd), wf = smooth(b2 - f, b2 + f, dd), wm = Math.max(0, 1 - wn - wf);
      [wn, wm, wf].forEach((w, k) => { H[k][bin] += w; W[k] += w; });
    }
    const n = m.w * m.h;
    const band = (k: number) => ({ hist: H[k].map((v: number) => v / (W[k] || 1)), area: W[k] / n });
    return { near: band(0), middle: band(1), far: band(2) };
  }

  /** Keeps a CPU copy of the refined distance map for tap-to-focus. */
  private async cacheDistance() {
    const s = this.s!;
    const m = s.maps;
    const raw = new Float32Array(await this.gpu.readTexture(m.masks[2], 0, 0, m.w, m.h, 16));
    const d = new Float32Array(m.w * m.h);
    for (let i = 0; i < d.length; i++) d[i] = raw[i * 4 + 3];
    s.distCPU = { w: m.w, h: m.h, data: d };
  }

  focusAt(x: number, y: number): number | undefined {
    const d = this.s?.distCPU;
    if (!d) return undefined;
    // Median over a small neighbourhood: a tap is imprecise on a phone.
    const cx = Math.round(x * (d.w - 1)), cy = Math.round(y * (d.h - 1));
    const v: number[] = [];
    for (let j = -3; j <= 3; j++) for (let i = -3; i <= 3; i++) {
      const xx = Math.min(d.w - 1, Math.max(0, cx + i)), yy = Math.min(d.h - 1, Math.max(0, cy + j));
      v.push(d.data[yy * d.w + xx]);
    }
    v.sort((a, b) => a - b);
    return v[v.length >> 1];
  }

  /**
   * The depth range of the object under a tap: grown from the tap across the
   * refined depth map through smooth depth changes (≤ 0.02 between neighbours)
   * within the same semantic region, so the whole object — not just the tapped
   * spot — stays sharp. Continuous surfaces (ground, floor, sky, terrain, or
   * anything over ≈ 35 % of the frame) keep a thin slice: they run from near to
   * far, and "the object" would switch the blur off.
   */
  focusRangeAt(x: number, y: number): { dist: number; range: [number, number] } | undefined {
    const s = this.s, d = s?.distCPU;
    const d0 = this.focusAt(x, y);
    if (!s || !d || d0 === undefined) return undefined;
    return { dist: d0, range: objectDepthRange(d, s.scene.seg, x, y, d0) };
  }

  importLook(name: string, text: string) {
    this.renderer.addLook(parseCube(text, name));
    return this.looks();
  }

  async export(format: ExportFormat, quality: number, space: "srgb" | "p3", stripRows = 512): Promise<{ blob: Blob; name: string; ms: number }> {
    const s = this.s;
    if (!s) throw new Error("No photo open");
    const t0 = performance.now();
    const gpu = this.gpu;
    const src = this.renderSource(true);
    const p = s.params;
    await this.ensureSelections(s, p);
    const W = src.width, H = src.height;
    const base = s.name.replace(/\.[^.]+$/, "");
    const P = this.profiler;
    const o = { wb: this.wbFor(p), gain: s.gain, lightLinear: s.lightLinear };
    const dof = p.enable.dof && p.dof.strength > 0;
    // Full-resolution exports render in strips: peak extra GPU memory stays at a
    // few tens of MB instead of several full-frame textures.
    const STRIP = Math.max(64, Math.round(stripRows));
    const strips = async (each: (y0: number, rows: number) => Promise<void>) => {
      for (let y0 = 0; y0 < H; y0 += STRIP) {
        this.progress("export", `rendering ${Math.round((y0 / H) * 100)}%`, y0 / H);
        await each(y0, Math.min(STRIP, H - y0));
      }
    };
    try {
      if (format === "dng") {
        const f = new Float32Array(W * H * 4);
        await P.time("export render (linear)", () => strips(async (y0, rows) => {
          const r = await this.renderer.renderLinear(src, s.maps, p, { ...o, output: "p3f16" }, { y0, rows });
          await this.readHalfRows(r.tex, r.top, W, rows, f.subarray(y0 * W * 4, (y0 + rows) * W * 4));
        }));
        this.progress("export", "writing DNG");
        const blob = await P.time("encode DNG", () => encodeLinearDng(f, W, H, s.decoded.meta));
        this.post({ type: "profile", stages: P.stages });
        return { blob, name: `${base}-processed-linear.dng`, ms: performance.now() - t0 };
      }
      if (format === "tiff16") {
        const f = new Float32Array(W * H * 4);
        await P.time("export render (16-bit)", () => strips(async (y0, rows) => {
          const r = await this.renderer.render(src, s.maps, p, { ...o, output: "p3f16" }, dof, { y0, rows });
          await this.readHalfRows(r.tex, r.top, W, rows, f.subarray(y0 * W * 4, (y0 + rows) * W * 4));
        }));
        this.progress("export", "writing TIFF");
        const blob = await P.time("encode TIFF", () => encodeTiff16(f, W, H, s.decoded.meta));
        this.post({ type: "profile", stages: P.stages });
        return { blob, name: `${base}-edit.tif`, ms: performance.now() - t0 };
      }
      if (format === "jpeg-hdr") {
        // SDR image + gain map, strip by strip. The gain map is ½ size (¼ above 24 MP);
        // strips start on multiples of the block size so its rows line up.
        const s2 = W * H > 24e6 ? 4 : 2;
        const stops = p.hdr?.headroom || 2;
        const gw = Math.ceil(W / s2), gh = Math.ceil(H / s2);
        const rgba = new Uint8ClampedArray(W * H * 4);
        const gain = new Uint8ClampedArray(gw * gh * 4);
        const HS = Math.max(64, Math.round(STRIP / s2) * s2);
        await P.time("export render (HDR)", async () => {
          for (let y0 = 0; y0 < H; y0 += HS) {
            this.progress("export", `rendering ${Math.round((y0 / H) * 100)}%`, y0 / H);
            const rows = Math.min(HS, H - y0);
            const r = await this.renderer.render(src, s.maps, { ...p, hdr: { headroom: stops } }, { ...o, output: space === "p3" ? "p38" : "srgb8", hdr: true, gainMap: { scale: s2, stops } }, dof, { y0, rows });
            rgba.set(new Uint8Array(await gpu.readTexture(r.tex, 0, r.top, W, rows, 4)), y0 * W * 4);
            if (r.gm) gain.set(new Uint8Array(await gpu.readTexture(r.gm, 0, 0, r.gmW!, r.gmRows!, 4)), (y0 / s2) * gw * 4);
          }
        }, () => `${W}×${H} + gain map ${gw}×${gh}, +${stops} EV`);
        this.progress("export", "encoding JPEG (HDR)");
        const blob = await P.time("encode JPEG (HDR)", () => encodeGainMapJpeg(rgba, W, H, gain, gw, gh, stops, space, quality, s.decoded.meta));
        this.post({ type: "profile", stages: P.stages });
        return { blob, name: `${base}-edit-hdr.jpg`, ms: performance.now() - t0 };
      }
      const rgba = new Uint8ClampedArray(W * H * 4);
      await P.time("export render", () => strips(async (y0, rows) => {
        const r = await this.renderer.render(src, s.maps, p, { ...o, output: space === "p3" ? "p38" : "srgb8" }, dof, { y0, rows });
        rgba.set(new Uint8Array(await gpu.readTexture(r.tex, 0, r.top, W, rows, 4)), y0 * W * 4);
      }), () => `${W}×${H} in ${Math.ceil(H / STRIP)} strips`);
      this.progress("export", `encoding ${format.toUpperCase()}`);
      const blob = await P.time(`encode ${format}`, () => format === "heic" ? encodeHeic(rgba, W, H, space, quality) : encodeJpeg(rgba, W, H, space, quality, s.decoded.meta));
      this.post({ type: "profile", stages: P.stages });
      return { blob, name: `${base}-edit.${format === "heic" ? "heic" : "jpg"}`, ms: performance.now() - t0 };
    } finally {
      // Strip targets and readback buffers are export-sized; the next preview re-creates its own.
      this.renderer.releaseTargets();
      this.gpu.flushStaging();
    }
  }

  private async readHalfRows(tex: GPUTexture, top: number, W: number, rows: number, out: Float32Array<ArrayBuffer>) {
    const half = new Uint16Array(await this.gpu.readTexture(tex, 0, top, W, rows, 8));
    halvesToFloats(half, out);
  }



  private summary(name: string): Summary {
    const s = this.s!;
    const m = s.decoded.meta;
    const src = s.decoded.source;
    const meta: Record<string, string | number> = {};
    if (m.make) meta.camera = `${m.make} ${m.model ?? ""}`.trim();
    if (m.iso) meta.ISO = m.iso;
    if (m.exposureTime) meta.shutter = m.exposureTime >= 1 ? `${m.exposureTime}s` : `1/${Math.round(1 / m.exposureTime)}s`;
    if (m.fNumber) meta.aperture = `f/${m.fNumber.toFixed(1)}`;
    if (m.focalLength) meta.focal = `${m.focalLength.toFixed(1)}mm${m.focalLength35 ? ` (${m.focalLength35}mm eq.)` : ""}`;
    if (s.work.camera) { meta["as-shot WB"] = `${Math.round(s.work.camera.temp)}K / ${s.work.camera.tint.toFixed(1)}`; meta["baseline exposure"] = `${s.work.camera.baselineExposure.toFixed(2)} EV`; }
    return {
      file: name,
      format: s.decoded.format,
      source: src.kind === "rgb" ? `display-referred RGB (${src.decoder})` : src.isProRaw ? "Apple ProRAW (LinearRaw)" : src.kind === "bayer" ? "Bayer RAW" : "LinearRaw DNG",
      width: src.width,
      height: src.height,
      working: { width: s.work.width, height: s.work.height, factor: s.work.factor },
      meta,
      coverage: Object.fromEntries(Object.entries(s.scene.coverage).map(([k, v]) => [k, Math.round(v * 1000) / 10])),
    };
  }

  profile() { return this.profiler.stages; }

  // ------------------------------------------------------------------ looks

  private thumb?: { base: GPUTexture; denoised: GPUTexture; w: number; h: number; long: number };

  private dropThumb() {
    const t = this.thumb;
    this.thumb = undefined;
    if (t) this.gpu.release(t.base, t.denoised === t.base ? undefined : t.denoised);
  }

  private async ensureThumb(long: number) {
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

  /** Renders the current photo at thumbnail size through each profile (same GPU path as the preview). */
  async thumbnails(profiles: LookProfile[], long: number) {
    const s = this.s;
    if (!s) return [];
    const t = await this.ensureThumb(long);
    const src: RenderSource = { base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width };
    const items: Array<{ id: string; width: number; height: number; data: ArrayBuffer }> = [];
    const t0 = performance.now();
    for (const raw of profiles) {
      const prof = normalizeProfile(raw);
      const p: Params = { ...s.params, profile: prof, enable: { ...s.params.enable, dof: false, sharpen: false, lut: true } };
      const r = await this.renderer.render(src, s.maps, p, { wb: this.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38" }, false);
      items.push({ id: prof.id, width: t.w, height: t.h, data: await this.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4) });
    }
    this.log(`look previews: ${profiles.length} profiles at ${t.w}×${t.h} in ${(performance.now() - t0).toFixed(0)} ms`);
    return items;
  }

  /** The technical rendering (no creative profile) at a small size, P3-encoded RGBA8. */
  private async technicalPixels(long = 384) {
    const s = this.s!;
    const t = await this.ensureThumb(long);
    const p: Params = { ...s.params, profile: neutralProfile(), enable: { ...s.params.enable, dof: false, lut: false } };
    const r = await this.renderer.render({ base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width }, s.maps, p, { wb: this.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false }, false);
    const data = new Uint8Array(await this.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4));
    return { data, w: t.w, h: t.h };
  }

  /** Weight = 1 − P(people) − P(sky): the global look statistics exclude regions that are matched (sky) or protected (people) separately. */
  private static generalWeights(seg: SceneMaps["seg"], w: number, h: number): Float32Array {
    const p = Engine.groupWeights(seg, w, h, GROUPS.indexOf("person"));
    const s = Engine.groupWeights(seg, w, h, GROUPS.indexOf("sky"));
    for (let i = 0; i < p.length; i++) p[i] = Math.max(0.02, 1 - p[i] - s[i]);
    return p;
  }

  /** Per-pixel weights of a semantic group, sampled from network-resolution probabilities. */
  private static groupWeights(seg: SceneMaps["seg"], w: number, h: number, g: number, invert = false): Float32Array {
    const out = new Float32Array(w * h);
    const plane = seg.width * seg.height;
    for (let y = 0; y < h; y++) {
      const sy = Math.min(seg.height - 1, Math.floor(((y + 0.5) / h) * seg.height));
      for (let x = 0; x < w; x++) {
        const sx = Math.min(seg.width - 1, Math.floor(((x + 0.5) / w) * seg.width));
        const v = seg.probs[g * plane + sy * seg.width + sx];
        out[y * w + x] = invert ? 1 - v : v;
      }
    }
    return out;
  }

  private static regionColors(rgba: Uint8Array, w: number, h: number, seg: SceneMaps["seg"]): RegionColors {
    const out: RegionColors = {};
    for (const g of ["sky", "vegetation", "water", "building", "terrain", "ground"] as Group[]) {
      const gi = GROUPS.indexOf(g);
      const wts = Engine.groupWeights(seg, w, h, gi);
      let mass = 0;
      for (const v of wts) mass += v;
      if (mass / (w * h) < 0.02) continue;
      const st = analyseColors(rgba, wts, 3);
      out[g] = { ...st, n: Math.round(mass) };
    }
    return out;
  }

  /**
   * What is under a tap, for a mask built from it: the region (network probabilities,
   * 3×3 around the tap), the distance and the tapped object's depth range (as focus
   * points use), and the colour before the layers (median of 5×5 at 384 px) in OkLab.
   */
  async pickAt(x: number, y: number, layer?: number, object = false): Promise<PickInfo | undefined> {
    const s = this.s;
    const f = this.focusRangeAt(x, y);
    if (!s || !f) return undefined;
    const seg = s.scene.seg, plane = seg.width * seg.height;
    const cx = Math.round(x * (seg.width - 1)), cy = Math.round(y * (seg.height - 1));
    const score = new Float64Array(GROUPS.length);
    for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
      const k = Math.min(seg.height - 1, Math.max(0, cy + j)) * seg.width + Math.min(seg.width - 1, Math.max(0, cx + i));
      for (let g = 0; g < GROUPS.length; g++) score[g] += seg.probs[g * plane + k] / 9;
    }
    let best = 0;
    for (let g = 1; g < GROUPS.length; g++) if (score[g] > score[best]) best = g;
    let region: Region = GROUPS[best];
    // People: Apple's skin matte (ProRAW) tells skin from clothes.
    const skin = s.decoded.masks?.find((m) => m.kind === "skin");
    if (region === "person" && skin && skin.data[Math.min(skin.height - 1, Math.round(y * (skin.height - 1))) * skin.width + Math.min(skin.width - 1, Math.round(x * (skin.width - 1)))] > 127) region = "skin";
    // The colour before the layers (the layers' masks compare against exactly that).
    const t = await this.ensureThumb(384);
    const p: Params = { ...s.params, layers: [], enable: { ...s.params.enable, dof: false } };
    const r = await this.renderer.render({ base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width }, s.maps, p, { wb: this.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false }, false);
    const px = new Uint8Array(await this.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4));
    const tx = Math.round(x * (t.w - 1)), ty = Math.round(y * (t.h - 1));
    const labs: Array<[number, number, number]> = [];
    for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++) {
      const k = (Math.min(t.h - 1, Math.max(0, ty + j)) * t.w + Math.min(t.w - 1, Math.max(0, tx + i))) * 4;
      const lin = mulVec(P3_TO_SRGB, [srgbEotf(px[k] / 255), srgbEotf(px[k + 1] / 255), srgbEotf(px[k + 2] / 255)]);
      labs.push(linSrgbToOklab(lin));
    }
    const med = (c: 0 | 1 | 2) => labs.map((l) => l[c]).sort((a, b) => a - b)[labs.length >> 1];
    // Surfaces running from near to far (ground, sky, a wall over a third of the frame) get
    // only a thin depth slice for focus; as a mask, "this object" is then the whole region.
    const range: [number, number] = f.range[1] - f.range[0] <= 0.0401 ? [0, 1] : f.range;
    const info: PickInfo = { x, y, region, prob: score[GROUPS.indexOf(region === "skin" ? "person" : region)], dist: f.dist, range, color: [med(0), med(1), med(2)] };
    if (layer !== undefined && layer >= 0) Object.assign(info, await this.maskUnderTap(s, x, y, layer, object));
    return info;
  }

  /**
   * Is the tap on something the layer's mask already covers? Read from the mask
   * itself (view 7 at 384 px, median of 3×3). If so and the tap selects objects:
   * is the tapped object one of the layer's own selections (its mask overlaps the
   * new tap's by IoU > 0.5)? Then that selection is what a tap removes.
   */
  private async maskUnderTap(s: Session, x: number, y: number, layer: number, object: boolean): Promise<{ inMask: number; sameAs?: string }> {
    await this.ensureSelections(s, s.params);
    const t = await this.ensureThumb(384);
    const r = await this.renderer.render({ base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width }, s.maps, s.params,
      { wb: this.wbFor(s.params), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false, debugView: 7, region: layer }, false);
    const px = new Uint8Array(await this.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4));
    const tx = Math.round(x * (t.w - 1)), ty = Math.round(y * (t.h - 1));
    const v: number[] = [];
    for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) v.push(px[(Math.min(t.h - 1, Math.max(0, ty + j)) * t.w + Math.min(t.w - 1, Math.max(0, tx + i))) * 4 + 1]);
    let inMask = v.sort((a, b) => a - b)[4] / 255;
    const st = s.sel;
    if (!object || !st) return { inMask };
    // An object tap is judged by the whole object, not the pixel under the finger (thin
    // things and soft mask edges read half-selected there): how much of the tapped
    // object the mask already covers.
    const { w: gw, h: gh } = s.maps;
    const tapObj = await this.selectionMask(s, st, { kind: "select", points: [[x, y, 1]], invert: false, feather: 1 });
    let n = 0, covered = 0;
    for (let gy = 0; gy < gh; gy += 2) for (let gx = 0; gx < gw; gx += 2) {
      if (tapObj[gy * gw + gx] <= 127) continue;
      n++;
      const k = (Math.min(t.h - 1, Math.round((gy / (gh - 1)) * (t.h - 1))) * t.w + Math.min(t.w - 1, Math.round((gx / (gw - 1)) * (t.w - 1)))) * 4 + 1;
      covered += px[k] / 255;
    }
    if (n > 8) inMask = covered / n;
    if (inMask <= 0.5) return { inMask };
    // The layer's own added selections that cover the tap.
    const L = liveLayers(s.params.layers ?? [], s.params.autoCurves ?? 1, s.params.enable)[layer];
    const w = gw, h = gh;
    const at = (mask: Uint8Array) => mask[Math.min(h - 1, Math.round(y * (h - 1))) * w + Math.min(w - 1, Math.round(x * (w - 1)))];
    const cands = L ? [L.mask, ...(L.mask.parts ?? []).filter((q) => q.op === "add")].filter((m) => m.kind === "select" && !m.invert && st.cache.get(selectKey(m)) && at(st.cache.get(selectKey(m))!) > 127) : [];
    if (!cands.length) return { inMask };
    const tap = tapObj;
    let best: string | undefined, bestIou = 0.5;
    for (const m of cands) {
      const c = st.cache.get(selectKey(m))!;
      let inter = 0, uni = 0;
      for (let i = 0; i < c.length; i++) { const a = c[i] > 127, b = tap[i] > 127; if (a && b) inter++; if (a || b) uni++; }
      const iou = uni ? inter / uni : 0;
      if (iou > bestIou) { bestIou = iou; best = selectKey(m); }
    }
    return { inMask, sameAs: best };
  }

  /**
   * Makes sure every selection the layers use has its mask in the selection
   * texture (encoding the photo on first use: MobileSAM, a few seconds on a
   * phone), then points the renderer at it. Cheap when nothing changed.
   */
  private async ensureSelections(s: Session, p: Params) {
    const want: MaskShape[] = [];
    for (const l of p.layers ?? []) for (const m of [l.mask, ...(l.mask.parts ?? [])]) if (m.kind === "select" && m.points?.length) want.push(m);
    const keys = [...new Set(want.map(selectKey))];
    const sel = s.sel;
    if (!keys.length && !sel?.keys.length) return;
    if (sel && keys.length === sel.keys.length && keys.every((k, i) => k === sel.keys[i])) return;
    const st: Selections = (s.sel ??= { sam: new SamSelector(this.base, s.work.width, s.work.height, phoneForced()), cache: new Map(), failed: new Set(), keys: [], version: 0 });
    for (const m of want) {
      const k = selectKey(m);
      if (st.cache.has(k) || st.failed.has(k)) continue;
      try { st.cache.set(k, await this.selectionMask(s, st, m)); }
      catch (e) { st.failed.add(k); this.log(`selection unavailable: ${e instanceof Error ? e.message : e}`); }
    }
    // Keep the last 16 masks (≈ 0.4 MB each): undo and switching readings are instant.
    for (const k of [...st.cache.keys()]) if (st.cache.size > 16 && !keys.includes(k)) st.cache.delete(k);
    const ready = keys.filter((k) => st.cache.has(k));
    const { w, h } = s.maps;
    this.gpu.release(st.tex);
    st.tex = this.gpu.tex("selection", w, h, "r8unorm", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, "2d", Math.max(1, ready.length));
    ready.forEach((k, z) => this.gpu.device.queue.writeTexture({ texture: st.tex!, origin: { x: 0, y: 0, z } }, st.cache.get(k)! as Uint8Array<ArrayBuffer>, { bytesPerRow: w, rowsPerImage: h }, { width: w, height: h }));
    st.keys = keys;
    st.version++;
    this.renderer.selection = { tex: st.tex, slotOf: (m) => ready.indexOf(selectKey(m)), version: st.version };
  }

  /** One selection's mask at the guide resolution: SAM's reading for its taps, snapped to the photo's edges. */
  private async selectionMask(s: Session, st: Selections, m: MaskShape): Promise<Uint8Array> {
    const { w, h } = s.maps;
    if (!st.sam.encoded) {
      this.progress("selection", "encode");
      await st.sam.encode(async () => {
        // The photo as SAM sees it: display-encoded, long side 1024, HWC 0…255.
        const [pw, ph] = st.sam.dims;
        const t = await downsample(this.gpu, s.work.tex, s.work.width, s.work.height, pw, ph, true, s.gain, "sam.input");
        const px = halvesToFloats(new Uint16Array(await this.gpu.readTexture(t, 0, 0, pw, ph, 8)));
        this.gpu.release(t);
        const img = new Float32Array(pw * ph * 3);
        for (let i = 0, j = 0; i < pw * ph; i++, j += 4) for (let c = 0; c < 3; c++) img[i * 3 + c] = 255 * Math.min(1, Math.max(0, px[j + c]));
        return img;
      });
      this.progress("");
    }
    if (!st.guide) {
      const g = halvesToFloats(new Uint16Array(await this.gpu.readTexture(s.maps.guide, 0, 0, w, h, 8)));
      st.guide = new Float32Array(w * h);
      for (let i = 0; i < w * h; i++) st.guide[i] = Math.min(1, Math.max(0, 0.2126 * g[i * 4] + 0.7152 * g[i * 4 + 1] + 0.0722 * g[i * 4 + 2]));
    }
    this.progress("selection", "decode");
    const { low, iou } = await st.sam.decode(m.points!);
    this.progress("selection", "filter");
    // No level chosen: SAM's own pick, its most confident of the three readings.
    const k = m.level === undefined ? [1, 2, 3].reduce((a, b) => (iou[b] > iou[a] ? b : a)) : levelsByArea(low)[m.level];
    return selectionMask(low, k, st.sam.dims, w, h, st.guide);
  }

  /**
   * Check: the edit as exported and the camera's rendering, both at 1024 px, looked
   * at for technical mistakes (src/analysis/check.ts). Returns the findings and the
   * edit's pixels (to show where a finding is).
   */
  async check(): Promise<{ items: CheckItem[]; rgba: Uint8Array; w: number; h: number }> {
    const s = this.s;
    if (!s) throw new Error("No photo open");
    await this.ensureSelections(s, s.params);
    const t = await this.ensureThumb(1024);
    const src = { base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width };
    const read = async (p: Params, dof: boolean) => {
      const r = await this.renderer.render(src, s.maps, p, { wb: this.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false }, dof);
      return new Uint8Array(await this.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4));
    };
    const p = s.params;
    const final = await read(p, p.enable.dof && p.dof.strength > 0);
    const before = await read(this.cameraParams(), false);
    const seg = { ...s.scene.seg, groups: GROUPS };
    const [{ checkPhoto }] = await checkCode();
    // The camera's own rendering (Apple's JPEG inside a ProRAW DNG): what "camera" means for
    // the whole-photo figures. Our plain development of the RAW is far paler than what the
    // phone shows, and would make any edit look "twice as colourful".
    if (s.cameraRef === undefined) s.cameraRef = /\.dng$/i.test(s.name) ? (await embeddedPreviewPixels(s.file, 256, isMobile() ? 24 : Infinity)) ?? null : null;
    const camera = s.cameraRef ?? undefined;
    const items = checkPhoto({ final: { rgba: final, w: t.w, h: t.h }, before: { rgba: before, w: t.w, h: t.h }, camera, seg, scene: { ev: sceneEV(s.decoded.meta) } });
    // The fixes are worked out next, as their own job (solveCheckFixes): the findings show at once.
    this.lastCheck = { s, items: items.map((i) => ({ ...i, mask: undefined })), seg, scene: { ev: sceneEV(s.decoded.meta) }, camera };
    return { items, rgba: final, w: t.w, h: t.h };
  }
  private lastCheck?: { s: Session; items: CheckItem[]; seg: NonNullable<CheckInput["seg"]>; scene: CheckInput["scene"]; camera?: CheckInput["camera"] };

  /** The fixes for the last check's findings, each reported as soon as it is worked out. */
  async solveCheckFixes(onFix: (id: CheckItem["id"], fix: FixChange[], partial: boolean) => void) {
    const c = this.lastCheck;
    this.lastCheck = undefined;
    if (!c || c.s !== this.s) return; // another photo since
    try {
      await this.solveFixes(c.s, c.items, c.seg, c.scene, c.camera, (it) => onFix(it.id, it.fix!, !!it.fixPartial));
    } catch (e) {
      if (!(e instanceof StaleCheck)) throw e; // edited or another photo: the answers would be for old settings
    }
  }

  /**
   * For every finding that is not fine: which control, set to what, fixes it. Each
   * finding measures how far it is from a comfortable result (CheckItem.err, > 0 =
   * needs fixing); one control at a time (in the order leversFor gives) is bisected,
   * on renders at 768 px, to where that crosses zero — a comfortable value, not the
   * edge of "fine". If no control gets there, the layer whose opacity does; failing
   * that, the control that helps most (a partial fix).
   *
   * Each render is its own short exclusive job, so slider edits and previews are
   * served in between; an edit (new params) or another photo stops the solving.
   */
  private async solveFixes(s: Session, items: CheckItem[], seg: NonNullable<CheckInput["seg"]>, scene: CheckInput["scene"], camera: CheckInput["camera"], onFix: (it: CheckItem) => void) {
    const todo = items.filter((i) => i.level !== "ok" && i.err !== undefined);
    if (!todo.length) return;
    const [{ checkPhoto, planesOf }, { getPath, leversFor, setPath, stepOf }] = await checkCode();
    const t0 = performance.now();
    let renders = 0, size = "";
    const base = s.params;
    const render = (p: Params) => this.exclusive(async () => {
      if (this.s !== s || s.params !== base) throw new StaleCheck();
      renders++;
      const t = await this.ensureThumb(768);
      size = `${t.w}×${t.h}`;
      const src = { base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width };
      const r = await this.renderer.render(src, s.maps, p, { wb: this.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false }, p.enable.dof && p.dof.strength > 0);
      return { rgba: new Uint8Array(await this.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4)), w: t.w, h: t.h };
    });
    const beforeImg = await render(this.cameraParams());
    const beforePlanes = planesOf(beforeImg);
    const errWith = async (id: CheckItem["id"], p: Params) =>
      checkPhoto({ final: await render(p), before: beforeImg, seg, scene, camera }, id, beforePlanes).find((i) => i.id === id)?.err ?? -1;
    for (const it of todo) {
      const e0 = await errWith(it.id, base);
      if (e0 <= 0) continue; // comfortable at this size already
      let best: { change: FixChange; err: number } | undefined;
      for (const lv of leversFor(it, base)) {
        const v0 = getPath(base, lv.path);
        const step = stepOf(lv.path);
        const at = (v: number) => { const p = structuredClone(base); setPath(p, lv.path, v); return errWith(it.id, p); };
        // Walk from the current value towards the end in 6 steps to the first comfortable
        // value (a two-sided target like exposure is passed, not just approached), then
        // bisect between it and the step before.
        let prev = v0, prevE = e0, before2 = v0, found: number | undefined, minE = e0, minV = v0;
        for (let k = 1; k <= 6; k++) {
          const v = v0 + ((lv.bound - v0) * k) / 6;
          const e = await at(v);
          if (e < minE) { minE = e; minV = v; }
          if (e <= 0) { found = v; break; }
          if (e > prevE && k > 1) {
            // Worse again after getting better: a narrow comfortable window may lie between
            // (stepped over). Look for the best value in (before2, v) by golden-section search.
            let a = before2, b = v;
            for (let g = 0; g < 5 && Math.abs(b - a) > step; g++) {
              const m1 = b - (b - a) * 0.618, m2 = a + (b - a) * 0.618;
              const [e1, e2] = [await at(m1), await at(m2)];
              if (e1 < minE) { minE = e1; minV = m1; }
              if (e2 < minE) { minE = e2; minV = m2; }
              if (Math.min(e1, e2) <= 0) break;
              if (e1 < e2) b = m2; else a = m1;
            }
            if (minE <= 0) { found = minV; prev = before2; }
            break;
          }
          before2 = prev; prev = v; prevE = e;
        }
        if (found === undefined) {
          // Not all the way: remember the value that helps most.
          if (minE < e0 * 0.6 && (!best || minE < best.err)) best = { change: { path: lv.path, from: v0, to: Math.round((Math.round(minV / step) * step) * 1000) / 1000 }, err: minE };
          continue;
        }
        let lo = prev, hi = found;
        for (let k = 0; k < 5 && Math.abs(hi - lo) > step; k++) { const mid = (lo + hi) / 2; if ((await at(mid)) <= 0) hi = mid; else lo = mid; }
        let to = Math.round(hi / step) * step;
        if (Math.sign(to - hi) === Math.sign(v0 - lv.bound)) to -= Math.sign(v0 - lv.bound) * step; // round towards the fixing side
        it.fix = [{ path: lv.path, from: v0, to: Math.round(to * 1000) / 1000 }];
        break;
      }
      if (it.fix) { onFix(it); continue; }
      // No control does it: the layer that does (its opacity; 0 = hide it).
      const live = liveLayers(base.layers ?? [], base.autoCurves ?? 1, base.enable);
      for (const L of [...live].reverse()) {
        const at = (o: number) => { const p = structuredClone(base); p.layers!.find((x) => x.id === L.id)!.opacity = o; return errWith(it.id, p); };
        if ((await at(0)) > 0) continue;
        let lo = L.opacity, hi = 0;
        for (let k = 0; k < 6; k++) { const mid = (lo + hi) / 2; if ((await at(mid)) <= 0) hi = mid; else lo = mid; }
        it.fix = [{ layer: L.id, name: L.name, from: L.opacity, to: Math.floor(hi * 20) / 20 }];
        break;
      }
      if (!it.fix && best) { it.fix = [best.change]; it.fixPartial = true; }
      if (it.fix) onFix(it);
    }
    this.log(`check fixes: ${todo.length} findings, ${renders} renders at ${size}, ${Math.round(performance.now() - t0)} ms`);
  }

  /**
   * Where the light probably is (a lens flare's default): the brightest spot in the
   * upper 60 % of the photo as edited, on a 256 px render, smoothed so a single
   * glint does not win over the sun.
   */
  async brightestPoint(): Promise<{ x: number; y: number }> {
    const s = this.s;
    if (!s) return { x: 0.3, y: 0.2 };
    const t = await this.ensureThumb(256);
    const p = s.params;
    const r = await this.renderer.render({ base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width }, s.maps, p, { wb: this.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false }, false);
    const px = new Uint8Array(await this.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4));
    const { w, h } = t, R = 4;
    let best = -1, bx = 0.3, by = 0.2;
    for (let y = R; y < Math.floor(h * 0.6); y += 2) for (let x = R; x < w - R; x += 2) {
      let sum = 0;
      for (let j = -R; j <= R; j += 2) for (let i = -R; i <= R; i += 2) { const o = ((y + j) * w + x + i) * 4; sum += Math.max(px[o], px[o + 1], px[o + 2]); }
      if (sum > best) { best = sum; bx = x / (w - 1); by = y / (h - 1); }
    }
    return { x: Math.round(bx * 1000) / 1000, y: Math.round(by * 1000) / 1000 };
  }

  async palette(): Promise<ColorStats> {
    const s = this.s;
    if (!s) throw new Error("No photo open");
    const px = await this.technicalPixels(384);
    return analyseColors(px.data, undefined, 7);
  }

  /** Reference image → editable profile (create) or a profile matching the current photo toward it (match). */
  async reference(file: File, mode: "create" | "match", amount: number): Promise<{ profile: LookProfile; reference: ColorStats; message: string }> {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const dec = await decodeFile(bytes, file.name, file.type);
    try {
      if (dec.source.kind !== "rgb") throw new Error("Use a rendered image (JPEG, HEIC, PNG) as the reference, not a RAW file.");
      const src = dec.source;
      const k = Math.min(1, 384 / Math.max(src.width, src.height));
      const w = Math.max(8, Math.round(src.width * k)), h = Math.max(8, Math.round(src.height * k));
      const cv = new OffscreenCanvas(w, h);
      const ctx = cv.getContext("2d", { colorSpace: "display-p3" }) as OffscreenCanvasRenderingContext2D;
      if ("close" in src.pixels) ctx.drawImage(src.pixels, 0, 0, w, h);
      else {
        const full = new OffscreenCanvas(src.width, src.height);
        const fctx = full.getContext("2d", { colorSpace: src.colorSpace === "display-p3" ? "display-p3" : "srgb" }) as OffscreenCanvasRenderingContext2D;
        fctx.putImageData(new ImageData(new Uint8ClampedArray(src.pixels.data.buffer as ArrayBuffer), src.width, src.height), 0, 0);
        ctx.drawImage(full, 0, 0, w, h);
      }
      const refPx = new Uint8Array(ctx.getImageData(0, 0, w, h, { colorSpace: "display-p3" }).data.buffer);
      // Segment the reference so regions are compared with the same regions.
      const f = new Float32Array(w * h * 4);
      for (let i = 0; i < w * h * 4; i++) f[i] = refPx[i] / 255;
      const refScene = await analyseScene(this.neural, { rgba: f, width: w, height: h }, (st) => this.progress("reference " + st), false);
      const refStats = analyseColors(refPx, Engine.generalWeights(refScene.seg, w, h), 7);
      const refRegions = Engine.regionColors(refPx, w, h, refScene.seg);
      if (mode === "create" || !this.s) {
        const profile = profileFromReference(refStats, file.name, refRegions);
        return { profile, reference: refStats, message: `Profile built from ${file.name}` };
      }
      const s = this.s;
      const px = await this.technicalPixels(384);
      const srcStats = analyseColors(px.data, Engine.generalWeights(s.scene.seg, px.w, px.h), 7);
      const srcRegions = Engine.regionColors(px.data, px.w, px.h, s.scene.seg);
      const profile = matchProfile(srcStats, refStats, file.name, srcRegions, refRegions, amount);
      return { profile, reference: refStats, message: profile.description ?? "" };
    } finally {
      dec.close();
    }
  }

  get session() { return this.s; }

  /** GPU memory the engine holds now and at most since the last call (MB): the memory guard's per-step numbers. */
  memStats(): { liveMB: number; peakMB: number } {
    const MB = 1048576;
    return { liveMB: +(this.gpu.liveBytes() / MB).toFixed(1), peakMB: +(this.gpu.takeStepPeak() / MB).toFixed(1) };
  }
  get wbCamera(): CameraColor | undefined { return this.s?.work.camera; }
}

/** Gain that puts the 60th-percentile luminance of the analysis image at 0.18 (clamped). */
function exposureGain(rgba: Float32Array): number {
  const n = rgba.length / 4;
  const ys = new Float32Array(n);
  for (let i = 0; i < n; i++) ys[i] = Math.max(0, 0.2627 * rgba[i * 4] + 0.678 * rgba[i * 4 + 1] + 0.0593 * rgba[i * 4 + 2]);
  return exposureGainY(ys);
}

/** exposureGain from linear luminances. */
function exposureGainY(y: Float32Array): number {
  const ys = y.slice().sort();
  const n = ys.length;
  const p60 = ys[Math.floor(n * 0.6)] || 1e-4;
  const p99 = ys[Math.floor(n * 0.99)] || 1;
  let k = 0.18 / Math.max(p60, 1e-5);
  k = Math.min(k, 1.6 / Math.max(p99, 1e-5), 32);
  return Math.max(0.25, k);
}

/** The Check's fixes were being worked out for settings (or a photo) that changed since. */
class StaleCheck extends Error {}
