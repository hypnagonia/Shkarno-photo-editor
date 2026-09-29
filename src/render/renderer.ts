/**
 * Modules: Exposure/Tone Engine, Camera Color Engine (stages B/C),
 * Semantic/Depth Processing, Optional Depth of Field, output transform.
 *
 *   in:  base + denoised working textures (full res or preview proxy),
 *        RefinedMaps, Params
 *   out: rgba8unorm (display/JPEG) or rgba16float (16-bit export) texture
 *
 * Resolution-independent: the same passes render the preview proxy and the
 * full-resolution export, so what is seen is what is saved.
 */
import { Gpu, Uniforms } from "../gpu/gpu.ts";
import toneWgsl from "../gpu/shaders/render_tone.wgsl?raw";
import layersWgsl from "../gpu/shaders/layers.wgsl?raw";
import { TONE_EQ_DETAIL, toneEqActive, toneEqLut } from "../tone/toneEq.ts";
import { contrastEqActive, edgeSigma, levels, stripApron, type ContrastEq } from "../tone/contrastEq.ts";
import ceqWgsl from "../gpu/shaders/render_ceq.wgsl?raw";
import { ATLAS_W, hasBlurLayers, packLayers, RECORD } from "../layers/gpu.ts";
import type { MaskShape } from "../layers/model.ts";
import detailWgsl from "../gpu/shaders/render_detail.wgsl?raw";
import dofWgsl from "../gpu/shaders/render_dof.wgsl?raw";
import outputWgsl from "../gpu/shaders/output.wgsl?raw";
import grainWgsl from "../gpu/shaders/render_grain.wgsl?raw";
import gainmapWgsl from "../gpu/shaders/render_gainmap.wgsl?raw";
import { floatsToHalves } from "../gpu/half.ts";
import { CURVE_LUT_SIZE, TONE_LUT_SIZE, curveLUT, isFlat, toneCurveLUT } from "./curves.ts";
import { buildLUT, LOOKS, type Look } from "./looks.ts";
import { DEPTH_CURVE_SIZE, HUE_CURVE_SIZE, PROFILE_CURVE_SIZE, depthTable, hueCurveTable, profileCurveTable, profileUniforms, isNeutral } from "../looks/profile.ts";
import { GROUPS } from "../neural/scene.ts";
import { CELLS, DEPTH_BANDS, neutralSemantic, type CellKey, type Curves, type DepthBand, type Params, type SemanticAdjust } from "../decision/params.ts";
import type { RefinedMaps } from "../refine/refine.ts";

export interface RenderSource {
  base: GPUTexture;
  denoised: GPUTexture; // may be the same texture as base
  width: number;
  height: number;
  /** Width of the full-resolution image (for scaling pixel radii). */
  fullWidth: number;
  /** Apple's skin matte from a ProRAW file, in image coordinates (optional). */
  skin?: GPUTexture;
}

export interface RenderOptions {
  wb: number[]; // 3×3 working-space white balance
  gain: number; // analysis/guide encoding gain k
  lightLinear: [number, number, number];
  output: "srgb8" | "p38" | "p3f16";
  debugView?: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11;
  /** Depth range highlighted by debug view 5. */
  zoneRange?: [number, number];
  /** Region index highlighted by debug view 4. */
  region?: number;
  dither?: boolean;
  /** A draft while a control is dragged: cheaper where it does not show (grain sampling). */
  draft?: boolean;
  /** Compute the HDR gain (carried in alpha after the tone pass) at the photo's headroom. */
  hdr?: boolean;
  /** Also produce the 8-bit gain map for the gain-map JPEG: block size and range (log2 stops). */
  gainMap?: { scale: number; stops: number };
}

const MIDDLE_GREY_EV = Math.log2(0.18);

const semNeutral = (s: SemanticAdjust | undefined) => !s || (["exposure", "highlights", "saturation", "vibrance", "hue", "warmth", "tint"] as const).every((k) => Math.abs(s[k] ?? 0) < 1e-6)
  && (["clarity", "texture", "sharpen", "denoise", "dehaze"] as const).every((k) => Math.abs((s[k] ?? 1) - 1) < 1e-6);
/** A relative adjustment as the shader's three vec4 (hue in radians). */
const semVec = (s: SemanticAdjust | undefined): number[] => {
  const n = s ?? neutralSemantic();
  return [n.exposure, n.highlights, n.saturation, n.vibrance, (n.hue * Math.PI) / 180, n.clarity, n.texture, n.sharpen, n.denoise, n.dehaze, n.warmth ?? 0, n.tint ?? 0];
};

export class Renderer {
  private gpu: Gpu;
  private toneLut?: GPUTexture;
  /** Adjustment layers: records (storage buffer) and their tables (atlas), cached by content. */
  private atlas?: GPUTexture;
  private atlasRows = 0;
  private layerKey = "";
  private layerBuf?: GPUBuffer;
  private layerCount = 0;
  private look?: GPUTexture;
  private lookKey = "";
  private toneKey = "";
  private sampler: GPUSampler;
  private customLooks = new Map<string, Look>();
  private profCurve?: GPUTexture;
  private profCurveKey = "";
  private depthTab?: GPUTexture;
  private depthKey = "";
  private hueTab?: GPUTexture;
  private skinDummy?: GPUTexture;

  /**
   * Tap-to-select masks (set by the engine): an r8 texture array at the guide
   * resolution, the layer of each selection, and a version that changes with them.
   */
  selection?: { tex: GPUTexture; slotOf: (m: MaskShape) => number; version: number };
  private selDummy?: GPUTexture;
  private noSelection(): GPUTexture {
    this.selDummy ??= this.gpu.tex("selection.none", 1, 1, "r8unorm", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, "2d", 1);
    return this.selDummy;
  }

  /** A 1×1 "no skin here" texture for photographs without an Apple matte. */
  private noSkin(): GPUTexture {
    this.skinDummy ??= this.gpu.tex("skin.none", 1, 1, "r8unorm", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST);
    return this.skinDummy;
  }
  private hueKey = "";

  constructor(gpu: Gpu) {
    this.gpu = gpu;
    this.sampler = gpu.device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
  }

  addLook(l: Look) { this.customLooks.set(l.id, l); }
  looks(): Look[] { return [...LOOKS, ...this.customLooks.values()]; }

  private ensureLuts(p: Params, hdrStops = 0) {
    const gpu = this.gpu;
    // g channel: the HDR gain at this headroom (1 everywhere for an SDR render).
    const tk = JSON.stringify([p.tone, hdrStops]);
    if (tk !== this.toneKey) {
      this.toneKey = tk;
      if (!this.toneLut) this.toneLut = gpu.tex("toneLUT", TONE_LUT_SIZE, 1, "rgba16float", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST);
      gpu.device.queue.writeTexture({ texture: this.toneLut }, floatsToHalves(toneCurveLUT(p.tone, hdrStops)), { bytesPerRow: TONE_LUT_SIZE * 8 }, { width: TONE_LUT_SIZE, height: 1 });
    }
    // Adjustment layers: records + tables, re-uploaded only when they change.
    const layKey = JSON.stringify([p.layers ?? [], p.autoCurves ?? 1, p.enable.curves, p.enable.semantic, this.selection?.version ?? 0]);
    if (layKey !== this.layerKey) {
      this.layerKey = layKey;
      const pk = packLayers(p.layers ?? [], p.autoCurves ?? 1, p.enable, this.selection?.slotOf);
      this.layerCount = pk.count;
      if (!this.layerBuf || this.layerBuf.size < pk.records.byteLength) {
        gpu.release(this.layerBuf);
        this.layerBuf = gpu.buf("layers", Math.max(pk.records.byteLength, RECORD * 4 * 16), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      }
      gpu.device.queue.writeBuffer(this.layerBuf, 0, pk.records as Float32Array<ArrayBuffer>);
      if (!this.atlas || this.atlasRows !== pk.rows) {
        gpu.release(this.atlas);
        this.atlas = gpu.tex("layers.atlas", ATLAS_W, pk.rows, "rgba16float", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST);
        this.atlasRows = pk.rows;
      }
      gpu.device.queue.writeTexture({ texture: this.atlas }, floatsToHalves(pk.atlas), { bytesPerRow: ATLAS_W * 8, rowsPerImage: pk.rows }, { width: ATLAS_W, height: pk.rows });
    }
    // Profile components, cached by content: switching between loaded profiles
    // re-uploads nothing unless a component actually differs.
    const prof = p.profile;
    const pk = JSON.stringify([prof.tone, prof.rgbCurves]);
    if (pk !== this.profCurveKey) {
      this.profCurveKey = pk;
      if (!this.profCurve) this.profCurve = gpu.tex("profileCurve", PROFILE_CURVE_SIZE, 1, "rgba16float", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST);
      gpu.device.queue.writeTexture({ texture: this.profCurve }, floatsToHalves(profileCurveTable(prof)), { bytesPerRow: PROFILE_CURVE_SIZE * 8 }, { width: PROFILE_CURVE_SIZE, height: 1 });
    }
    const dk = JSON.stringify(prof.depth);
    if (dk !== this.depthKey) {
      this.depthKey = dk;
      if (!this.depthTab) this.depthTab = gpu.tex("profileDepth", DEPTH_CURVE_SIZE, 2, "rgba16float", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST);
      gpu.device.queue.writeTexture({ texture: this.depthTab }, floatsToHalves(depthTable(prof)), { bytesPerRow: DEPTH_CURVE_SIZE * 8, rowsPerImage: 2 }, { width: DEPTH_CURVE_SIZE, height: 2 });
    }
    const hk = JSON.stringify([prof.hueCurves, prof.satByLum]); // both live in this table
    if (hk !== this.hueKey) {
      this.hueKey = hk;
      if (!this.hueTab) this.hueTab = gpu.tex("profileHue", HUE_CURVE_SIZE, 2, "rgba16float", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST);
      gpu.device.queue.writeTexture({ texture: this.hueTab }, floatsToHalves(hueCurveTable(prof)), { bytesPerRow: HUE_CURVE_SIZE * 8, rowsPerImage: 2 }, { width: HUE_CURVE_SIZE, height: 2 });
    }
    const look = (prof.lut.id && this.looks().find((l) => l.id === prof.lut.id)) || LOOKS[0];
    // Imported tables keep their native size (17/33/65); procedural looks use the profile's size.
    const size = (look.table ? (look.table.size >= 65 ? 65 : look.table.size >= 33 ? 33 : 17) : prof.lut.size) as 17 | 33 | 65;
    const lk = look.id + "|" + size;
    if (lk !== this.lookKey) {
      this.lookKey = lk;
      gpu.release(this.look);
      this.look = gpu.tex("look3D", size, size, "rgba16float", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, "3d", size);
      gpu.device.queue.writeTexture({ texture: this.look }, floatsToHalves(buildLUT(look, size)), { bytesPerRow: size * 8, rowsPerImage: size }, { width: size, height: size, depthOrArrayLayers: size });
    }
    return { size, identity: look.id === "neutral" || !prof.lut.id };
  }

  private toneUniforms(p: Params, src: RenderSource, maps: RefinedMaps, o: RenderOptions, lutSize: number, lutOn: boolean): ArrayBuffer {
    const e = p.enable;
    const bits = (e.denoise ? 1 : 0) | (e.wb ? 2 : 0) | (e.exposure ? 4 : 0) | (e.localTone ? 8 : 0) | (e.semantic ? 16 : 0) | (e.dehaze ? 32 : 0) | (e.sharpen ? 64 : 0) | (e.curves ? 128 : 0);
    const u = new Uniforms(188)
      .u32(src.width, src.height, maps.w, maps.h)
      .mat3(o.wb)
      .f32(p.exposure, o.gain, p.denoise.luma, p.denoise.chroma)
      .f32(p.denoise.shadowBoost, p.dehaze.strength, p.dehaze.beta, p.dehaze.minT)
      .f32(o.lightLinear[0], o.lightLinear[1], o.lightLinear[2], 0)
      .f32(p.local.compression, p.local.clarity, p.local.texture, p.local.anchorEV ?? MIDDLE_GREY_EV)
      .f32(p.tone.shadows, p.tone.highlights, p.depth.near, p.depth.far)
      .f32(p.color.saturation, p.color.vibrance, o.region ?? 0, lutSize)
      .u32(bits, 0, lutOn ? 1 : 0, o.debugView ?? 0);
    // Regions, then skin (index 11: a layer over the regions, see render_tone.wgsl).
    for (const g of [...GROUPS, "skin"] as const) {
      const s = g === "skin" ? (p.skin ?? neutralSemantic()) : p.semantic[g];
      u.f32(s.exposure, s.highlights, s.saturation, s.vibrance);
      u.f32((s.hue * Math.PI) / 180, s.clarity, s.texture, s.sharpen);
      u.f32(s.denoise, s.dehaze, s.warmth ?? 0, s.tint ?? 0);
    }
    return u.bytes();
  }

  /**
   * Contrast equalizer (render_ceq.wgsl) on the tone pass's output, in place: levels
   * ping-pong between the detail target (free until the detail pass) and a texture of
   * their own, the bands gathering in a buffer; the result goes back into `t1`.
   */
  private contrastEq(enc: GPUCommandEncoder, temp: Array<GPUBuffer | GPUTexture>, c: ContrastEq, t1: GPUTexture, t2: GPUTexture, W: number, H: number, scale: number) {
    const gpu = this.gpu;
    const lv = levels(scale);
    if (!lv.length) return;
    // Only for this render (freed once it is submitted): cached, a preview's pair stayed
    // allocated through a full-size export and pushed the phone past its GPU budget.
    const acc = gpu.buf("render.ceqAcc", W * H * 8, GPUBufferUsage.STORAGE);
    const other = gpu.tex("render.ceq", W, H, "rgba16float", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING);
    temp.push(acc, other);
    this.ceqSampler ??= gpu.device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
    const sigma = edgeSigma(c.edges);
    let from = t1, to = t2;
    lv.forEach((l, i) => {
      const u = gpu.uniform(new Uniforms(8).u32(W, H, i === 0 ? 1 : 0, 0).f32(l.step, c.luma[l.band] ?? 0, c.chroma[l.band] ?? 0, sigma).bytes(), "ceq.u");
      temp.push(u);
      gpu.dispatch(enc, gpu.pipeline("render.ceq", ceqWgsl, "level"), [u, from.createView(), this.ceqSampler, to.createView(), acc], Math.ceil(W / 8), Math.ceil(H / 8));
      from = to;
      to = to === t2 ? other : t2;
    });
    const u = gpu.uniform(new Uniforms(8).u32(W, H, 0, 0).f32(0, 0, 0, 0).bytes(), "ceq.u");
    temp.push(u);
    gpu.dispatch(enc, gpu.pipeline("render.ceq", ceqWgsl, "finalize"), [u, undefined, undefined, t1.createView(), acc], Math.ceil(W / 8), Math.ceil(H / 8));
  }
  private ceqSampler?: GPUSampler;

  // ------------------------------------------------------------------ targets
  /** Render targets are reused across renders (no per-frame allocation churn). */
  private targets = new Map<string, GPUTexture>();
  /** Cached by name, format *and* size, so preview, draft and thumbnail renders
   * (different sizes) don't reallocate each other's targets. Oldest entries are
   * evicted beyond a small budget. */
  private dofMips?: { key: string; tex: GPUTexture };
  private target(name: string, w: number, h: number, fmt: GPUTextureFormat): GPUTexture {
    const key = `${name}|${fmt}|${w}x${h}`;
    let t = this.targets.get(key);
    if (t) {
      this.targets.delete(key);
      this.targets.set(key, t); // most recently used last
      return t;
    }
    t = this.gpu.tex(`render.${name}`, w, h, fmt, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC);
    this.targets.set(key, t);
    while (this.targets.size > 18) {
      const [k, old] = this.targets.entries().next().value as [string, GPUTexture];
      this.targets.delete(k);
      this.gpu.release(old);
    }
    return t;
  }
  /** Frees all cached render targets (after an export, or when a photo closes). */
  releaseTargets() {
    for (const t of this.targets.values()) this.gpu.release(t);
    this.targets.clear();
    this.dofMips?.tex.destroy();
    this.dofMips = undefined;
  }

  private toneDispatch(enc: GPUCommandEncoder, temp: Array<GPUBuffer | GPUTexture>, src: RenderSource, maps: RefinedMaps, p: Params, o: RenderOptions,
    lutSize: number, lutOn: boolean, profileOn: boolean, dst: GPUTexture, distT: GPUTexture, ty0: number, th: number, gainT?: GPUTexture) {
    const gpu = this.gpu;
    // Tone uniforms followed by the target rectangle (tgt: offset x/y, width, height).
    const base = this.toneUniforms(p, src, maps, o, lutSize, lutOn);
    // Tone uniforms, then tgt, hl, vig (amount, midpoint, feather, roundness), vig2 (vignette highlights, depth band edges near|middle, middle|far, band crossfade), hdr (on).
    // …, hdr, dsem (distance detail, 9 vec4), lay (layer count, atlas rows, distance bits).
    // …, lay, then the tone equalizer: teq (detail, mask exposure, mask contrast) and its 64-value curve.
    const buf = new ArrayBuffer(base.byteLength + 80 + 144 + 16 + 16 + 256);
    new Uint8Array(buf).set(new Uint8Array(base));
    new Int32Array(buf, base.byteLength, 4).set([0, ty0, src.width, th]);
    // hl: view 5's zone range; zw: the object kept whole in focus (group + 1, 0 none) and its focus distance.
    const pts = p.dof.points ?? [];
    const kept = pts.find((q) => q.group !== undefined);
    const keep = p.enable.dof && p.dof.mode !== "zones" ? (kept ? { g: kept.group!, d: kept.dist } : !pts.length && p.dof.protect !== undefined ? { g: p.dof.protect, d: p.dof.focus } : undefined) : undefined;
    new Float32Array(buf, base.byteLength + 16, 4).set([o.zoneRange?.[0] ?? -1, o.zoneRange?.[1] ?? 2, keep ? keep.g + 1 : 0, keep?.d ?? 0]);
    const v = p.vignette ?? { amount: 0, midpoint: 0.5, feather: 0.6, roundness: 0.3, highlights: 0.5 };
    const db = p.depthBands ?? [0.33, 0.66];
    new Float32Array(buf, base.byteLength + 32, 8).set([v.amount, v.midpoint, v.feather, v.roundness, v.highlights, db[0], db[1], 0.06]);
    // hdr: write the HDR gain; y z w: the img rendering (on, purity, strength).
    const rd = p.render;
    new Float32Array(buf, base.byteLength + 64, 4).set([gainT ? 1 : 0, rd?.engine === "img" ? 1 : 0, rd?.purity ?? 0, rd?.strength ?? 1]);
    // By distance and by region at a distance: relative settings, then which are active.
    const regionsOn = p.enable.semantic;
    new Float32Array(buf, base.byteLength + 80, 36).set(DEPTH_BANDS.flatMap((b) => semVec(p.distance?.[b])));
    let dbits = 0;
    DEPTH_BANDS.forEach((b, i) => { if (p.enable.semantic && !semNeutral(p.distance?.[b])) dbits |= 1 << i; });
    new Uint32Array(buf, base.byteLength + 224, 4).set([this.layerCount, this.atlasRows, dbits, p.protectHighlights === false ? 0 : 1]);
    // Off unless it changes something — or its mask is being shown (view 9).
    const eq = p.toneEq;
    const eqOn = !!eq && eq.enabled && (toneEqActive(eq) || o.debugView === 9 || o.debugView === 10 || o.debugView === 11);
    new Float32Array(buf, base.byteLength + 240, 4).set([eqOn ? TONE_EQ_DETAIL.indexOf(eq!.detail) : -1, eq?.maskExposure ?? 0, eq?.maskContrast ?? 1, 0]);
    if (eqOn) new Float32Array(buf, base.byteLength + 256, 64).set(toneEqLut(eq!));
    const u = gpu.uniform(buf, "tone.u");
    const profU = gpu.uniform(profileUniforms(p.profile, profileOn, lutOn, lutSize), "profile.u");
    temp.push(u, profU);
    gpu.dispatch(enc, gpu.pipeline("render.tone", toneWgsl + "\n" + layersWgsl), [
      u, src.base.createView(), src.denoised.createView(), maps.guide.createView(),
      maps.masks[0].createView(), maps.masks[1].createView(), maps.masks[2].createView(),
      maps.toneC.createView(), maps.toneM.createView(),
      this.toneLut!.createView(), this.atlas!.createView(), this.look!.createView({ dimension: "3d" }),
      this.sampler, dst.createView(), distT.createView(),
      profU, this.profCurve!.createView(), this.depthTab!.createView(), this.hueTab!.createView(),
      (src.skin ?? this.noSkin()).createView(),
      (gainT ?? this.target("gainDummy", 1, 1, "r32float")).createView(),
      this.layerBuf!,
      (this.selection?.tex ?? this.noSelection()).createView({ dimension: "2d-array" }),
    ], Math.ceil(src.width / 8), Math.ceil(th / 8));
  }

  /**
   * Renders rows [y0, y0 + rows) of `src` (default: the whole image). The
   * result texture belongs to the renderer and stays valid until the next
   * render; the requested rows start at row `top` of it (the rest is apron for
   * sharpening / depth-of-field neighbourhoods). Rendering in strips keeps the
   * extra memory of a full-resolution export to a few tens of MB.
   */
  async render(src: RenderSource, maps: RefinedMaps, p: Params, o: RenderOptions, dofOn: boolean, strip?: { y0: number; rows: number }): Promise<{ tex: GPUTexture; top: number; rows: number; gm?: GPUTexture; gmW?: number; gmRows?: number }> {
    const gpu = this.gpu;
    const { width: W, height: H } = src;
    const hdrStops = o.hdr ? (p.hdr?.headroom ?? 0) : 0;
    const { size: lutSize, identity } = this.ensureLuts(p, hdrStops);
    // The blur pass: depth of field, and/or Blur layers (a radius of 3 % of the long side at amount 1).
    const depthDof = dofOn && p.dof.strength > 0;
    const blurR = ((o.debugView ?? 0) < 7 || o.debugView === 11) && hasBlurLayers(p.layers ?? [], p.autoCurves ?? 1, p.enable) ? 0.03 * Math.max(W, H) : 0;
    const dof = depthDof || blurR > 0;
    const maxRadius = depthDof ? p.dof.strength * 0.022 * Math.max(W, H) : 0;
    const y0 = strip?.y0 ?? 0, rows = strip?.rows ?? H;
    const ceqOn = contrastEqActive(p.contrastEq) && (o.debugView ?? 0) < 7;
    const apron = strip ? Math.max(dof ? Math.ceil(Math.max(maxRadius, blurR)) + 4 : 3, ceqOn ? stripApron(W / src.fullWidth) : 0) : 0;
    // Strip starts are aligned to 64 rows so the depth-of-field mip grid (up to
    // 2^5-row texels) lines up with the full-image grid: no seams between strips.
    // Heights are rounded up to 64 rows too (mip level sizes round down, so an
    // odd height would squeeze the coarse levels relative to the full image).
    const ty0 = Math.max(0, Math.floor((y0 - apron) / 64) * 64);
    const ty1 = strip ? Math.min(H, ty0 + Math.ceil((y0 + rows + apron - ty0) / 64) * 64) : H;
    const th = ty1 - ty0;
    const dofLevels = Math.max(1, Math.min(6, Math.floor(Math.log2(Math.min(W, H) / 16))));
    const top = y0 - ty0;
    const t1 = this.target("tone", W, th, "rgba16float");
    const t2 = this.target("detail", W, th, "rgba16float");
    const gainT = hdrStops > 0 ? this.target("gain", W, th, "r32float") : undefined;
    const distT = dof ? this.target("dist", W, th, "r32float") : this.target("distDummy", 1, 1, "r32float");
    const scale = W / src.fullWidth;
    await gpu.run("render.tone+detail", (enc, temp) => {
      this.toneDispatch(enc, temp, src, maps, p, o, lutSize, !identity, p.enable.lut && !isNeutral(p.profile), t1, distT, ty0, th, gainT);
      if (ceqOn) this.contrastEq(enc, temp, p.contrastEq!, t1, t2, W, th, scale);
      // Sharpening radius is defined at full resolution; a preview sees it scaled.
      const radius = p.sharpen.radius * Math.max(scale, 0.35);
      const amount = p.enable.sharpen ? p.sharpen.amount * Math.min(1, scale * 1.5 + 0.2) : 0;
      const ud = gpu.uniform(new Uniforms(8).u32(W, th, gainT ? 1 : 0, 0).f32(amount, radius, p.sharpen.threshold, 0).bytes(), "detail.u");
      temp.push(ud);
      gpu.dispatch(enc, gpu.pipeline("render.detail", detailWgsl), [ud, t1.createView(), t2.createView(), (gainT ?? this.target("gainDummy", 1, 1, "r32float")).createView()], Math.ceil(W / 8), Math.ceil(th / 8));
    });
    let final = t2;
    let finalLinear = false;
    if (dof) {
      // Into the tone target: free once the detail pass has read it (saves a full-size buffer).
      final = await this.depthOfField(t2, t1, distT, W, th, p, maxRadius, dofLevels, blurR, depthDof);
      finalLinear = true;
    }
    const gr = p.grain;
    if (gr && gr.amount > 0 && o.debugView !== 1 && o.debugView !== 2 && (o.debugView ?? 0) < 7) {
      // Last, on the finished image. Particle size is set for a ~12 MP frame and
      // scales with the image; `scale` is this render's pixels per full-image pixel.
      const sizePx = (0.7 + 2.3 * gr.size) * Math.max(W / scale, H / scale) / 4032;
      // Into whichever of tone / detail the image is not in now (no buffer of its own).
      const out = final === t1 ? t2 : t1;
      await gpu.run("render.grain", (enc, temp) => {
        const u = gpu.uniform(new Uniforms(12).u32(W, th, ty0, finalLinear ? 1 : 0).f32(gr.amount * 0.055, sizePx, gr.roughness, gr.color).f32(1 / scale, o.draft ? 1 : 0, 0, 0).bytes(), "grain.u");
        temp.push(u);
        gpu.dispatch(enc, gpu.pipeline("render.grain", grainWgsl), [u, final.createView(), out.createView()], Math.ceil(W / 8), Math.ceil(th / 8));
      });
      final = out;
    }
    // Gain map for the requested rows (alpha of `final` is the HDR gain).
    let gm: GPUTexture | undefined, gmW = 0, gmRows = 0;
    if (o.gainMap && gainT) {
      const s = o.gainMap.scale;
      gmW = Math.ceil(W / s); gmRows = Math.ceil(rows / s);
      gm = this.target("gm8", gmW, gmRows, "rgba8unorm");
      await gpu.run("render.gainmap", (enc, temp) => {
        const u = gpu.uniform(new Uniforms(12).u32(W, th, top, s).u32(finalLinear ? 1 : 0, gmW, rows, 0).f32(0, o.gainMap!.stops, 1, 1 / 64).bytes(), "gainmap.u");
        temp.push(u);
        gpu.dispatch(enc, gpu.pipeline("render.gainmap", gainmapWgsl), [u, final.createView(), gm!.createView()], Math.ceil(gmW / 8), Math.ceil(gmRows / 8));
      });
    }
    if (o.output === "p3f16" && !finalLinear) return { tex: final, top, rows, gm, gmW, gmRows };
    if (o.output === "p3f16") return { tex: await this.encodeF16(final, W, th), top, rows, gm, gmW, gmRows };
    const out = this.target("out8", W, th, "rgba8unorm");
    await gpu.run("render.output", (enc, temp) => {
      const u = gpu.uniform(new Uniforms(8).u32(W, th, 0, 0, o.output === "srgb8" ? 0 : 1, o.dither === false ? 0 : 1, finalLinear ? 1 : 0, ty0).bytes());
      temp.push(u);
      gpu.dispatch(enc, gpu.pipeline("output.encode", outputWgsl, "encode"), [u, final.createView(), out.createView()], Math.ceil(W / 8), Math.ceil(th / 8));
    });
    return { tex: out, top, rows, gm, gmW, gmRows };
  }

  /** Scene-linear Rec.2020 (after denoise, WB, dehaze, exposure) for the linear DNG export; same strip contract as render(). */
  async renderLinear(src: RenderSource, maps: RefinedMaps, p: Params, o: RenderOptions, strip?: { y0: number; rows: number }): Promise<{ tex: GPUTexture; top: number; rows: number }> {
    const gpu = this.gpu;
    const { size: lutSize, identity } = this.ensureLuts(p);
    const y0 = strip?.y0 ?? 0, rows = strip?.rows ?? src.height;
    const t1 = this.target("linear", src.width, rows, "rgba16float");
    const distT = this.target("distDummy", 1, 1, "r32float");
    await gpu.run("render.linear", (enc, temp) => {
      this.toneDispatch(enc, temp, src, maps, p, { ...o, debugView: 3 as never }, lutSize, !identity, false, t1, distT, y0, rows);
    });
    return { tex: t1, top: 0, rows };
  }

  private async encodeF16(src: GPUTexture, W: number, H: number): Promise<GPUTexture> {
    // Re-encode linear P3 → sRGB-curve P3 in rgba16float (reuses the tone path's encoding).
    const gpu = this.gpu;
    const code = `@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let s = textureDimensions(src);
  if (id.x >= s.x || id.y >= s.y) { return; }
  let c = textureLoad(src, vec2<i32>(id.xy), 0);
  textureStore(dst, vec2<i32>(id.xy), vec4<f32>(srgb_oetf(clamp(c.rgb, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0));
}`;
    const out = this.target("out16", W, H, "rgba16float");
    await gpu.run("render.encode16", (enc) => {
      gpu.dispatch(enc, gpu.pipeline("encode16", code), [src.createView(), out.createView()], Math.ceil(W / 8), Math.ceil(H / 8));
    });
    return out;
  }

  private async depthOfField(sharp: GPUTexture, out: GPUTexture, distT: GPUTexture, W: number, H: number, p: Params, maxRadius: number, fullLevels: number, blurR = 0, depthOn = true): Promise<GPUTexture> {
    const gpu = this.gpu;
    // Same level count as a full-image render (a strip may be shorter), limited by what fits.
    const levels = Math.max(1, Math.min(fullLevels, Math.floor(Math.log2(Math.min(W, H))) + 1));
    // Kept between renders (a 20 MB allocation and a GPU stall per frame otherwise).
    const mipKey = `${W}x${H}x${levels}`;
    if (this.dofMips?.key !== mipKey) {
      this.dofMips?.tex.destroy();
      this.dofMips = { key: mipKey, tex: gpu.device.createTexture({
        label: "dof.mips", size: { width: W, height: H }, format: "rgba16float", mipLevelCount: levels,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      }) };
    }
    const mipTex = this.dofMips.tex;
    await gpu.run("dof", (enc, temp) => {
      const u0 = gpu.uniform(new Uniforms(8).u32(W, H, 0, 0, 0, 0, 0, 0).bytes());
      temp.push(u0);
      gpu.dispatch(enc, gpu.pipeline("output.linearize", outputWgsl, "linearize"), [u0, sharp.createView(), undefined, mipTex.createView({ baseMipLevel: 0, mipLevelCount: 1 })], Math.ceil(W / 8), Math.ceil(H / 8));
      let w = W, h = H;
      for (let l = 1; l < levels; l++) {
        const nw = Math.max(1, w >> 1), nh = Math.max(1, h >> 1);
        const u = gpu.uniform(new Uniforms(8).u32(nw, nh, w, h, 0, 0, 0, 0).bytes());
        temp.push(u);
        gpu.dispatch(enc, gpu.pipeline("output.mip", outputWgsl, "mip"), [u, mipTex.createView({ baseMipLevel: l - 1, mipLevelCount: 1 }), undefined, mipTex.createView({ baseMipLevel: l, mipLevelCount: 1 })], Math.ceil(nw / 8), Math.ceil(nh / 8));
        w = nw; h = nh;
      }
      const pts = p.dof.points.slice(0, 8);
      // Each point keeps its object's whole depth range sharp (a flat spot: just its distance).
      const foci = Array.from({ length: 8 }, (_, i) => pts[i]?.range?.[0] ?? pts[i]?.dist ?? 0);
      const fociHi = Array.from({ length: 8 }, (_, i) => pts[i]?.range?.[1] ?? pts[i]?.dist ?? 0);
      const span = p.dof.focusSpan ?? [0, 0];
      const zonesOn = p.dof.mode === "zones" && p.dof.zones?.length === 5 && p.dof.zoneBounds?.length === 4;
      const zc = zonesOn ? [...p.dof.zoneBounds!, 0, 0, 0, 0] : new Array(8).fill(0);
      const zv = zonesOn ? [...p.dof.zones!, 0, 0, 0] : new Array(8).fill(0);
      const u = gpu.uniform(new Uniforms(48).u32(W, H, 0, 0).f32(p.dof.focus, maxRadius, 0.6, span[0]).f32(pts.length, zonesOn ? 1 : 0, levels, span[1]).f32(...foci).f32(...zc).f32(...zv).f32(...fociHi).f32(blurR, depthOn ? 1 : 0, 0, 0).bytes());
      temp.push(u);
      const cocT = this.target("dof.coc", W, H, "rg32float");
      gpu.dispatch(enc, gpu.pipeline("render.dof.coc", dofWgsl, "coc_pass"), [u, undefined, distT.createView(), undefined, undefined, undefined, cocT.createView()], Math.ceil(W / 8), Math.ceil(H / 8));
      gpu.dispatch(enc, gpu.pipeline("render.dof", dofWgsl), [u, mipTex.createView(), undefined, cocT.createView(), this.sampler, out.createView()], Math.ceil(W / 8), Math.ceil(H / 8));
    });
    return out;
  }

  destroy() {
    this.releaseTargets();
    this.gpu.release(this.toneLut, this.atlas, this.layerBuf, this.look, this.profCurve, this.depthTab, this.hueTab, this.skinDummy, this.selDummy);
  }
}
