/**
 * Burst merge: several shots of one scene → one cleaner image (HDR+-style).
 *
 * Frames stream through one at a time — never the whole series in memory:
 *
 *   Pass A (scan)   each frame developed, reduced to ≈1024 px luma, freed.
 *                   On the scans: the sharpest frame becomes the reference,
 *                   every frame is aligned to it (align.ts, coarse to fine),
 *                   its exposure ratio and the series' noise are measured.
 *   Pass B (merge)  each frame developed again at full size, its alignment
 *                   refined on the GPU (±2 px, sub-pixel), and added to a robust
 *                   running mean (burst.wgsl): where it disagrees with the mean
 *                   beyond noise (something moved) or is clipped, it adds nothing.
 *
 * The result is an ordinary working image (linear Rec.2020, rgba16float) that the
 * rest of the app edits like any photo.
 */
import { Gpu, Uniforms } from "../gpu/gpu.ts";
import burstWgsl from "../gpu/shaders/burst.wgsl?raw";
import { halvesToFloats } from "../gpu/half.ts";
import { decodeFile } from "../decode/decode.ts";
import type { DecodedImage } from "../decode/types.ts";
import { develop, type WorkingImage } from "../raw/develop.ts";
import { downsample } from "../refine/refine.ts";
import { denoiseGPU } from "../restore/denoise.ts";
import { measureBlocks, noiseProfile } from "../analysis/analysis.ts";
import { alignTiles, coveredTiles, searchScale, exposureRatio, fitGlobal, flowAt, half, modelAt, pyramid, regularise, resamplePlane, rescale, scaleOf, scanNoise, sharpness, texturedTiles, warp, type Affine, type Flow, type Plane } from "./align.ts";

export type BurstMode = "clean";

export interface SeriesOptions {
  /** Working-resolution divisor for a photo of this size (the engine's rule; the first frame decides). */
  factorFor: (w: number, h: number) => number;
  mode: BurstMode;
  /** Frame to use as the reference (index into files); default: the sharpest. */
  ref?: number;
  /** Exposure normalisation gain from linear luminances (the engine's exposureGain). */
  exposureGain: (y: Float32Array) => number;
  progress: (stage: string, detail?: string, frac?: number) => void;
  log: (text: string) => void;
  /** Registers something to free if the open does not complete. */
  track: (free: () => void) => void;
  cancelled: () => boolean;
}

export interface SeriesResult {
  decoded: DecodedImage;
  work: WorkingImage;
  file: File;
  refIndex: number;
  /** Frames merged (after dropping unusable ones). */
  used: number;
  factor: number;
  /** Measured noise σ (linear, mid-tones) of one frame and of the merge, from the scans. */
  noise: { single: number; merged: number };
}

const SCAN_LONG = 1024;
const TILE = 16;

interface Scan {
  index: number; file: File;
  /** Luma at scan size, and R, G, B at a quarter of that (colour matching between lenses). */
  y: Plane; rgb: [Plane, Plane, Plane];
  ev: number; sharp: number;
  /** 35 mm-equivalent focal length (which lens), and the frame's developed size. */
  focal?: number; W: number; H: number; factor: number;
}

/** One frame's alignment to the reference. */
interface Aligned {
  scan: Scan;
  /** Tile field (scan pixels of the reference) the GPU starts from: all motion, or what the global mapping leaves. */
  flow: Flow;
  g: number;
  /** Another lens: the reference → frame mapping at full size, per-channel colour gains, and the frame's weight. */
  lens?: { aff: Affine; gc: [number, number, number]; weight: number };
  /** Less weight for a frame shaken more than the reference (its detail is softer). */
  sharpWeight: number;
}

async function decodeDev(gpu: Gpu, file: File, factor: number | ((w: number, h: number) => number)): Promise<{ decoded: DecodedImage; work: WorkingImage }> {
  const decoded = await decodeFile(new Uint8Array(await file.arrayBuffer()), file.name, file.type);
  try {
    const f = typeof factor === "number" ? factor : factor(decoded.source.width, decoded.source.height);
    const work = await develop(gpu, decoded, { factor: f });
    return { decoded, work };
  } catch (e) { decoded.close(); throw e; }
}

/** Luma (linear) of a working texture at scan size, and its R, G, B at a quarter of that. */
async function scanOf(gpu: Gpu, work: WorkingImage): Promise<{ y: Plane; rgb: [Plane, Plane, Plane] }> {
  const k = Math.min(1, SCAN_LONG / Math.max(work.width, work.height));
  const w = Math.max(TILE * 4, Math.round(work.width * k)), h = Math.max(TILE * 4, Math.round(work.height * k));
  const t = await downsample(gpu, work.tex, work.width, work.height, w, h, false, 1, "burst.scan");
  const f = halvesToFloats(new Uint16Array(await gpu.readTexture(t, 0, 0, w, h, 8)));
  gpu.release(t);
  const d = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) d[i] = Math.max(0, 0.2627 * f[i * 4] + 0.678 * f[i * 4 + 1] + 0.0593 * f[i * 4 + 2]);
  const tw = w >> 2, th = h >> 2;
  const rgb = [0, 1, 2].map((c) => {
    const q = new Float32Array(tw * th);
    for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) {
      let s = 0;
      for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) s += f[((y * 4 + j) * w + x * 4 + i) * 4 + c];
      q[y * tw + x] = Math.max(0, s / 16);
    }
    return { w: tw, h: th, d: q };
  }) as [Plane, Plane, Plane];
  return { y: { w, h, d }, rgb };
}

const sqrtPlane = (p: Plane, g: number): Plane => ({ w: p.w, h: p.h, d: p.d.map((v) => Math.sqrt(Math.max(0, v * g))) });
const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
/** Brightness of a scan (median of mid-grey-ish pixels, robust to framing shifts). */
const level = (p: Plane) => { const s: number[] = []; for (let i = 0; i < p.d.length; i += 7) if (p.d[i] >= 0) s.push(p.d[i]); return Math.max(1e-5, median(s)); };
const diag = (p: { w: number; h: number }) => Math.hypot(p.w, p.h);
/** Tiles set in both masks. */
const both = (a: Uint8Array, b: Uint8Array) => a.map((v, i) => v & b[i]);

export async function mergeSeries(gpu: Gpu, files: File[], opt: SeriesOptions): Promise<SeriesResult> {
  const t0 = performance.now();
  const n = files.length;
  // ------------------------------------------------------------------ pass A: scan
  const scans: Scan[] = [];
  for (let i = 0; i < n; i++) {
    opt.progress("series scan", `${i + 1}/${n}`, (i / n) * 0.35);
    const { decoded, work } = await decodeDev(gpu, files[i], opt.factorFor);
    try {
      decoded.close();
      const { y, rgb } = await scanOf(gpu, work);
      scans.push({ index: i, file: files[i], y, rgb, ev: Math.log2(level(y)), sharp: 0, focal: decoded.meta.focalLength35 || undefined, W: work.width, H: work.height, factor: work.factor });
    } finally { gpu.release(work.tex); }
    if (opt.cancelled()) throw new Error("cancelled");
  }
  if (!scans.length) throw new Error("No usable frames in the series");

  // Reference: from the lens most frames were taken with, at the series' usual
  // exposure (brackets aside), the sharpest.
  const lensOf = (s: Scan) => s.focal ? `${Math.round(s.focal)}mm` : `${s.W}×${s.H}`;
  const groups = new Map<string, Scan[]>();
  for (const s of scans) groups.set(lensOf(s), [...(groups.get(lensOf(s)) ?? []), s]);
  const main = [...groups.values()].sort((a, b) => b.length - a.length || b[0].W * b[0].H - a[0].W * a[0].H)[0];
  const evMid = median(main.map((s) => s.ev));
  // Sharpness at half the scan size (noise mostly averaged out), each frame at its own level.
  for (const s of scans) s.sharp = sharpness(half(sqrtPlane(s.y, 1 / 2 ** s.ev)));
  const byUser = opt.ref !== undefined ? scans.find((s) => s.index === opt.ref) : undefined;
  const ref = byUser ?? main.filter((s) => Math.abs(s.ev - evMid) < 0.35).sort((a, b) => b.sharp - a.sharp)[0] ?? main[0];
  const W = ref.W, H = ref.H, factor = ref.factor;
  opt.log(`series: ${scans.length} frames; lenses ${[...groups].map(([k, v]) => `${k} ×${v.length}`).join(", ")}; reference ${ref.file.name} ${W}×${H}${byUser ? " (chosen)" : " (sharpest)"}; levels ${scans.map((s) => (s.ev - ref.ev).toFixed(2)).join(", ")} EV`);

  // Coarse alignment on the scans, and each frame's exposure (and, across lenses, colour) ratio.
  const refPyr = pyramid(sqrtPlane(ref.y, 1));
  const rw = ref.y.w, rh = ref.y.h;
  const aligned: Aligned[] = [];
  const warpedY: Plane[] = [];
  let textured: Uint8Array | undefined;
  for (const s of scans) {
    if (s === ref) continue;
    let g = 2 ** (ref.ev - s.ev);
    const sameLens = lensOf(s) === lensOf(ref) && s.W === W && s.H === H;
    // Another lens: its view is brought to the reference's scale first (EXIF focal lengths, else a search).
    let s0 = 1;
    if (!sameLens) {
      const sF = s.focal && ref.focal ? s.focal / ref.focal : searchScale(sqrtPlane(ref.y, 1), s.y, g, diag(s.y) / diag(ref.y));
      s0 = sF * (diag(s.y) / diag(ref.y));
    }
    const base = sameLens ? s.y : resamplePlane(s.y, rw, rh, { a: s0, b: 0, tx: 0, ty: 0 });
    const raw = alignTiles(refPyr, pyramid(sqrtPlane(base, g)));
    textured ??= texturedTiles(refPyr[0], raw, 0.01);
    const mask = sameLens ? textured : both(textured, coveredTiles(base, raw));
    const usable = mask.reduce((a, v) => a + v, 0);
    if (usable < mask.length * 0.05) { opt.log(`series: ${s.file.name} left out (too little of the scene in common)`); continue; }
    const model = fitGlobal(raw, rw, rh, mask);
    const flow = regularise(raw, model, rw, rh, mask, sameLens ? 2 : 6);
    const wy = warp(base, flow);
    g = exposureRatio(ref.y, wy);
    const shift = Math.hypot(model.tx, model.ty) * (W / rw);
    const rot = (Math.atan2(model.b, model.a) * 180) / Math.PI;
    let lens: Aligned["lens"];
    let init = flow;
    if (!sameLens) {
      // Reference → frame at scan size, then at full size and on the colour thumbnails.
      const affScan: Affine = { a: s0 * model.a, b: s0 * model.b, tx: s0 * model.tx, ty: s0 * model.ty };
      const aff = rescale(affScan, W / rw, s.W / s.y.w);
      const affT = rescale(affScan, ref.rgb[0].w / rw, s.rgb[0].w / s.y.w);
      const ch = [0, 1, 2].map((c) => {
        const wc = resamplePlane(s.rgb[c], ref.rgb[c].w, ref.rgb[c].h, affT);
        return exposureRatio(ref.rgb[c], wc, 0.9);
      });
      const l = 0.2627 * ch[0] + 0.678 * ch[1] + 0.0593 * ch[2];
      const gc = ch.map((v) => Math.min(1.25, Math.max(0.8, v / l))) as [number, number, number];
      // A frame spread over more reference pixels than it has (a wider lens) carries less detail.
      lens = { aff, gc, weight: Math.min(1, scaleOf(aff) ** 2) };
      // The GPU resamples the frame by that mapping: what is left of the field is the start.
      init = { ...flow, dx: flow.dx.slice(), dy: flow.dy.slice() };
      for (let i = 0; i < flow.dx.length; i++) {
        const cx = Math.min((i % flow.tw) * TILE, Math.max(0, rw - TILE)) + TILE / 2, cy = Math.min(Math.floor(i / flow.tw) * TILE, Math.max(0, rh - TILE)) + TILE / 2;
        const [mx, my] = modelAt(model, cx, cy, rw, rh);
        init.dx[i] -= mx; init.dy[i] -= my;
      }
      opt.log(`series: ${s.file.name} another lens (${lensOf(s)}): scale ×${(1 / scaleOf(aff)).toFixed(3)} to the reference, colour ×${gc.map((v) => v.toFixed(3)).join("/")}, weight ${lens.weight.toFixed(2)}`);
    }
    opt.log(`series: ${s.file.name} shift ${shift.toFixed(1)} px, rotation ${rot.toFixed(2)}°, exposure ×${g.toFixed(3)}`);
    // A frame softer than the reference (hand shake): its share falls with the square of that.
    const sharpWeight = Math.min(1, Math.max(0.1, (s.sharp / Math.max(1e-12, ref.sharp)) ** 2));
    opt.log(`series: ${s.file.name} sharpness ${(s.sharp / Math.max(1e-12, ref.sharp)).toFixed(2)} of the reference → weight ${sharpWeight.toFixed(2)}`);
    aligned.push({ scan: s, flow: init, g, lens, sharpWeight });
    warpedY.push({ ...wy, d: wy.d.map((v) => (v < 0 ? -1 : v * g)) });
  }
  const sig = scanNoise(ref.y, warpedY);
  const midSig = sig[5];
  // Pass B needs none of the scan planes (≈ 7 MB a frame): freed before its GPU peak.
  warpedY.length = 0;
  for (const s of scans) if (s !== ref) { s.y = { w: s.y.w, h: s.y.h, d: new Float32Array(0) }; s.rgb = [s.y, s.y, s.y]; }
  opt.log(`series: noise σ per √Y bin (scan) ${sig.map((v) => (v * 1000).toFixed(2)).join(" ")} ×1e-3`);
  if (opt.cancelled()) throw new Error("cancelled");

  // ------------------------------------------------------------------ pass B: merge
  const pipe = (e: string) => gpu.pipeline("burst", burstWgsl, e);
  const sampler = gpu.device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
  const tw = Math.ceil(W / TILE), th = Math.ceil(H / TILE);
  const scanK = W / rw;
  const uni = (g: number, mode = 0, a?: Aligned) => {
    const L = a?.lens;
    return new Uniforms(28).u32(W, H, tw, th).f32(g, scanK / 3, aligned.length + 1).u32(mode).f32(...sig)
      .f32(L?.aff.a ?? 1, L?.aff.b ?? 0, L?.aff.tx ?? 0, L?.aff.ty ?? 0).f32(a?.scan.W ?? W, a?.scan.H ?? H, 0, 0)
      .f32(...(L?.gc ?? [1, 1, 1]), (L?.weight ?? 1) * (a?.sharpWeight ?? 1)).bytes();
  };

  opt.progress("series merge", `1/${aligned.length + 1}`, 0.35);
  const R = await decodeDev(gpu, ref.file, factor);
  opt.track(() => R.decoded.close());
  const acc = gpu.buf("burst.acc", W * H * 8, GPUBufferUsage.STORAGE);
  const wbuf = gpu.buf("burst.weights", Math.ceil(W / 2) * H * 4, GPUBufferUsage.STORAGE);
  const flowBuf = gpu.buf("burst.flow", tw * th * 16, GPUBufferUsage.STORAGE);
  const initBuf = gpu.buf("burst.init", tw * th * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const refL = gpu.tex("burst.refLuma", W, H, "r32float");
  const frmL = gpu.tex("burst.frameLuma", W, H, "r32float");
  const temps = [acc, wbuf, flowBuf, initBuf, refL, frmL];
  let out: GPUTexture | undefined;
  try {
    // Noise of one full-size frame (for the areas only the reference covers).
    const gain = opt.exposureGain(ref.y.d);
    const noise = noiseProfile(await measureBlocks(gpu, R.work.tex, W, H, gain, 0.02));
    await gpu.run("burst.init", (enc, temp) => {
      const u = gpu.uniform(uni(1)); temp.push(u);
      gpu.dispatch(enc, pipe("init"), [u, R.work.tex.createView(), undefined, undefined, undefined, undefined, undefined, undefined, acc], Math.ceil(W / 8), Math.ceil(H / 8));
      gpu.dispatch(enc, pipe("luma"), [u, R.work.tex.createView(), refL.createView()], Math.ceil(W / 16), Math.ceil(H / 16));
    });
    // The reference's pixels live in the accumulator now; its decode keeps only metadata.
    R.decoded.close();
    R.decoded.close = () => {};
    if (R.decoded.source.kind !== "rgb") R.decoded.source.data = new Uint16Array(0);
    const log = [...R.work.log];
    gpu.release(R.work.tex);

    for (let k = 0; k < aligned.length; k++) {
      const a = aligned[k];
      opt.progress("series merge", `${k + 2}/${aligned.length + 1}`, 0.35 + ((k + 1) / (aligned.length + 1)) * 0.6);
      const F = await decodeDev(gpu, a.scan.file, a.scan.factor);
      F.decoded.close();
      let frame = F.work.tex;
      try {
        if (a.lens) {
          // Another lens: resampled into the reference's view first (colour matched, uncovered parts marked).
          const warped = gpu.tex("burst.resampled", W, H, "rgba16float");
          const native = F.work.tex;
          frame = warped; // (released in finally, whatever happens)
          await gpu.run("burst.resample", (enc, temp) => {
            const u = gpu.uniform(uni(a.g, 0, a)); temp.push(u);
            gpu.dispatch(enc, pipe("resample"), [u, native.createView(), undefined, undefined, undefined, undefined, undefined, sampler, undefined, undefined, undefined, warped.createView()], Math.ceil(W / 8), Math.ceil(H / 8));
          }, true);
          gpu.release(native);
        }
        // Starting offsets per full-size tile, from the scan's field.
        const init = new Float32Array(tw * th * 2);
        for (let ty = 0; ty < th; ty++) for (let tx = 0; tx < tw; tx++) {
          const [dx, dy] = flowAt(a.flow, (tx * TILE + TILE / 2) / scanK, (ty * TILE + TILE / 2) / scanK, rw, rh);
          init[(ty * tw + tx) * 2] = dx * scanK; init[(ty * tw + tx) * 2 + 1] = dy * scanK;
        }
        gpu.device.queue.writeBuffer(initBuf, 0, init);
        const fv = frame.createView();
        await gpu.run("burst.frame", (enc, temp) => {
          const u = gpu.uniform(uni(a.g, 0, a)); temp.push(u);
          gpu.dispatch(enc, pipe("luma"), [u, fv, frmL.createView()], Math.ceil(W / 16), Math.ceil(H / 16));
          gpu.dispatch(enc, pipe("align"), [u, undefined, undefined, refL.createView(), frmL.createView(), initBuf, flowBuf], tw, th);
          gpu.dispatch(enc, pipe("weights"), [u, fv, undefined, undefined, undefined, undefined, flowBuf, sampler, acc, wbuf], Math.ceil(W / 16), Math.ceil(H / 8));
          gpu.dispatch(enc, pipe("accumulate"), [u, fv, undefined, undefined, undefined, undefined, flowBuf, sampler, acc, wbuf], Math.ceil(W / 8), Math.ceil(H / 8));
        }, true);
      } finally { gpu.release(F.work.tex, frame); }
      if (opt.cancelled()) throw new Error("cancelled");
    }
    gpu.release(wbuf, flowBuf, initBuf, refL, frmL);

    // Finalize: the mean, then ghost areas (few frames agreed) mixed toward a denoised version.
    opt.progress("series finish", undefined, 0.97);
    out = gpu.tex("working", W, H, "rgba16float");
    const dummy = gpu.tex("burst.dummy", 1, 1, "rgba16float");
    await gpu.run("burst.finalize", (enc, temp) => {
      const u = gpu.uniform(uni(1, 0)); temp.push(u);
      gpu.dispatch(enc, pipe("finalize"), [u, undefined, undefined, undefined, undefined, undefined, undefined, undefined, acc, undefined, dummy.createView(), out!.createView()], Math.ceil(W / 8), Math.ceil(H / 8));
    }, true);
    gpu.release(dummy);
    if (aligned.length >= 1) {
      const dn = await denoiseGPU(gpu, out, W, H, gain, noise);
      try {
        await gpu.run("burst.ghosts", (enc, temp) => {
          const u = gpu.uniform(uni(1, 1)); temp.push(u);
          gpu.dispatch(enc, pipe("finalize"), [u, undefined, undefined, undefined, undefined, undefined, undefined, undefined, acc, undefined, dn.createView(), out!.createView()], Math.ceil(W / 8), Math.ceil(H / 8));
        }, true);
      } finally { gpu.release(dn); }
    }
    gpu.release(acc);
    const used = aligned.length + 1;
    // Expected noise of the merge: one frame's ÷ √(frames), where they agreed.
    const merged = midSig / Math.sqrt(used);
    log.push(`series merge (${opt.mode}): ${used} frames in ${Math.round(performance.now() - t0)} ms; noise σ ≈ ${(midSig * 1000).toFixed(2)} → ${(merged * 1000).toFixed(2)} ×1e-3`);
    return { decoded: R.decoded, work: { ...R.work, tex: out, log }, file: ref.file, refIndex: ref.index, used, factor, noise: { single: midSig, merged } };
  } catch (e) {
    gpu.release(...temps, out);
    try { R.decoded.close(); } catch { /* closed */ }
    throw e;
  }
}

/** Develops one frame of the series alone (the A/B comparison's "one shot"). */
export async function developSingle(gpu: Gpu, file: File, factor: number): Promise<WorkingImage> {
  const { decoded, work } = await decodeDev(gpu, file, factor);
  decoded.close();
  return work;
}

