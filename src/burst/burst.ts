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
import { alignTiles, exposureRatio, fitGlobal, flowAt, pyramid, regularise, scanNoise, sharpness, texturedTiles, warp, type Flow, type Plane } from "./align.ts";

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

interface Scan { index: number; file: File; y: Plane; ev: number; sharp: number; exposure?: number }

async function decodeDev(gpu: Gpu, file: File, factor: number | ((w: number, h: number) => number)): Promise<{ decoded: DecodedImage; work: WorkingImage }> {
  const decoded = await decodeFile(new Uint8Array(await file.arrayBuffer()), file.name, file.type);
  try {
    const f = typeof factor === "number" ? factor : factor(decoded.source.width, decoded.source.height);
    const work = await develop(gpu, decoded, { factor: f });
    return { decoded, work };
  } catch (e) { decoded.close(); throw e; }
}

/** Luma plane (linear) of a working texture at scan size. */
async function scanOf(gpu: Gpu, work: WorkingImage): Promise<Plane> {
  const k = Math.min(1, SCAN_LONG / Math.max(work.width, work.height));
  const w = Math.max(TILE * 4, Math.round(work.width * k)), h = Math.max(TILE * 4, Math.round(work.height * k));
  const t = await downsample(gpu, work.tex, work.width, work.height, w, h, false, 1, "burst.scan");
  const f = halvesToFloats(new Uint16Array(await gpu.readTexture(t, 0, 0, w, h, 8)));
  gpu.release(t);
  const d = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) d[i] = Math.max(0, 0.2627 * f[i * 4] + 0.678 * f[i * 4 + 1] + 0.0593 * f[i * 4 + 2]);
  return { w, h, d };
}

const sqrtPlane = (p: Plane, g: number): Plane => ({ w: p.w, h: p.h, d: p.d.map((v) => Math.sqrt(Math.max(0, v * g))) });
const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
/** Brightness of a scan (median of mid-grey-ish pixels, robust to framing shifts). */
const level = (p: Plane) => { const s: number[] = []; for (let i = 0; i < p.d.length; i += 7) s.push(p.d[i]); return Math.max(1e-5, median(s)); };

export async function mergeSeries(gpu: Gpu, files: File[], opt: SeriesOptions): Promise<SeriesResult> {
  const t0 = performance.now();
  const n = files.length;
  // ------------------------------------------------------------------ pass A: scan
  const scans: Scan[] = [];
  let W = 0, H = 0, factor = 0;
  for (let i = 0; i < n; i++) {
    opt.progress("series scan", `${i + 1}/${n}`, (i / n) * 0.35);
    const { decoded, work } = await decodeDev(gpu, files[i], factor || opt.factorFor);
    factor ||= work.factor;
    try {
      decoded.close();
      if (i === 0 || !W) { W = work.width; H = work.height; }
      if (work.width !== W || work.height !== H) { opt.log(`series: ${files[i].name} skipped (${work.width}×${work.height}, not ${W}×${H})`); continue; }
      const y = await scanOf(gpu, work);
      const m = decoded.meta;
      scans.push({ index: i, file: files[i], y, ev: Math.log2(level(y)), sharp: 0, exposure: m.exposureTime && m.iso ? m.exposureTime * m.iso : undefined });
    } finally { gpu.release(work.tex); }
    if (opt.cancelled()) throw new Error("cancelled");
  }
  if (!scans.length) throw new Error("No usable frames in the series");

  // Reference: among frames at the series' usual exposure (brackets aside), the sharpest.
  const evMid = median(scans.map((s) => s.ev));
  for (const s of scans) s.sharp = sharpness(sqrtPlane(s.y, 1 / 2 ** evMid));
  const byUser = opt.ref !== undefined ? scans.find((s) => s.index === opt.ref) : undefined;
  const ref = byUser ?? scans.filter((s) => Math.abs(s.ev - evMid) < 0.35).sort((a, b) => b.sharp - a.sharp)[0] ?? scans[0];
  opt.log(`series: ${scans.length} frames ${W}×${H}; reference ${ref.file.name}${byUser ? " (chosen)" : " (sharpest)"}; levels ${scans.map((s) => (s.ev - ref.ev).toFixed(2)).join(", ")} EV`);

  // Coarse alignment on the scans, and each frame's exposure ratio.
  const refPyr = pyramid(sqrtPlane(ref.y, 1));
  const aligned: Array<{ scan: Scan; flow: Flow; g: number }> = [];
  const warpedY: Plane[] = [];
  let textured: Uint8Array | undefined;
  for (const s of scans) {
    if (s === ref) continue;
    let g = 2 ** (ref.ev - s.ev);
    const raw = alignTiles(refPyr, pyramid(sqrtPlane(s.y, g)));
    textured ??= texturedTiles(refPyr[0], raw, 0.01);
    const model = fitGlobal(raw, refPyr[0].w, refPyr[0].h, textured);
    const flow = regularise(raw, model, refPyr[0].w, refPyr[0].h, textured, 2);
    const wy = warp(s.y, flow);
    g = exposureRatio(ref.y, wy);
    const shift = Math.hypot(model.tx, model.ty) * (W / ref.y.w);
    const rot = (Math.atan2(model.b, model.a) * 180) / Math.PI;
    opt.log(`series: ${s.file.name} shift ${shift.toFixed(1)} px, rotation ${rot.toFixed(2)}°, exposure ×${g.toFixed(3)}`);
    aligned.push({ scan: s, flow, g });
    warpedY.push({ ...wy, d: wy.d.map((v) => v * g) });
  }
  const sig = scanNoise(ref.y, warpedY);
  const midSig = sig[5];
  opt.log(`series: noise σ per √Y bin (scan) ${sig.map((v) => (v * 1000).toFixed(2)).join(" ")} ×1e-3`);
  if (opt.cancelled()) throw new Error("cancelled");

  // ------------------------------------------------------------------ pass B: merge
  const pipe = (e: string) => gpu.pipeline("burst", burstWgsl, e);
  const sampler = gpu.device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
  const tw = Math.ceil(W / TILE), th = Math.ceil(H / TILE);
  const scanK = W / ref.y.w;
  const uni = (g: number, mode = 0) => new Uniforms(24).u32(W, H, tw, th).f32(g, scanK / 3, aligned.length + 1).u32(mode).f32(...sig).bytes();

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
      const F = await decodeDev(gpu, a.scan.file, factor);
      F.decoded.close();
      try {
        // Starting offsets per full-size tile, from the scan's field.
        const init = new Float32Array(tw * th * 2);
        for (let ty = 0; ty < th; ty++) for (let tx = 0; tx < tw; tx++) {
          const [dx, dy] = flowAt(a.flow, (tx * TILE + TILE / 2) / scanK, (ty * TILE + TILE / 2) / scanK, ref.y.w, ref.y.h);
          init[(ty * tw + tx) * 2] = dx * scanK; init[(ty * tw + tx) * 2 + 1] = dy * scanK;
        }
        gpu.device.queue.writeBuffer(initBuf, 0, init);
        const fv = F.work.tex.createView();
        await gpu.run("burst.frame", (enc, temp) => {
          const u = gpu.uniform(uni(a.g)); temp.push(u);
          gpu.dispatch(enc, pipe("luma"), [u, fv, frmL.createView()], Math.ceil(W / 16), Math.ceil(H / 16));
          gpu.dispatch(enc, pipe("align"), [u, undefined, undefined, refL.createView(), frmL.createView(), initBuf, flowBuf], tw, th);
          gpu.dispatch(enc, pipe("weights"), [u, fv, undefined, undefined, undefined, undefined, flowBuf, sampler, acc, wbuf], Math.ceil(W / 16), Math.ceil(H / 8));
          gpu.dispatch(enc, pipe("accumulate"), [u, fv, undefined, undefined, undefined, undefined, flowBuf, sampler, acc, wbuf], Math.ceil(W / 8), Math.ceil(H / 8));
        }, true);
      } finally { gpu.release(F.work.tex); }
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
      await gpu.run("burst.ghosts", (enc, temp) => {
        const u = gpu.uniform(uni(1, 1)); temp.push(u);
        gpu.dispatch(enc, pipe("finalize"), [u, undefined, undefined, undefined, undefined, undefined, undefined, undefined, acc, undefined, dn.createView(), out!.createView()], Math.ceil(W / 8), Math.ceil(H / 8));
      }, true);
      gpu.release(dn);
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

