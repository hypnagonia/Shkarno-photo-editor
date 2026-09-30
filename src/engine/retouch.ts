/** Magic brush in the engine: the working image as the strokes say (fills, and their undo). */
import type { Engine } from "./engine.ts";
import { floatsToHalves, halvesToFloats } from "../gpu/half.ts";
import type { Params, RetouchStroke } from "../decision/params.ts";
import { featherMask, prefixKeys, resize, strokeMask, strokeRect, type Rect } from "../retouch/geometry.ts";
import { WorkerInpainter, type Inpainter } from "../retouch/inpaint.ts";
import { fromDisplay, toDisplay } from "../restore/display.ts";
import { phoneForced } from "../device.ts";
import { type Session, isMobile } from "./session.ts";

/**
 * Magic brush: the working image as `p.retouch` says. Strokes already filled while the
 * ones before them are unchanged stay; from the first that differs, fills are undone
 * (last first, their pixels put back) and the rest filled again, in order.
 */
export async function ensureRetouch(eng: Engine, s: Session, p: Params) {
  const want = p.retouch ?? [];
  if (!want.length && !s.retouch?.applied.length) return;
  const st = (s.retouch ??= { applied: [] });
  const keys = prefixKeys(want);
  let k = 0;
  while (k < st.applied.length && k < keys.length && st.applied[k].key === keys[k]) k++;
  // (A stroke that failed is not tried again until the strokes change.)
  const done = k === st.applied.length && (k === keys.length || st.failed === keys[k]);
  if (done) return;
  for (let i = st.applied.length - 1; i >= k; i--) eng.writeCrop(s, st.applied[i].rect, st.applied[i].before);
  st.applied.length = k;
  st.failed = undefined;
  for (let i = k; i < want.length; i++) {
    eng.progress("retouch", want.length - k > 1 ? `${i - k + 1}/${want.length - k}` : undefined);
    try { st.applied.push({ key: keys[i], ...(await eng.inpaintStroke(s, st, want[i])) }); }
    catch (e) {
      st.failed = keys[i];
      const msg = e instanceof Error ? e.message : String(e);
      eng.log(`magic brush failed: ${msg}`);
      eng.post({ type: "error", message: `Magic brush: ${msg}`, stage: "retouch" });
      break;
    }
  }
  eng.dropThumb();
  await eng.makeProxy();
}

/** Fills one stroke: its region to display RGB, the network, then laid back in linear light inside a soft edge. */
export async function inpaintStroke(eng: Engine, s: Session, st: NonNullable<Session["retouch"]>, stroke: RetouchStroke) : Promise<{ rect: Rect; before: Uint16Array[] }> {
  const W = s.work.width, H = s.work.height;
  const rect = strokeRect(stroke, W, H);
  const texes = s.denoised !== s.work.tex ? [s.work.tex, s.denoised] : [s.work.tex];
  const before: Uint16Array[] = [];
  for (const t of texes) before.push(new Uint16Array(await eng.gpu.readTexture(t, rect.x, rect.y, rect.w, rect.h, 8)));
  const n = rect.w * rect.h;
  // The network sees the restored image (cleaner), in display RGB.
  const src = halvesToFloats(before[before.length - 1]);
  const img = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) img[i * 3 + c] = toDisplay(src[i * 4 + c] * s.gain);
  const long = Math.max(W, H);
  const hole = strokeMask(stroke, W, H, rect, rect.w, rect.h, Math.max(2, Math.round(long / 1000)));
  const t0 = performance.now();
  const fill = await (await painterFor(eng, s)).run(img, rect.w, rect.h, hole);
  eng.log(`magic brush: ${rect.w}×${rect.h} region filled in ${Math.round(performance.now() - t0)} ms`);
  const a = featherMask(hole, rect.w, rect.h, Math.max(3, rect.w / 128));
  const lin = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) lin[i] = fromDisplay(fill[i]) / s.gain;
  texes.forEach((t, j) => {
    const f = halvesToFloats(before[j]);
    for (let i = 0; i < n; i++) {
      const w = a[i];
      if (w <= 0) continue;
      for (let c = 0; c < 3; c++) f[i * 4 + c] += (lin[i * 3 + c] - f[i * 4 + c]) * w;
      f[i * 4 + 3] *= 1 - w; // (alpha: the clipped share — a fill is not clipped)
    }
    eng.writeTexels(t, rect, floatsToHalves(f));
  });
  return { rect, before };
}

/** Puts a fill's replaced pixels back (into the working texture, and the restored one when separate). */
export function writeCrop(eng: Engine, s: Session, rect: Rect, before: Uint16Array[]) {
  const texes = s.denoised !== s.work.tex ? [s.work.tex, s.denoised] : [s.work.tex];
  texes.forEach((t, j) => eng.writeTexels(t, rect, before[j] ?? before[0]));
}

export function writeTexels(eng: Engine, t: GPUTexture, rect: Rect, data: Uint16Array) {
  eng.gpu.device.queue.writeTexture({ texture: t, origin: { x: rect.x, y: rect.y } }, data as Uint16Array<ArrayBuffer>, { bytesPerRow: rect.w * 8, rowsPerImage: rect.h }, { width: rect.w, height: rect.h });
}

/** The 2× stage swapped the working image: the fills' saved pixels and regions follow (bilinear — undo there is a touch softer). */
export function scaleRetouch(eng: Engine, s: Session, k: number) {
  const st = s.retouch;
  if (!st?.applied.length) return;
  const texes = s.denoised !== s.work.tex ? 2 : 1;
  st.applied = st.applied.map((a) => {
    const r = { x: a.rect.x * k, y: a.rect.y * k, w: a.rect.w * k, h: a.rect.h * k };
    // (One texture now: the upscaler worked from the restored one, the last kept.)
    const before = a.before.slice(a.before.length - texes).map((b) => floatsToHalves(resize(halvesToFloats(b), a.rect.w, a.rect.h, 4, r.w, r.h)));
    return { key: a.key, rect: r, before };
  });
}

/**
 * The photo's inpainting network (LaMa on computers, MI-GAN on phones), shared by the magic
 * brush and a moving object's background. On a phone, tap-to-select's worker is ended
 * first: two networks' runtimes side by side (≈ 450 MB each, never shrinking) went past
 * the page's memory budget when a stroke came within 30 s of a selection. It restarts
 * from the embedding the engine keeps.
 */
export async function painterFor(eng: Engine, s: Session): Promise<Inpainter> {
  if (isMobile() && s.sel?.sam.running) {
    s.sel.sam.dispose();
    await new Promise((r) => setTimeout(r, 500)); // (its memory goes back a moment after it ends)
  }
  const st = (s.retouch ??= { applied: [] });
  const model = isMobile() ? "migan" : "lama";
  return (st.painter ??= new WorkerInpainter(model, eng.base, phoneForced(), {
    log: (t) => eng.log(t),
    progress: (loaded, total) => eng.progress(`download ${model}`, `${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`, loaded / total),
  }));
}
