/** Looks: thumbnails, and a look made from (or matched to) a reference photo. */
import type { Engine } from "./engine.ts";
import { decodeFile } from "../decode/decode.ts";
import { analyseScene, type SceneMaps, GROUPS, type Group } from "../neural/scene.ts";
import type { Params } from "../decision/params.ts";
import type { RenderSource } from "../render/renderer.ts";
import { parseCube } from "../render/looks.ts";
import { neutralProfile, normalizeProfile, type LookProfile } from "../looks/profile.ts";
import { analyseColors, type ColorStats } from "../looks/palette.ts";
import { matchProfile, profileFromReference, type RegionColors } from "../looks/reference.ts";

export function importLook(eng: Engine, name: string, text: string) {
  eng.renderer.addLook(parseCube(text, name));
  return eng.looks();
}

/** Renders the current photo at thumbnail size through each profile (same GPU path as the preview). */
export async function thumbnails(eng: Engine, profiles: LookProfile[], long: number) {
  const s = eng.s;
  if (!s) return [];
  const t = await eng.ensureThumb(long);
  const src: RenderSource = { base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width };
  const items: Array<{ id: string; width: number; height: number; data: ArrayBuffer }> = [];
  const t0 = performance.now();
  for (const raw of profiles) {
    const prof = normalizeProfile(raw);
    const p: Params = { ...s.params, profile: prof, enable: { ...s.params.enable, dof: false, sharpen: false, lut: true } };
    const r = await eng.renderer.render(src, s.maps, p, { wb: eng.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38" }, false);
    items.push({ id: prof.id, width: t.w, height: t.h, data: await eng.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4) });
  }
  eng.log(`look previews: ${profiles.length} profiles at ${t.w}×${t.h} in ${(performance.now() - t0).toFixed(0)} ms`);
  return items;
}

/** The technical rendering (no creative profile) at a small size, P3-encoded RGBA8. */
export async function technicalPixels(eng: Engine, long = 384) {
  const s = eng.s!;
  const t = await eng.ensureThumb(long);
  const p: Params = { ...s.params, profile: neutralProfile(), enable: { ...s.params.enable, dof: false, lut: false } };
  const r = await eng.renderer.render({ base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width }, s.maps, p, { wb: eng.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false }, false);
  const data = new Uint8Array(await eng.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4));
  return { data, w: t.w, h: t.h };
}

/** Weight = 1 − P(people) − P(sky): the global look statistics exclude regions that are matched (sky) or protected (people) separately. */
export function generalWeights(seg: SceneMaps["seg"], w: number, h: number) : Float32Array {
  const p = groupWeights(seg, w, h, GROUPS.indexOf("person"));
  const s = groupWeights(seg, w, h, GROUPS.indexOf("sky"));
  for (let i = 0; i < p.length; i++) p[i] = Math.max(0.02, 1 - p[i] - s[i]);
  return p;
}

/** Per-pixel weights of a semantic group, sampled from network-resolution probabilities. */
export function groupWeights(seg: SceneMaps["seg"], w: number, h: number, g: number, invert = false) : Float32Array {
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

export function regionColors(rgba: Uint8Array, w: number, h: number, seg: SceneMaps["seg"]) : RegionColors {
  const out: RegionColors = {};
  for (const g of ["sky", "vegetation", "water", "building", "terrain", "ground"] as Group[]) {
    const gi = GROUPS.indexOf(g);
    const wts = groupWeights(seg, w, h, gi);
    let mass = 0;
    for (const v of wts) mass += v;
    if (mass / (w * h) < 0.02) continue;
    const st = analyseColors(rgba, wts, 3);
    out[g] = { ...st, n: Math.round(mass) };
  }
  return out;
}

/** Reference image → editable profile (create) or a profile matching the current photo toward it (match). */
export async function reference(eng: Engine, file: File, mode: "create" | "match", amount: number) : Promise<{ profile: LookProfile; reference: ColorStats; message: string }> {
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
    const refScene = await analyseScene(eng.neural, { rgba: f, width: w, height: h }, (st) => eng.progress("reference " + st), false);
    const refStats = analyseColors(refPx, generalWeights(refScene.seg, w, h), 7);
    const refRegions = regionColors(refPx, w, h, refScene.seg);
    if (mode === "create" || !eng.s) {
      const profile = profileFromReference(refStats, file.name, refRegions);
      return { profile, reference: refStats, message: `Profile built from ${file.name}` };
    }
    const s = eng.s;
    const px = await eng.technicalPixels(384);
    const srcStats = analyseColors(px.data, generalWeights(s.scene.seg, px.w, px.h), 7);
    const srcRegions = regionColors(px.data, px.w, px.h, s.scene.seg);
    const profile = matchProfile(srcStats, refStats, file.name, srcRegions, refRegions, amount);
    return { profile, reference: refStats, message: profile.description ?? "" };
  } finally {
    dec.close();
  }
}
