/** Questions the page asks about the photo: depth, focus, zones, light, colours, motion. */
import type { Engine } from "./engine.ts";
import { halvesToFloats } from "../gpu/half.ts";
import { objectDepthRange } from "../decision/focus.ts";
import type { Params } from "../decision/params.ts";
import { analyseColors, type ColorStats } from "../looks/palette.ts";
import { GROUPS } from "../neural/scene.ts";
import { neutralToneEq, TONE_EQ_DETAIL, type MaskHist, type ToneEqDetail } from "../tone/toneEq.ts";
import { type Session } from "./session.ts";

/** The object under x, y worth keeping whole in focus (person / animal / vehicle), as a group index. */
export function protectGroupAt(eng: Engine, x: number, y: number) : number | undefined {
  const s = eng.s;
  if (!s) return undefined;
  const seg = s.scene.seg, plane = seg.width * seg.height;
  const cx = Math.round(x * (seg.width - 1)), cy = Math.round(y * (seg.height - 1));
  for (const kind of ["person", "animal", "vehicle"] as const) {
    const g = GROUPS.indexOf(kind);
    let p = 0;
    for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) p += seg.probs[g * plane + Math.min(seg.height - 1, Math.max(0, cy + j)) * seg.width + Math.min(seg.width - 1, Math.max(0, cx + i))] / 9;
    if (p > 0.5) return g;
  }
  return undefined;
}

/** The tone-equalizer zone at x, y (0…1): its mask (view 9) at 384 px, read back. */
export async function toneEqZoneAt(eng: Engine, x: number, y: number) : Promise<number | undefined> {
  const s = eng.s;
  if (!s) return undefined;
  const t = await eng.ensureThumb(384);
  const p: Params = { ...s.params, toneEq: { ...(s.params.toneEq ?? neutralToneEq()), enabled: true }, enable: { ...s.params.enable, dof: false } };
  const r = await eng.renderer.render({ base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width }, s.maps, p,
    { wb: eng.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false, debugView: 9 }, false);
  const px = new Uint8Array(await eng.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4));
  const k = (Math.round(y * (t.h - 1)) * t.w + Math.round(x * (t.w - 1))) * 4;
  return Math.round((px[k + 1] / 255) * 8);
}

/**
 * Luminance histograms (32 bins of log2 scene luminance, −14 … +4 EV, the same
 * binning as the region statistics) for the near / middle / far bands, soft-
 * weighted as the renderer blends them. From the guide-resolution image the
 * refinement keeps, so it costs one small readback.
 */
/**
 * The tone equalizer's mask as histograms (EV, 0.1 EV bins over −16 … +4), one per
 * detail setting, as the photo renders now (the mask follows exposure and local tone):
 * view 10 at 256 px, read back.
 */
export async function toneEqHistograms(eng: Engine) : Promise<Record<ToneEqDetail, MaskHist> | undefined> {
  const s = eng.s;
  if (!s) return undefined;
  const t = await eng.ensureThumb(256);
  const lo = -16, hi = 4, nb = 200;
  const out = {} as Record<ToneEqDetail, MaskHist>;
  for (const d of TONE_EQ_DETAIL) {
    const p: Params = { ...s.params, toneEq: { ...(s.params.toneEq ?? neutralToneEq()), enabled: true, detail: d }, enable: { ...s.params.enable, dof: false } };
    const r = await eng.renderer.render({ base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width }, s.maps, p,
      { wb: eng.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false, debugView: 10 }, false);
    const px = new Uint8Array(await eng.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4));
    const bins = new Array(nb).fill(0);
    for (let i = 1; i < px.length; i += 4) bins[Math.min(nb - 1, Math.floor((px[i] / 255) * nb))]++;
    out[d] = { lo, hi, bins };
  }
  return out;
}

export async function depthBandHistograms(eng: Engine, s: Session, b1: number, b2: number) {
  const m = s.maps, d = s.distCPU!;
  const lin = new Float32Array(m.w * m.h * 4);
  halvesToFloats(new Uint16Array(await eng.gpu.readTexture(m.lin, 0, 0, m.w, m.h, 8)), lin);
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

/** The distance map averaged onto a coarse grid (for drawing on the page), and the vanishing point. */
export function depthField(eng: Engine, cols = 24) : { w: number; h: number; data: number[]; vanish: [number, number] } {
  const d = eng.s?.distCPU;
  if (!d) return { w: 1, h: 1, data: [0.5], vanish: [0.5, 0.5] };
  const w = Math.min(cols, d.w), h = Math.max(1, Math.round((w * d.h) / d.w));
  const sum = new Float64Array(w * h), n = new Float64Array(w * h);
  for (let y = 0; y < d.h; y++) for (let x = 0; x < d.w; x++) {
    const k = Math.min(h - 1, Math.floor((y * h) / d.h)) * w + Math.min(w - 1, Math.floor((x * w) / d.w));
    sum[k] += d.data[y * d.w + x]; n[k]++;
  }
  return { w, h, data: Array.from(sum, (v, i) => (n[i] ? v / n[i] : 0.5)), vanish: eng.renderer.vanishing };
}

/**
 * A layer's mask (view 7 at 384 px) on the depth grid, and where the part it covers
 * recedes to: the centre of its farthest tenth (mask-weighted) — the far end of a
 * train, not the sky above it. Nothing covered: the photo's own vanishing point.
 */
export async function motionField(eng: Engine, layer: number) : Promise<{ w: number; h: number; data: number[]; mask: number[]; vanish: [number, number]; range: [number, number]; reach: number }> {
  const base = eng.depthField();
  const s = eng.s, d = s?.distCPU;
  if (!s || !d || layer < 0) return { ...base, mask: base.data.map(() => 1), range: [0, 1], reach: 0.6 };
  await eng.ensureSelections(s, s.params);
  const t = await eng.ensureThumb(384);
  const r = await eng.renderer.render({ base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width }, s.maps, s.params,
    { wb: eng.wbFor(s.params), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false, debugView: 7, region: layer }, false);
  const px = new Uint8Array(await eng.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4));
  const m = new Float32Array(t.w * t.h), dist = new Float32Array(t.w * t.h);
  for (let y = 0; y < t.h; y++) for (let x = 0; x < t.w; x++) {
    const i = y * t.w + x;
    m[i] = px[i * 4 + 1] / 255;
    dist[i] = d.data[Math.min(d.h - 1, Math.floor((y * d.h) / t.h)) * d.w + Math.min(d.w - 1, Math.floor((x * d.w) / t.w))];
  }
  // The farthest and the nearest tenth of what the mask covers (by mask weight); the
  // vanishing point lies beyond the far end, along the part's own axis (near → far):
  // its perspective lines meet past it, and the whole part stays on one side of the
  // point — one direction of motion, no streaks fanning out in the middle of it.
  const idx = Array.from(m.keys()).filter((i) => m[i] > 0.25).sort((a, b) => dist[b] - dist[a]);
  let vanish = eng.renderer.vanishing;
  if (idx.length) {
    const total = idx.reduce((a, i) => a + m[i], 0);
    const centre = (order: number[]): [number, number] => {
      let acc = 0, sx = 0, sy = 0, sw = 0;
      for (const i of order) {
        if (acc > total * 0.1) break;
        acc += m[i]; sx += ((i % t.w) + 0.5) * m[i]; sy += (Math.floor(i / t.w) + 0.5) * m[i]; sw += m[i];
      }
      return sw > 0 ? [sx / sw / t.w, sy / sw / t.h] : [0.5, 0.5];
    };
    const far = centre(idx), near = centre([...idx].reverse());
    // Past the far end by 40 % of the part's length (in picture proportions, then back).
    const aspect = t.w / t.h;
    const ax = (far[0] - near[0]) * aspect, ay = far[1] - near[1];
    const len = Math.hypot(ax, ay);
    vanish = len > 0.02 ? [far[0] + (ax * 0.4) / aspect, far[1] + ay * 0.4] : far;
  }
  // The part's own depth (5th … 95th percentile, near to far) and its extent from the
  // vanishing point (95th percentile, in heights).
  let range: [number, number] = [0, 1], reach = 0.6;
  if (idx.length) {
    const at = (q: number) => idx[Math.min(idx.length - 1, Math.floor(idx.length * q))];
    range = [dist[at(0.95)], dist[at(0.05)]];
    const aspect = t.w / t.h;
    const rs = idx.map((i) => Math.hypot((((i % t.w) + 0.5) / t.w - vanish[0]) * aspect, (Math.floor(i / t.w) + 0.5) / t.h - vanish[1])).sort((a, b) => a - b);
    reach = rs[Math.floor(rs.length * 0.95)] ?? 0.6;
  }
  const mask = new Array(base.w * base.h).fill(0), n = new Array(base.w * base.h).fill(0);
  for (let y = 0; y < t.h; y++) for (let x = 0; x < t.w; x++) {
    const k = Math.min(base.h - 1, Math.floor((y * base.h) / t.h)) * base.w + Math.min(base.w - 1, Math.floor((x * base.w) / t.w));
    mask[k] += m[y * t.w + x]; n[k]++;
  }
  return { ...base, mask: mask.map((v, k) => (n[k] ? v / n[k] : 0)), vanish, range, reach };
}

export function focusAt(eng: Engine, x: number, y: number) : number | undefined {
  const d = eng.s?.distCPU;
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
export function focusRangeAt(eng: Engine, x: number, y: number) : { dist: number; range: [number, number] } | undefined {
  const s = eng.s, d = s?.distCPU;
  const d0 = eng.focusAt(x, y);
  if (!s || !d || d0 === undefined) return undefined;
  return { dist: d0, range: objectDepthRange(d, s.scene.seg, x, y, d0) };
}

/**
 * Where the light probably is (a lens flare's default): the brightest spot in the
 * upper 60 % of the photo as edited, on a 256 px render, smoothed so a single
 * glint does not win over the sun.
 */
export async function brightestPoint(eng: Engine) : Promise<{ x: number; y: number }> {
  const s = eng.s;
  if (!s) return { x: 0.3, y: 0.2 };
  const t = await eng.ensureThumb(256);
  const p = s.params;
  const r = await eng.renderer.render({ base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width }, s.maps, p, { wb: eng.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false }, false);
  const px = new Uint8Array(await eng.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4));
  const { w, h } = t, R = 4;
  let best = -1, bx = 0.3, by = 0.2;
  for (let y = R; y < Math.floor(h * 0.6); y += 2) for (let x = R; x < w - R; x += 2) {
    let sum = 0;
    for (let j = -R; j <= R; j += 2) for (let i = -R; i <= R; i += 2) { const o = ((y + j) * w + x + i) * 4; sum += Math.max(px[o], px[o + 1], px[o + 2]); }
    if (sum > best) { best = sum; bx = x / (w - 1); by = y / (h - 1); }
  }
  return { x: Math.round(bx * 1000) / 1000, y: Math.round(by * 1000) / 1000 };
}

export async function palette(eng: Engine) : Promise<ColorStats> {
  const s = eng.s;
  if (!s) throw new Error("No photo open");
  const px = await eng.technicalPixels(384);
  return analyseColors(px.data, undefined, 7);
}
