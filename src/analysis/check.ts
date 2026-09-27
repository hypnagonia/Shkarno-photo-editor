/**
 * Check: the finished photo, looked at for technical mistakes.
 *
 * Two renders at the same size: the edit as it will be exported (`final`) and
 * the camera's own rendering (`before`: no edits). Every check measures the
 * final photo and, where it matters, compares it with the camera's, so a
 * problem the edits made ("highlights blown by the edit") is told apart from one
 * the scene had anyway (a lamp that was always white).
 *
 * Each finding: ok / warn / bad, its numbers (the UI words them), and where it
 * is (a mask at the image size) when that can be shown on the photo.
 */
import { linSrgbToOklab } from "../color/oklab.ts";
import { srgbEotf } from "../color/transfer.ts";
import type { FixChange } from "./checkFix.ts";

export interface CheckImage { rgba: Uint8Array; w: number; h: number }
export interface CheckInput {
  final: CheckImage;
  before: CheckImage;
  /** Scene analysis: group probabilities (plane per group) and the group names. */
  seg?: { width: number; height: number; probs: Float32Array; groups: readonly string[] };
  /** The scene's light level: EV at ISO 100 from the photo's exposure settings (absent when unknown). */
  scene?: { ev?: number };
}
export type CheckLevel = "ok" | "warn" | "bad";
export interface CheckItem {
  id: "highlights" | "shadows" | "colorClip" | "saturation" | "skin" | "cast" | "exposure" | "contrast" | "noise" | "halos" | "banding" | "vignette";
  level: CheckLevel;
  /** Numbers for the wording (percentages already ×100, rounded). */
  v: Record<string, number | string>;
  /** Where (1 = here), at the image size; only for findings that are not ok. */
  mask?: Uint8Array;
  /** What fixes it, worked out by re-rendering (engine): control values or a layer's opacity. */
  fix?: FixChange[];
  /** The fix only makes it better, not fine. */
  fixPartial?: boolean;
  /**
   * How far from a comfortable result (> 0: needs fixing, ≤ 0: comfortably fine),
   * signed so that moving the right control crosses zero once: what the fix
   * solver bisects. Absent for findings no control is solved for.
   */
  err?: number;
}

/** Linear Display P3 → linear sRGB (as P3_TO_SRGB in common.wgsl). */
const M = [1.2249401, -0.2249404, 0, -0.0420569, 1.0420571, 0, -0.0196376, -0.0786361, 1.0982735];

interface Px { L: Float32Array; a: Float32Array; b: Float32Array; C: Float32Array; h: Float32Array; min: Uint8Array; max: Uint8Array }
function toLab(img: CheckImage): Px {
  const n = img.w * img.h;
  const L = new Float32Array(n), a = new Float32Array(n), b = new Float32Array(n), C = new Float32Array(n), h = new Float32Array(n);
  const mn = new Uint8Array(n), mx = new Uint8Array(n);
  const lut = new Float32Array(256).map((_, i) => srgbEotf(i / 255));
  for (let i = 0; i < n; i++) {
    const r8 = img.rgba[i * 4], g8 = img.rgba[i * 4 + 1], b8 = img.rgba[i * 4 + 2];
    const r = lut[r8], g = lut[g8], bl = lut[b8];
    const lab = linSrgbToOklab([M[0] * r + M[1] * g + M[2] * bl, M[3] * r + M[4] * g + M[5] * bl, M[6] * r + M[7] * g + M[8] * bl]);
    L[i] = lab[0]; a[i] = lab[1]; b[i] = lab[2];
    C[i] = Math.hypot(lab[1], lab[2]);
    let hh = (Math.atan2(lab[2], lab[1]) * 180) / Math.PI;
    if (hh < 0) hh += 360;
    h[i] = hh;
    mn[i] = Math.min(r8, g8, b8); mx[i] = Math.max(r8, g8, b8);
  }
  return { L, a, b, C, h, min: mn, max: mx };
}

const pct = (x: number) => Math.round(x * 1000) / 10;
function quantile(v: Float32Array, q: number, step = 7): number {
  const s: number[] = [];
  for (let i = 0; i < v.length; i += step) s.push(v[i]);
  s.sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))] ?? 0;
}
/** The scene group most of `mask` falls in (for "blown in the sky"). */
function whereName(mask: Uint8Array, w: number, h: number, seg?: CheckInput["seg"]): string {
  if (!seg) return "";
  const plane = seg.width * seg.height;
  const acc = new Float64Array(seg.groups.length);
  let n = 0;
  for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
    if (!mask[y * w + x]) continue;
    const k = Math.min(seg.height - 1, Math.floor((y / h) * seg.height)) * seg.width + Math.min(seg.width - 1, Math.floor((x / w) * seg.width));
    for (let g = 0; g < seg.groups.length; g++) acc[g] += seg.probs[g * plane + k];
    n++;
  }
  if (!n) return "";
  let best = 0;
  for (let g = 1; g < acc.length; g++) if (acc[g] > acc[best]) best = g;
  return acc[best] / n > 0.35 ? seg.groups[best] : "";
}
function segAt(seg: CheckInput["seg"], group: string, x: number, y: number, w: number, h: number): number {
  if (!seg) return 0;
  const g = seg.groups.indexOf(group);
  if (g < 0) return 0;
  const k = Math.min(seg.height - 1, Math.floor((y / h) * seg.height)) * seg.width + Math.min(seg.width - 1, Math.floor((x / w) * seg.width));
  return seg.probs[g * seg.width * seg.height + k];
}
/** Box average of a plane with radius r (summed-area table). */
function boxMean(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const W1 = w + 1, sat = new Float64Array(W1 * (h + 1));
  for (let y = 0; y < h; y++) { let row = 0; for (let x = 0; x < w; x++) { row += src[y * w + x]; sat[(y + 1) * W1 + x + 1] = sat[y * W1 + x + 1] + row; } }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
      out[y * w + x] = (sat[y1 * W1 + x1] - sat[y0 * W1 + x1] - sat[y1 * W1 + x0] + sat[y0 * W1 + x0]) / ((x1 - x0) * (y1 - y0));
    }
  }
  return out;
}

export type CheckId = CheckItem["id"];
/** The Lab planes of a render (computed once, e.g. for the camera's rendering while solving). */
export type CheckPlanes = Px;
export const planesOf = (img: CheckImage): CheckPlanes => toLab(img);

/** Kinds of scene by their light, and the median brightness (OkLab L) that looks natural for each. */
export type SceneKind = "night" | "dim" | "indoor" | "overcast" | "day" | "bright";
export const SCENE_BAND: Record<SceneKind, [number, number]> = {
  night: [0.22, 0.4], dim: [0.32, 0.5], indoor: [0.42, 0.6], overcast: [0.46, 0.63], day: [0.48, 0.67], bright: [0.58, 0.8],
};
/**
 * The scene, from its light level (EV at ISO 100, from shutter, aperture and ISO:
 * sun ≈ 15, overcast ≈ 12, indoors ≈ 7–9, night streets ≈ 3–5) and the camera's own
 * rendering; without exposure settings (screenshots, re-saved files) from the
 * camera's rendering alone.
 */
export function sceneKind(ev: number | undefined, cameraMedian: number, skyShare: number): SceneKind {
  if (ev === undefined || !Number.isFinite(ev)) {
    return cameraMedian < 0.26 ? "night" : cameraMedian < 0.38 ? "dim" : cameraMedian > 0.7 ? "bright" : "day";
  }
  if (ev < 5) return "night";
  if (ev < 7.5) return "dim";
  if (ev < 10) return skyShare > 0.08 ? "overcast" : "indoor";
  if (ev < 12.5) return skyShare > 0.08 ? "overcast" : "day";
  return ev >= 14 && cameraMedian > 0.55 ? "bright" : "day";
}

export function checkPhoto(inp: CheckInput, only?: CheckId, beforePlanes?: CheckPlanes): CheckItem[] {
  const { w, h } = inp.final;
  const n = w * h;
  const F = toLab(inp.final), B = beforePlanes ?? toLab(inp.before);
  const want = (id: CheckId) => !only || only === id;
  const items: CheckItem[] = [];
  const mask = () => new Uint8Array(n);

  // ---- highlights: blown white (every channel at the top), versus the camera.
  if (want("highlights")) {
    const m = mask(); let f = 0, b = 0;
    for (let i = 0; i < n; i++) { if (F.min[i] >= 250) { f++; if (B.min[i] < 250) m[i] = 1; } if (B.min[i] >= 250) b++; }
    const added = f / n - b / n;
    const level: CheckLevel = added > 0.01 ? "bad" : added > 0.003 || f / n > 0.03 ? "warn" : "ok";
    items.push({ id: "highlights", level, err: added - 0.001, v: { pct: pct(f / n), camera: pct(b / n), where: whereName(m, w, h, inp.seg) }, mask: level === "ok" ? undefined : (added > 0.003 ? m : Uint8Array.from(F.min, (v) => (v >= 250 ? 1 : 0))) });
  }
  // ---- shadows: crushed black (every channel at the bottom), versus the camera.
  if (want("shadows")) {
    const m = mask(); let f = 0, b = 0;
    for (let i = 0; i < n; i++) { if (F.max[i] <= 4) { f++; if (B.max[i] > 4) m[i] = 1; } if (B.max[i] <= 4) b++; }
    const added = f / n - b / n;
    const level: CheckLevel = added > 0.03 ? "bad" : added > 0.008 ? "warn" : "ok";
    items.push({ id: "shadows", level, err: added - 0.003, v: { pct: pct(f / n), camera: pct(b / n), where: whereName(m, w, h, inp.seg) }, mask: level === "ok" ? undefined : m });
  }
  // ---- colour clipping: a saturated colour with a channel at the end of its range,
  //      in the mid tones (texture gone: flat patches of pure colour).
  if (want("colorClip")) {
    const m = mask(); let f = 0, b = 0;
    for (let i = 0; i < n; i++) {
      const clipF = (F.min[i] <= 1 || F.max[i] >= 254) && F.C[i] > 0.12 && F.L[i] > 0.25 && F.L[i] < 0.92;
      const clipB = (B.min[i] <= 1 || B.max[i] >= 254) && B.C[i] > 0.12 && B.L[i] > 0.25 && B.L[i] < 0.92;
      if (clipF) { f++; if (!clipB) m[i] = 1; }
      if (clipB) b++;
    }
    const added = (f - b) / n;
    const level: CheckLevel = added > 0.02 ? "bad" : added > 0.005 ? "warn" : "ok";
    items.push({ id: "colorClip", level, err: added - 0.002, v: { pct: pct(f / n), camera: pct(b / n), where: whereName(m, w, h, inp.seg) }, mask: level === "ok" ? undefined : m });
  }
  // ---- saturation overall, versus the camera.
  if (want("saturation")) {
    let cf = 0, cb = 0, loud = 0;
    const m = mask();
    for (let i = 0; i < n; i++) { cf += F.C[i]; cb += B.C[i]; if (F.C[i] > 0.24 && F.C[i] > B.C[i] * 1.3) { loud++; m[i] = 1; } }
    const ratio = cb > 1e-6 ? cf / cb : 1;
    const level: CheckLevel = ratio > 1.6 || loud / n > 0.08 ? "bad" : ratio > 1.3 || loud / n > 0.03 ? "warn" : ratio < 0.6 ? "warn" : "ok";
    items.push({ id: "saturation", level, err: ratio < 0.8 ? 0.75 - ratio : Math.max(ratio - 1.2, loud / n - 0.02), v: { ratio: Math.round(ratio * 100), loud: pct(loud / n) }, mask: level === "ok" ? undefined : m });
  }
  // ---- skin: people's skin (people region, skin-like in the camera's rendering) in the final.
  if (want("skin") && inp.seg?.groups.includes("person")) {
    let n2 = 0, L = 0, a = 0, b = 0;
    const m = mask();
    for (let y = 0; y < h; y += 1) for (let x = 0; x < w; x += 1) {
      const i = y * w + x;
      if (segAt(inp.seg, "person", x, y, w, h) < 0.6) continue;
      const hb = B.h[i];
      if (!(hb > 20 && hb < 90 && B.C[i] > 0.025 && B.C[i] < 0.2 && B.L[i] > 0.35 && B.L[i] < 0.92)) continue;
      n2++; L += F.L[i]; a += F.a[i]; b += F.b[i]; m[i] = 1;
    }
    if (n2 > n * 0.004) {
      L /= n2; a /= n2; b /= n2;
      const C = Math.hypot(a, b);
      let hue = (Math.atan2(b, a) * 180) / Math.PI; if (hue < 0) hue += 360;
      // Natural skin in OkLCH: hue ≈ 40–70°, chroma ≈ 0.04–0.13 (light to deep skin).
      const issue = hue < 33 ? "red" : hue > 78 ? (hue > 100 ? "green" : "yellow") : C > 0.15 ? "saturated" : C < 0.03 ? "pale" : "";
      const level: CheckLevel = !issue ? "ok" : (hue < 26 || hue > 95 || C > 0.18 || C < 0.02) ? "bad" : "warn";
      items.push({ id: "skin", level, v: { issue, hue: Math.round(hue), chroma: Math.round(C * 1000) / 1000, light: Math.round(L * 100) }, mask: level === "ok" ? undefined : m });
    }
  }
  // ---- colour cast: what was grey / white in the camera's rendering, tinted in the final.
  if (want("cast")) {
    let n2 = 0, da = 0, db = 0, ba = 0, bb = 0;
    for (let i = 0; i < n; i += 3) {
      if (B.C[i] < 0.025 && B.L[i] > 0.35 && B.L[i] < 0.95) { n2++; da += F.a[i]; db += F.b[i]; ba += B.a[i]; bb += B.b[i]; }
    }
    if (n2 > n / 3 * 0.02) {
      da /= n2; db /= n2; ba /= n2; bb /= n2;
      const sa = da - ba, sb = db - bb, shift = Math.hypot(sa, sb);
      let hue = (Math.atan2(sb, sa) * 180) / Math.PI; if (hue < 0) hue += 360;
      const tint = hue < 30 || hue >= 330 ? "magenta" : hue < 95 ? "warm" : hue < 160 ? "green" : hue < 250 ? "cyan" : "blue";
      const level: CheckLevel = shift > 0.03 ? "bad" : shift > 0.015 ? "warn" : "ok";
      items.push({ id: "cast", level, err: shift - 0.01, v: { tint, amount: Math.round(shift * 1000) } });
    }
  }
  // ---- exposure: judged for the kind of scene (a night shot should look like night,
  //      snow bright), not against a fixed middle: the scene's light level (EV from the
  //      photo's exposure settings) and the camera's own rendering decide the scene.
  if (want("exposure")) {
    const med = quantile(F.L, 0.5), cam = quantile(B.L, 0.5);
    let sky = 0;
    if (inp.seg) { const g = inp.seg.groups.indexOf("sky"), pl = inp.seg.width * inp.seg.height; if (g >= 0) { for (let i = 0; i < pl; i++) sky += inp.seg.probs[g * pl + i]; sky /= pl; } }
    const kind = sceneKind(inp.scene?.ev, cam, sky);
    const [lo, hi] = SCENE_BAND[kind];
    const dist = med < lo ? lo - med : med > hi ? med - hi : 0;
    const level: CheckLevel = dist === 0 ? "ok" : dist <= 0.06 ? "warn" : "bad";
    // Into the scene's range, a little inside it (not the edge), from whichever side.
    const err = med < lo ? lo + 0.03 - med : med > hi ? med - (hi - 0.03) : -1;
    items.push({ id: "exposure", level, err, v: {
      scene: kind, ev: inp.scene?.ev !== undefined ? Math.round(inp.scene.ev * 10) / 10 : "", median: Math.round(med * 100),
      lo: Math.round(lo * 100), hi: Math.round(hi * 100), camera: Math.round(cam * 100), dir: med < lo ? "dark" : "bright",
    } });
  }
  // ---- contrast: tonal range, and whether there are real blacks and whites.
  if (want("contrast")) {
    const lo = quantile(F.L, 0.005), hi = quantile(F.L, 0.995);
    const span = hi - lo;
    const issue = span < 0.45 ? "flat" : lo > 0.22 ? "noBlack" : hi < 0.72 ? "noWhite" : span > 0.97 && quantile(F.L, 0.05) < 0.08 ? "harsh" : "";
    const level: CheckLevel = issue === "flat" || (issue === "noBlack" && lo > 0.3) ? "bad" : issue ? "warn" : "ok";
    const cErr = issue === "flat" ? 0.52 - span : issue === "noBlack" ? lo - 0.15 : issue === "noWhite" ? 0.78 - hi : issue === "harsh" ? span - 0.95 : -1;
    items.push({ id: "contrast", level, err: cErr, v: { issue, black: Math.round(lo * 100), white: Math.round(hi * 100) } });
  }
  // ---- noise: fine grain in flat areas of the camera's rendering, final versus camera.
  //      (Clarity, texture, sharpening and lifted shadows all amplify it.)
  if (want("noise")) {
    const hp = (P: Px) => { const s = boxMean(P.L, w, h, 1); const o = new Float32Array(n); for (let i = 0; i < n; i++) o[i] = Math.abs(P.L[i] - s[i]); return o; };
    const hf = hp(F), hb = hp(B);
    const grad = boxMean(hb, w, h, 4);
    let sf = 0, sb = 0, k = 0;
    const m = mask();
    for (let i = 0; i < n; i++) {
      if (grad[i] > 0.012 || B.L[i] < 0.12) continue; // flat, not black
      sf += hf[i]; sb += hb[i]; k++;
      if (hf[i] > 0.02 && hf[i] > 2 * hb[i] + 0.004) m[i] = 1;
    }
    const ratio = k > n * 0.03 && sb > 1e-6 ? sf / sb : 1;
    const level: CheckLevel = ratio > 2.4 ? "bad" : ratio > 1.7 ? "warn" : "ok";
    items.push({ id: "noise", level, err: ratio - 1.4, v: { ratio: Math.round(ratio * 10) / 10 }, mask: level === "ok" ? undefined : m });
  }
  // ---- halos: along strong edges, the final overshoots beyond both sides of the edge
  //      as the camera had them (a bright rim on the dark side, a dark rim on the bright).
  if (want("halos")) {
    const r = Math.max(2, Math.round(Math.max(w, h) / 256));
    const gx = new Float32Array(n);
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      gx[i] = Math.hypot(B.L[i + 1] - B.L[i - 1], B.L[i + w] - B.L[i - w]);
    }
    const near = boxMean(Float32Array.from(gx, (g) => (g > 0.18 ? 1 : 0)), w, h, r * 2);
    // Compare the edge detail only (each image minus its own local mean), so a photo
    // made brighter or darker overall is not read as a halo: a halo is detail next to
    // an edge swinging much further than the camera's did, on the same side.
    const mF = boxMean(F.L, w, h, r * 2), mB = boxMean(B.L, w, h, r * 2);
    let edge = 0, halo = 0;
    const m = mask();
    for (let i = 0; i < n; i++) {
      if (near[i] < 0.02) continue;
      edge++;
      const dF = F.L[i] - mF[i], dB = B.L[i] - mB[i];
      if (Math.abs(dF) > Math.abs(dB) * 1.6 + 0.05 && Math.sign(dF) === Math.sign(dB || dF)) { halo++; m[i] = 1; }
    }
    const share = edge ? halo / edge : 0;
    const level: CheckLevel = share > 0.12 ? "bad" : share > 0.05 ? "warn" : "ok";
    items.push({ id: "halos", level, err: share - 0.03, v: { pct: pct(share) }, mask: level === "ok" ? undefined : m });
  }
  // ---- banding: in smooth gradients (sky, walls), visible steps instead of a ramp.
  if (want("banding")) {
    const B16 = 16;
    let smooth = 0, banded = 0;
    const m = mask();
    for (let by = 0; by + B16 <= h; by += B16) for (let bx = 0; bx + B16 <= w; bx += B16) {
      // Smooth in the camera's rendering: a ramp (any slope), no texture — measured by
      // the change of slope, so a steep sky gradient still counts as smooth.
      let tex = 0, gmin = 255, gmax = 0;
      const seen = new Uint8Array(256);
      for (let y = by; y < by + B16; y++) for (let x = bx; x < bx + B16; x++) {
        const i = y * w + x;
        if (x > bx + 1) tex += Math.abs(B.L[i] - 2 * B.L[i - 1] + B.L[i - 2]);
        const g = inp.final.rgba[i * 4 + 1];
        seen[g] = 1; if (g < gmin) gmin = g; if (g > gmax) gmax = g;
      }
      if (tex / (B16 * (B16 - 2)) > 0.003) continue;
      const range = gmax - gmin;
      if (range < 4) continue;
      smooth++;
      let used = 0; for (let g = gmin; g <= gmax; g++) used += seen[g];
      // A ramp over `range` levels that uses under half of them: steps.
      if (used / (range + 1) < 0.5) {
        banded++;
        for (let y = by; y < by + B16; y++) for (let x = bx; x < bx + B16; x++) m[y * w + x] = 1;
      }
    }
    const share = smooth ? banded / smooth : 0;
    const level: CheckLevel = smooth < 6 ? "ok" : share > 0.25 ? "bad" : share > 0.1 ? "warn" : "ok";
    items.push({ id: "banding", level, v: { pct: pct(share) }, mask: level === "ok" ? undefined : m });
  }
  // ---- vignette: corners versus the centre, final versus camera.
  if (want("vignette")) {
    const region = (cx: number, cy: number) => { let s = 0, k = 0; const rw = Math.round(w * 0.12), rh = Math.round(h * 0.12);
      for (let y = Math.max(0, cy - rh); y < Math.min(h, cy + rh); y += 2) for (let x = Math.max(0, cx - rw); x < Math.min(w, cx + rw); x += 2) { s += F.L[y * w + x] - B.L[y * w + x]; k++; }
      return k ? s / k : 0; };
    const corners = (region(0, 0) + region(w - 1, 0) + region(0, h - 1) + region(w - 1, h - 1)) / 4;
    const centre = region(Math.round(w / 2), Math.round(h / 2));
    const drop = centre - corners; // how much darker the edit made the corners than the centre
    const level: CheckLevel = drop > 0.18 ? "bad" : drop > 0.1 ? "warn" : "ok";
    items.push({ id: "vignette", level, err: drop - 0.05, v: { drop: Math.round(drop * 100) } });
  }
  return items;
}
