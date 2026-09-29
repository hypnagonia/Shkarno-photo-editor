/**
 * Camera colour science, Stage B: tone curve and RGB curves as 1D LUTs.
 *
 * toneCurveLUT: scene log2 luminance (−14 … +6 EV) → display-linear luminance.
 *   A Naka–Rushton (log-logistic) curve anchored so that scene middle grey
 *   0.18 renders at MID_OUT, with a contrast exponent, a highlight shoulder
 *   whose softness is `rolloff`, and a black floor. It is applied to luminance
 *   as a ratio, so it never shifts hue.
 *
 * curveLUT: the user/automatic point curves (L, R, G, B) on display-encoded
 *   values, interpolated with a monotone cubic so they cannot overshoot.
 */
import type { CurvePoint, Params } from "../decision/params.ts";

export const TONE_LUT_SIZE = 4096;
export const TONE_EV_MIN = -14;
export const TONE_EV_RANGE = 20;
export const CURVE_LUT_SIZE = 1024;

/**
 * Rendering intent: scene middle grey (0.18) is displayed at MID_OUT
 * (display-linear; ≈ 58% sRGB). A fixed property of the rendering — the same
 * for every photograph — so the camera's exposure decides brightness, not a
 * per-image target. Calibrated against the camera's own renderings of the same
 * files (daylight, dim interior, night): the earlier 0.23 with a steeper toe
 * came out ~0.6 EV darker than the phone across the frame.
 */
export const MID_OUT = 0.29;

/** The tone settings that change nothing (the rendering intent itself). */
export const NEUTRAL_TONE: Params["tone"] = { highlights: 0, shadows: 0, whites: 0, blacks: 0, contrast: 0, rolloff: 0.5 };

/**
 * The tone curve: scene luminance → display luminance.
 *
 * A display-referred source (JPEG / HEIC, `tone.displayReferred`) is already a finished
 * rendering: rendering it again (mid grey 0.18 → 0.29, a new toe and shoulder) made an
 * untouched iPhone photo 20 % brighter and flatter. For it the curve is the identity,
 * and the settings apply as the change they make to the neutral scene curve: the value
 * is read back to the scene level the neutral curve would show it at, and rendered with
 * the settings — so neutral settings return it unchanged. Above white (exposure raised
 * it) a soft shoulder rolls it into 1 instead of clipping.
 */
export function toneCurve(tone: Params["tone"]): (Y: number) => number {
  if (!tone.displayReferred) return sceneCurve(tone);
  const edited = sceneCurve(tone), neutral = sceneCurve(NEUTRAL_TONE);
  const inv = inverseOf(neutral);
  const K = 0.92;
  return (Y: number) => {
    const y = Y <= K ? Math.max(Y, 0) : 1 - (1 - K) * Math.exp(-(Y - K) / (1 - K));
    return Math.min(1, edited(inv(y)));
  };
}

/** The inverse of a monotone curve on 0 … 1 (bisection in log2 of its input, cached on a grid). */
function inverseOf(f: (Y: number) => number): (y: number) => number {
  const N = 1024, lo = -20, hi = 8, tab = new Float64Array(N + 1);
  for (let i = 0; i <= N; i++) {
    const target = i / N;
    let a = lo, b = hi;
    for (let k = 0; k < 50; k++) { const m = (a + b) / 2; if (f(2 ** m) < target) a = m; else b = m; }
    tab[i] = 2 ** ((a + b) / 2);
  }
  return (y: number) => {
    const x = Math.min(1, Math.max(0, y)) * N, i = Math.min(N - 1, Math.floor(x)), t = x - i;
    // Interpolated in log (the inverse is steep near black).
    return Math.exp(Math.log(Math.max(tab[i], 1e-12)) * (1 - t) + Math.log(Math.max(tab[i + 1], 1e-12)) * t);
  };
}

function sceneCurve(tone: Params["tone"]) {
  const c = 1.02 + 0.35 * tone.contrast;
  // Peak slightly above 1: rolloff = 1 → asymptotic shoulder (softest), 0 → reaches white early.
  const Yw = 1.0 + 0.3 * (1 - tone.rolloff) * (1 + 0.5 * tone.whites);
  const mid = 0.18;
  const kc = Math.pow(mid, c) * (Yw / MID_OUT - 1);
  const b = tone.blacks;
  return (Y: number) => {
    const yc = Math.pow(Math.max(Y, 0), c);
    let d = Math.min(1, (Yw * yc) / (yc + kc));
    // Soft toe: deep shadows are compressed toward black, never clipped to it,
    // so shadow separation survives (d²/(d+t) ≈ d above the toe).
    const t = 0.0006;
    d = (d * d) / (d + t) * (1 + t);
    if (b < 0) {
      // Deeper blacks as a stronger toe, never a clip: dark levels are pushed
      // toward black but stay distinct (and white stays white).
      const f = -b * 0.03;
      d = (d * d) / (d + f) * (1 + f);
    } else if (b > 0) {
      const f = b * 0.02;
      d = f + d * (1 - f);
    }
    return Math.min(1, d);
  };
}

/**
 * HDR rendition as a gain over the SDR tone curve (for HDR screens and the
 * gain-map JPEG). Below the knee — the scene level the SDR curve shows at
 * display-linear 0.5, well above middle grey — the gain is exactly 1, so
 * shadows and mid-tones are the SDR rendering. Above it, a second
 * Naka–Rushton curve with the same contrast exponent, anchored at the knee and
 * peaking at the headroom H = 2^stops, takes over across one stop (C¹ at the
 * knee): skies a stop or two above get ≈ 1.3–1.7×, speculars and the sun
 * approach H. HDR = SDR × gain, monotone because both factors are.
 */
export const HDR_KNEE = 0.5;

/** Scene luminance at which the SDR curve reaches HDR_KNEE (bisection in log2). */
export function hdrKnee(tone: Params["tone"]): number {
  const f = toneCurve(tone);
  let lo = -20, hi = 10;
  for (let k = 0; k < 60; k++) { const m = (lo + hi) / 2; if (f(Math.pow(2, m)) < HDR_KNEE) lo = m; else hi = m; }
  return Math.pow(2, (lo + hi) / 2);
}

/** Linear gain HDR / SDR at scene luminance Y (1 below the knee, ≤ 2^stops). */
export function hdrGain(tone: Params["tone"], stops: number): (Y: number) => number {
  if (!(stops > 0)) return () => 1;
  const sdr = toneCurve(tone);
  const H = Math.pow(2, stops);
  const c = 1.02 + 0.35 * tone.contrast;
  const Yk = hdrKnee(tone);
  const kch = Math.pow(Yk, c) * (H / HDR_KNEE - 1);
  return (Y: number) => {
    if (Y <= Yk) return 1;
    const yc = Math.pow(Y, c);
    const nr = (H * yc) / (yc + kch);
    const d = Math.max(0, Math.log2(nr / Math.max(sdr(Y), 1e-6)));
    const t = Math.min(1, Math.max(0, Math.log2(Y / Yk)));
    const w = t * t * (3 - 2 * t);
    return Math.min(H, Math.max(1, Math.pow(2, w * d)));
  };
}

/** Tone table: r = SDR display luminance, g = HDR gain (1 when `hdrStops` is 0), b = SDR, a = 1. */
export function toneCurveLUT(tone: Params["tone"], hdrStops = 0): Float32Array {
  const f = toneCurve(tone);
  const g = hdrGain(tone, hdrStops);
  const out = new Float32Array(TONE_LUT_SIZE * 4);
  for (let i = 0; i < TONE_LUT_SIZE; i++) {
    const ev = TONE_EV_MIN + (i / (TONE_LUT_SIZE - 1)) * TONE_EV_RANGE;
    const Y = Math.pow(2, ev);
    const v = f(Y);
    out[i * 4] = v;
    out[i * 4 + 1] = g(Y); out[i * 4 + 2] = v; out[i * 4 + 3] = 1;
  }
  return out;
}

/** Monotone cubic (Fritsch–Carlson) through sorted points. */
export function monotoneCurve(points: CurvePoint[]): (x: number) => number {
  const p = [...points].sort((a, b) => a.x - b.x);
  const n = p.length;
  if (n < 2) return (x) => x;
  const d: number[] = [], m: number[] = new Array(n).fill(0);
  for (let i = 0; i < n - 1; i++) d.push((p[i + 1].y - p[i].y) / Math.max(1e-9, p[i + 1].x - p[i].x));
  m[0] = d[0];
  m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) { m[i] = 0; m[i + 1] = 0; continue; }
    const a = m[i] / d[i], b = m[i + 1] / d[i];
    const s = a * a + b * b;
    if (s > 9) { const t = 3 / Math.sqrt(s); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]; }
  }
  return (x: number) => {
    if (x <= p[0].x) return p[0].y;
    if (x >= p[n - 1].x) return p[n - 1].y;
    let i = 0;
    while (i < n - 2 && x > p[i + 1].x) i++;
    const h = p[i + 1].x - p[i].x, t = (x - p[i].x) / h;
    const t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * p[i].y + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * p[i + 1].y + (t3 - t2) * h * m[i + 1];
  };
}

export function isFlat(points: CurvePoint[]): boolean {
  return points.every((q) => Math.abs(q.x - q.y) < 1e-4);
}

export function curveLUT(curves: Params["curves"]): Float32Array {
  const fs = [curves.l, curves.r, curves.g, curves.b].map(monotoneCurve);
  const out = new Float32Array(CURVE_LUT_SIZE * 4);
  for (let i = 0; i < CURVE_LUT_SIZE; i++) {
    const x = i / (CURVE_LUT_SIZE - 1);
    for (let c = 0; c < 4; c++) out[i * 4 + c] = Math.min(1, Math.max(0, fs[c](x)));
  }
  return out;
}

/**
 * Tone-range sliders over a point curve: black level, five tone ranges
 * (shadows, darks, midtones, lights, highlights at x = 0.1 … 0.9) and white
 * level. They are a *view* of the curve, not extra parameters: reading
 * samples the curve at those places, writing rebuilds the curve through them,
 * so a curve set any other way (an older edit, the Ask-AI answer) still shows
 * up on the sliders.
 */
export const TONE_BANDS = [0.1, 0.3, 0.5, 0.7, 0.9];
/** Output shift of a tone-range slider at ±1. */
export const BAND_RANGE = 0.12;
/** Black lift / white drop at slider 1 / −1. */
export const END_RANGE = 0.15;

export interface CurveBands {
  black: number; bands: number[]; white: number;
  /**
   * A toe anchored at the photo's own shadow floor: input level x renders at y
   * (≤ x). Blended back into the curve by a second point at ≈ 2.2·x; band points
   * below that are left out (the toe shapes that range).
   */
  toe?: [number, number];
}

const clampN = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const round4 = (v: number) => Math.round(v * 10000) / 10000;

export function bandsFromCurve(points: CurvePoint[]): CurveBands {
  const f = monotoneCurve(points);
  return {
    black: round4(clampN(f(0) / END_RANGE, 0, 1)),
    bands: TONE_BANDS.map((x) => round4(clampN((f(x) - x) / BAND_RANGE, -1, 1))),
    white: round4(clampN((f(1) - 1) / END_RANGE, -1, 0)),
  };
}

export function curveFromBands(b: CurveBands): CurvePoint[] {
  const pts: CurvePoint[] = [{ x: 0, y: clampN(b.black, 0, 1) * END_RANGE }];
  let from = 0;
  if (b.toe && b.toe[1] < b.toe[0] - 1e-4) {
    const [x1, y1] = b.toe;
    // Rejoins the curve soon above the floor, so midtones keep their level.
    const x2 = Math.min(0.4, x1 * 1.8), y2 = x2 - (x1 - y1) * 0.2;
    pts.push({ x: x1, y: y1 }, { x: x2, y: y2 });
    from = x2 + 0.05;
  }
  TONE_BANDS.forEach((x, i) => { if (x > from) pts.push({ x, y: clampN(x + clampN(b.bands[i] ?? 0, -1, 1) * BAND_RANGE, 0, 1) }); });
  pts.push({ x: 1, y: 1 + clampN(b.white, -1, 0) * END_RANGE });
  // Never inverted: each point at or above the one before it.
  for (let i = 1; i < pts.length; i++) pts[i].y = Math.max(pts[i].y, pts[i - 1].y);
  const out = pts.map((q) => ({ x: q.x, y: round4(q.y) }));
  return isFlat(out) ? [{ x: 0, y: 0 }, { x: 1, y: 1 }] : out;
}
