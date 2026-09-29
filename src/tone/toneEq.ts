/**
 * Tone equalizer: exposure by brightness zone, the zone of a pixel read from an
 * edge-preserving smoothed luminance mask (not from the pixel itself), so shadows
 * and highlights move without flattening local contrast or haloing edges. The idea
 * of darktable's module; this implementation is our own.
 *
 *   mask      log2 of scene luminance (linear, after white balance and exposure),
 *             smoothed by the guided filter the refinement already computes (the
 *             same base local tone uses), at the chosen detail preservation
 *   mask comp (mask + exposure + 4) · contrast − 4: slides / stretches the mask so
 *             the photo's tones use all nine zones
 *   curve     nine zones at −8 … 0 EV, a gain of ±2 EV each; a smooth monotone cubic
 *             through them (optionally smoothed), flat beyond the outer zones
 *   apply     rgb × 2^curve(mask) in scene-linear light, before the tone curve
 */

export type ToneEqDetail = "none" | "fine" | "balanced" | "smooth";
export const TONE_EQ_DETAIL: ToneEqDetail[] = ["none", "fine", "balanced", "smooth"];

export interface ToneEq {
  enabled: boolean;
  /** Gain (EV) of the zones at −8, −7 … 0 EV. */
  gains: number[];
  /** 0 = the curve passes through every node; 1 = strongly smoothed between them. */
  smoothing: number;
  /** Which smoothed luminance decides a pixel's zone. */
  detail: ToneEqDetail;
  /** Mask compensation: shift (EV) and stretch around −4 EV. */
  maskExposure: number;
  maskContrast: number;
}

export const ZONES = 9;
/** Zone centres in EV: −8 … 0. */
export const ZONE_EV = Array.from({ length: ZONES }, (_, i) => i - 8);
export const MAX_GAIN = 2;
/** The GPU table: LUT_N samples of the curve over LUT_LO … LUT_HI EV (mask values). */
export const LUT_N = 64, LUT_LO = -10, LUT_HI = 2;
export const MASK_PIVOT = -4;

export const neutralToneEq = (): ToneEq => ({ enabled: true, gains: new Array(ZONES).fill(0), smoothing: 0, detail: "balanced", maskExposure: 0, maskContrast: 1 });

/** Does it change anything? (Off, or every zone at 0.) */
export const toneEqActive = (eq: ToneEq | undefined): eq is ToneEq => !!eq && eq.enabled && eq.gains.some((g) => Math.abs(g) > 1e-4);

/**
 * The curve as a function of mask EV: a monotone cubic (Fritsch–Carlson) through the
 * nodes — smooth, and never beyond its neighbouring nodes (raising one zone does not
 * darken the next, which a Gaussian basis does). `smoothing` first averages each node
 * with its neighbours. Outside −8 … 0 EV it holds the outer zones' value.
 */
export function toneEqCurve(eq: ToneEq): (ev: number) => number {
  const n = ZONES;
  let y = Array.from({ length: n }, (_, i) => Math.max(-MAX_GAIN, Math.min(MAX_GAIN, eq.gains[i] ?? 0)));
  const k = Math.min(1, Math.max(0, eq.smoothing)) * 0.5;
  for (let pass = 0; pass < 2 && k > 0; pass++) y = y.map((v, i) => (1 - k) * v + (k / 2) * ((y[i - 1] ?? v) + (y[i + 1] ?? v)));
  // Slopes (unit spacing): secants, then tangents limited so each segment stays monotone.
  const d = y.slice(1).map((v, i) => v - y[i]);
  const m = y.map((_, i) => (i === 0 ? d[0] : i === n - 1 ? d[n - 2] : d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2));
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) { m[i] = 0; m[i + 1] = 0; continue; }
    const a = m[i] / d[i], b = m[i + 1] / d[i], h = a * a + b * b;
    if (h > 9) { const t = 3 / Math.sqrt(h); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]; }
  }
  return (ev: number) => {
    const x = Math.min(0, Math.max(-8, ev)) + 8; // 0 … 8
    const i = Math.min(n - 2, Math.floor(x)), t = x - i, t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * m[i] + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * m[i + 1];
  };
}

/** The curve sampled for the GPU: LUT_N values over LUT_LO … LUT_HI (EV of the compensated mask). */
export function toneEqLut(eq: ToneEq): Float32Array {
  const f = toneEqCurve(eq), out = new Float32Array(LUT_N);
  for (let i = 0; i < LUT_N; i++) out[i] = f(LUT_LO + ((LUT_HI - LUT_LO) * i) / (LUT_N - 1));
  return out;
}

/** The compensated mask value of a raw mask EV. */
export const compensate = (eq: Pick<ToneEq, "maskExposure" | "maskContrast">, ev: number) => (ev + eq.maskExposure - MASK_PIVOT) * eq.maskContrast + MASK_PIVOT;

/** A histogram of mask EV: counts per bin over lo … hi. */
export interface MaskHist { lo: number; hi: number; bins: number[] }

/** The value below which `q` of the histogram's mass lies. */
export function histQuantile(h: MaskHist, q: number): number {
  const total = h.bins.reduce((a, b) => a + b, 0);
  if (!total) return (h.lo + h.hi) / 2;
  const step = (h.hi - h.lo) / h.bins.length;
  let acc = 0;
  for (let i = 0; i < h.bins.length; i++) {
    const next = acc + h.bins[i];
    if (next >= q * total) return h.lo + step * (i + (q * total - acc) / Math.max(h.bins[i], 1e-9));
    acc = next;
  }
  return h.hi;
}

/**
 * Mask compensation that spreads the photo's tones (5th–95th percentile of the mask,
 * at the current exposure) over −7 … −1 EV, so every zone has something in it.
 * `evShift`: the exposure (EV) the histogram was not measured at.
 */
export function autoFitMask(h: MaskHist, evShift = 0): { maskExposure: number; maskContrast: number } {
  const p5 = histQuantile(h, 0.05) + evShift, p95 = histQuantile(h, 0.95) + evShift;
  const k = Math.min(4, Math.max(0.5, 6 / Math.max(1e-3, p95 - p5)));
  // (p5 + e − pivot)·k + pivot = −7  →  e = −3/k + pivot − p5
  const e = -3 / k + MASK_PIVOT - p5;
  return { maskExposure: Math.round(Math.min(6, Math.max(-6, e)) * 100) / 100, maskContrast: Math.round(k * 100) / 100 };
}

/** Starting points (EV per zone, −8 … 0). */
export type ToneEqPresetId = "compressSoft" | "compressMedium" | "compressStrong" | "liftShadows" | "tameHighlights" | "contrast";
export const TONE_EQ_PRESETS: Array<{ id: ToneEqPresetId; gains: number[] }> = [
  { id: "compressSoft", gains: [0.9, 0.75, 0.55, 0.35, 0.15, 0, -0.15, -0.35, -0.55] },
  { id: "compressMedium", gains: [1.6, 1.35, 1.05, 0.7, 0.3, 0, -0.3, -0.65, -1] },
  { id: "compressStrong", gains: [2, 1.8, 1.45, 1, 0.45, 0, -0.45, -0.95, -1.4] },
  { id: "liftShadows", gains: [1.5, 1.35, 1.05, 0.65, 0.25, 0, 0, 0, 0] },
  { id: "tameHighlights", gains: [0, 0, 0, 0, 0, -0.2, -0.5, -0.9, -1.3] },
  { id: "contrast", gains: [-0.7, -0.6, -0.45, -0.25, 0, 0.15, 0.3, 0.45, 0.5] },
];
