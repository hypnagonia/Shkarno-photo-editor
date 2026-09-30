import { srgbEotf as eotf, srgbOetf as oetf } from "../color/transfer.ts";
import { linSrgbToOklab, oklabToLinSrgb } from "../color/oklab.ts";
/**
 * Gradients for the Gradient Map and Gradient Fill layers: colour stops (sRGB
 * hex, location 0…1, opacity), preset palettes, harmonies generated in OkLCH,
 * and the 1024-sample table the GPU reads (display-encoded Display P3, the
 * tone pass's working encoding).
 *
 * Interpolation is in encoded sRGB, as Photoshop's "classic" gradients, or in
 * OkLab ("smooth": even perceived steps, no muddy or over-bright middles).
 */

export interface GradientStop { pos: number; color: string; /** 0…1 */ alpha: number }
export interface Gradient {
  stops: GradientStop[];
  /** Interpolation between stops: encoded sRGB (classic, the default) or OkLab (smooth). */
  space?: "srgb" | "oklab";
}

export type GradientGroup = "palettes" | "cinematic" | "warm" | "film" | "nature" | "duotone" | "pastel" | "bold";
export const GRADIENT_GROUPS: GradientGroup[] = ["palettes", "cinematic", "warm", "film", "nature", "duotone", "pastel", "bold"];
export interface GradientPreset { id: string; group: GradientGroup; colors: string[] }

// ---------------------------------------------------------------- OkLCH helpers

/** Encoded sRGB hex → OkLab. */
export function hexToOklab(hex: string): [number, number, number] {
  const [r, g, b] = hexToRgb(hex);
  return linSrgbToOklab([eotf(r), eotf(g), eotf(b)]);
}
const inGamut = (lin: readonly number[]) => lin.every((v) => v >= -1e-4 && v <= 1 + 1e-4);
/** OkLab → encoded sRGB hex. Out-of-gamut colours keep their lightness and hue and lose chroma until they fit (no clipping). */
export function oklabToHex(L: number, a: number, b: number): string {
  L = Math.min(1, Math.max(0, L));
  let lin = oklabToLinSrgb([L, a, b]);
  if (!inGamut(lin)) {
    let lo = 0, hi = 1;
    for (let i = 0; i < 24; i++) {
      const k = (lo + hi) / 2;
      if (inGamut(oklabToLinSrgb([L, a * k, b * k]))) lo = k; else hi = k;
    }
    lin = oklabToLinSrgb([L, a * lo, b * lo]);
  }
  return rgbToHex(oetf(Math.min(1, Math.max(0, lin[0]))), oetf(Math.min(1, Math.max(0, lin[1]))), oetf(Math.min(1, Math.max(0, lin[2]))));
}
/** OkLCH (hue in degrees) → encoded sRGB hex, gamut-mapped by chroma. */
export function oklchToHex(L: number, C: number, h: number): string {
  const r = (h * Math.PI) / 180;
  return oklabToHex(L, C * Math.cos(r), C * Math.sin(r));
}
/** OkLCH of a hex colour (hue in degrees, 0…360). */
export function hexToOklch(hex: string): [number, number, number] {
  const [L, a, b] = hexToOklab(hex);
  let h = (Math.atan2(b, a) * 180) / Math.PI;
  if (h < 0) h += 360;
  return [L, Math.hypot(a, b), h];
}
/** Authored palettes are written in OkLCH: lightness rises by construction and every colour is in gamut. */
const lch = (...cs: Array<[number, number, number]>) => cs.map(([L, C, h]) => oklchToHex(L, C, h));

/**
 * Preset palettes, dark → light (so a Gradient Map puts the first colour in the
 * shadows and the last in the highlights). "tealGold" is schemecolor.com's
 * "Teal and Gold" palette (Skobeloff, Teal, Dark Goldenrod, Chinese Gold, Goldenrod).
 */
export const GRADIENT_PRESETS: GradientPreset[] = [
  // Palettes: 3–5 distinct colours that belong together, each clearly coloured (not a
  // dark → light ramp of one or two hues), still ordered dark → light for gradient maps.
  { id: "harbor", group: "palettes", colors: lch([0.32, 0.09, 255], [0.48, 0.1, 210], [0.63, 0.11, 170], [0.78, 0.11, 110], [0.9, 0.08, 85]) },
  { id: "tropical", group: "palettes", colors: lch([0.36, 0.11, 260], [0.53, 0.12, 190], [0.7, 0.16, 145], [0.84, 0.16, 105], [0.93, 0.11, 90]) },
  { id: "sunsetBlvd", group: "palettes", colors: lch([0.34, 0.13, 300], [0.5, 0.17, 350], [0.65, 0.17, 30], [0.79, 0.14, 62], [0.91, 0.1, 92]) },
  { id: "retroPop", group: "palettes", colors: lch([0.38, 0.15, 280], [0.58, 0.2, 350], [0.78, 0.16, 75], [0.92, 0.11, 110]) },
  { id: "berryCream", group: "palettes", colors: lch([0.32, 0.12, 335], [0.52, 0.16, 5], [0.74, 0.1, 40], [0.92, 0.05, 80]) },
  { id: "terracotta", group: "palettes", colors: lch([0.35, 0.08, 35], [0.53, 0.13, 45], [0.7, 0.11, 70], [0.87, 0.06, 90]) },
  { id: "citrusGrove", group: "palettes", colors: lch([0.4, 0.1, 155], [0.62, 0.15, 130], [0.8, 0.16, 105], [0.92, 0.12, 95]) },
  { id: "lagoon", group: "palettes", colors: lch([0.35, 0.08, 230], [0.6, 0.11, 195], [0.85, 0.1, 170]) },
  { id: "mossStone", group: "palettes", colors: lch([0.33, 0.05, 145], [0.5, 0.07, 120], [0.68, 0.06, 90], [0.85, 0.04, 70]) },
  { id: "arctic", group: "palettes", colors: lch([0.38, 0.09, 265], [0.6, 0.1, 230], [0.82, 0.08, 200]) },
  { id: "candyShop", group: "palettes", colors: lch([0.45, 0.16, 320], [0.62, 0.15, 260], [0.76, 0.12, 190], [0.9, 0.12, 120]) },
  { id: "autumnWalk", group: "palettes", colors: lch([0.3, 0.07, 20], [0.48, 0.13, 35], [0.64, 0.14, 60], [0.78, 0.13, 95], [0.9, 0.07, 110]) },
  // Cinematic: cool shadows, warm or neutral highlights.
  { id: "tealGold", group: "cinematic", colors: ["#006C77", "#008182", "#B7850E", "#CC9802", "#DBA620"] },
  { id: "tealOrange", group: "cinematic", colors: ["#0E2A33", "#1D6A73", "#D9894A", "#FFD9A8"] },
  { id: "navyGold", group: "cinematic", colors: ["#0D1B3E", "#2B4C7E", "#C9A227", "#F6E7A1"] },
  { id: "noirBlue", group: "cinematic", colors: lch([0.16, 0.04, 255], [0.34, 0.07, 240], [0.58, 0.06, 220], [0.82, 0.03, 90], [0.95, 0.02, 85]) },
  { id: "emberSteel", group: "cinematic", colors: lch([0.18, 0.03, 250], [0.38, 0.05, 235], [0.6, 0.1, 55], [0.8, 0.1, 70], [0.95, 0.04, 85]) },
  { id: "bleachBypass", group: "cinematic", colors: lch([0.14, 0.01, 250], [0.4, 0.02, 230], [0.66, 0.025, 80], [0.9, 0.015, 90]) },
  { id: "moonlight", group: "cinematic", colors: lch([0.12, 0.05, 270], [0.32, 0.08, 260], [0.6, 0.06, 230], [0.88, 0.03, 200]) },
  // Warm light.
  { id: "goldenHour", group: "warm", colors: ["#2A170C", "#8A4A1D", "#E39A3F", "#FFE6AE"] },
  { id: "sunset", group: "warm", colors: ["#2B1055", "#7A2C6E", "#E2615A", "#FFC878"] },
  { id: "desertDusk", group: "warm", colors: lch([0.2, 0.05, 20], [0.42, 0.1, 35], [0.64, 0.12, 60], [0.85, 0.09, 80], [0.96, 0.04, 90]) },
  { id: "amberNight", group: "warm", colors: lch([0.14, 0.04, 300], [0.34, 0.1, 20], [0.6, 0.14, 55], [0.86, 0.1, 85]) },
  // Film and print processes.
  { id: "vintage", group: "film", colors: ["#1D1E2C", "#5A5470", "#C7A46A", "#F3E8D1"] },
  { id: "sepia", group: "film", colors: ["#1E120A", "#6B4A2E", "#C29E74", "#F5E8D3"] },
  { id: "cyanotype", group: "film", colors: ["#0B1E3F", "#2C5A8C", "#9CC0DB", "#F1F6FA"] },
  { id: "warmFilm", group: "film", colors: lch([0.18, 0.03, 40], [0.42, 0.06, 55], [0.68, 0.07, 75], [0.9, 0.05, 90]) },
  { id: "fadedPrint", group: "film", colors: lch([0.3, 0.03, 260], [0.5, 0.04, 200], [0.72, 0.05, 90], [0.92, 0.03, 80]) },
  { id: "platinum", group: "film", colors: lch([0.15, 0.01, 60], [0.45, 0.015, 70], [0.75, 0.02, 80], [0.95, 0.015, 85]) },
  { id: "selenium", group: "film", colors: lch([0.14, 0.03, 330], [0.4, 0.035, 340], [0.7, 0.02, 30], [0.95, 0.01, 60]) },
  // Nature.
  { id: "forest", group: "nature", colors: ["#0F1A14", "#2E4A3A", "#99A276", "#ECE5CE"] },
  { id: "nordic", group: "nature", colors: ["#18212C", "#475B6E", "#A5B2BC", "#F0EEE9"] },
  { id: "ocean", group: "nature", colors: lch([0.16, 0.05, 250], [0.36, 0.09, 230], [0.58, 0.1, 200], [0.8, 0.08, 180], [0.95, 0.03, 160]) },
  { id: "meadow", group: "nature", colors: lch([0.2, 0.05, 150], [0.42, 0.09, 140], [0.65, 0.12, 125], [0.85, 0.12, 105], [0.96, 0.06, 95]) },
  { id: "autumn", group: "nature", colors: lch([0.18, 0.05, 30], [0.38, 0.11, 35], [0.6, 0.14, 55], [0.8, 0.13, 80], [0.94, 0.06, 90]) },
  { id: "lavender", group: "nature", colors: lch([0.2, 0.06, 300], [0.42, 0.09, 295], [0.66, 0.08, 300], [0.85, 0.05, 320], [0.96, 0.02, 90]) },
  { id: "glacier", group: "nature", colors: lch([0.2, 0.04, 240], [0.45, 0.06, 225], [0.7, 0.06, 210], [0.88, 0.04, 200], [0.97, 0.01, 200]) },
  // Duotones: two inks.
  { id: "bw", group: "duotone", colors: ["#000000", "#FFFFFF"] },
  { id: "duoBlueOrange", group: "duotone", colors: lch([0.24, 0.1, 265], [0.58, 0.13, 20], [0.88, 0.1, 75]) },
  { id: "duoMagentaYellow", group: "duotone", colors: lch([0.26, 0.13, 330], [0.62, 0.18, 30], [0.92, 0.13, 100]) },
  { id: "duoGreenPink", group: "duotone", colors: lch([0.28, 0.08, 165], [0.6, 0.09, 120], [0.88, 0.07, 10]) },
  { id: "duoNavyCream", group: "duotone", colors: lch([0.22, 0.07, 260], [0.6, 0.08, 220], [0.93, 0.04, 90]) },
  { id: "duoPlumPeach", group: "duotone", colors: lch([0.26, 0.09, 320], [0.6, 0.12, 10], [0.88, 0.08, 55]) },
  // Pastel: lifted shadows, soft colour.
  { id: "pastel", group: "pastel", colors: ["#4E5D80", "#95AFC6", "#F1C4C0", "#FFF3DE"] },
  { id: "roseGold", group: "pastel", colors: ["#2E1A1F", "#8C585C", "#D8A49F", "#FBE8E3"] },
  { id: "cottonCandy", group: "pastel", colors: lch([0.45, 0.06, 280], [0.62, 0.08, 320], [0.8, 0.08, 10], [0.93, 0.05, 60]) },
  { id: "sherbet", group: "pastel", colors: lch([0.5, 0.08, 20], [0.68, 0.1, 50], [0.85, 0.09, 90], [0.96, 0.04, 110]) },
  { id: "mist", group: "pastel", colors: lch([0.42, 0.03, 240], [0.62, 0.04, 220], [0.82, 0.03, 150], [0.95, 0.015, 100]) },
  // Bold.
  { id: "cyber", group: "bold", colors: ["#12002B", "#5B1A8C", "#1FA6A6", "#B8FFF2"] },
  { id: "neon", group: "bold", colors: lch([0.15, 0.1, 300], [0.4, 0.2, 320], [0.66, 0.18, 350], [0.86, 0.14, 80]) },
  { id: "electric", group: "bold", colors: lch([0.16, 0.08, 270], [0.4, 0.18, 265], [0.65, 0.16, 210], [0.88, 0.14, 160]) },
  { id: "infrared", group: "bold", colors: lch([0.14, 0.06, 300], [0.38, 0.2, 350], [0.62, 0.2, 30], [0.84, 0.16, 70], [0.96, 0.08, 100]) },
  { id: "spectrum", group: "bold", colors: lch([0.25, 0.1, 300], [0.42, 0.08, 250], [0.6, 0.1, 180], [0.78, 0.16, 140], [0.93, 0.18, 105]) },
];

/** Evenly spaced, opaque stops from a list of colours (`space` given: that interpolation). */
export function gradientFrom(colors: string[], space?: Gradient["space"]): Gradient {
  const n = Math.max(1, colors.length - 1);
  const g: Gradient = { stops: colors.map((c, i) => ({ pos: colors.length === 1 ? 0 : i / n, color: c, alpha: 1 })) };
  if (space) g.space = space;
  return g;
}
/**
 * A palette laid over a gradient that fades (any stop not fully opaque): its colours, the
 * old gradient's opacity. Stops at the old positions (so the fade keeps its exact shape:
 * a flare's ghost stays a soft disc, a graduated filter still fades out) and at the
 * palette's own (so none of its colours is lost). A gradient that never fades: the palette as it is.
 */
export function withAlphaOf(next: Gradient, prev: Gradient): Gradient {
  if (prev.stops.every((s) => s.alpha >= 0.999)) return next;
  const at = new Map<number, number>();
  for (const s of [...prev.stops, ...next.stops]) at.set(Math.round(s.pos * 1e4) / 1e4, 0);
  const stops = [...at.keys()].sort((a, b) => a - b).map((pos) => {
    const c = gradientAt(next, pos);
    return { pos, color: rgbToHex(c[0], c[1], c[2]), alpha: Math.round(gradientAt(prev, pos)[3] * 1000) / 1000 };
  });
  return { ...next, stops };
}
/** Every stop in one colour, each keeping its opacity (a flare's ghost, glow or streak recoloured). */
export function recolor(g: Gradient, color: string): Gradient {
  return { ...g, stops: g.stops.map((s) => ({ ...s, color })) };
}
export const presetGradient = (id: string, space?: Gradient["space"]): Gradient => gradientFrom(GRADIENT_PRESETS.find((p) => p.id === id)?.colors ?? ["#000000", "#FFFFFF"], space);

export function hexToRgb(hex: string): [number, number, number] {
  let h = hex.trim().replace(/^#/, "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  const v = parseInt(h.slice(0, 6), 16);
  if (!Number.isFinite(v) || h.length < 6) return [0, 0, 0];
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}
export function rgbToHex(r: number, g: number, b: number): string {
  const c = (x: number) => Math.round(Math.min(1, Math.max(0, x)) * 255).toString(16).padStart(2, "0");
  return `#${c(r)}${c(g)}${c(b)}`.toUpperCase();
}

const sorted = (g: Gradient) => [...g.stops].sort((a, b) => a.pos - b.pos);

/** Colour and opacity at `x` (0…1), encoded sRGB. */
export function gradientAt(g: Gradient, x: number): [number, number, number, number] {
  const s = sorted(g);
  if (!s.length) return [0, 0, 0, 1];
  if (x <= s[0].pos) return [...hexToRgb(s[0].color), s[0].alpha];
  const last = s[s.length - 1];
  if (x >= last.pos) return [...hexToRgb(last.color), last.alpha];
  let i = 0;
  while (i < s.length - 2 && x > s[i + 1].pos) i++;
  const a = s[i], b = s[i + 1];
  const t = b.pos > a.pos ? (x - a.pos) / (b.pos - a.pos) : 0;
  const alpha = a.alpha + (b.alpha - a.alpha) * t;
  if (g.space === "oklab") {
    const la = hexToOklab(a.color), lb = hexToOklab(b.color);
    const lin = oklabToLinSrgb([la[0] + (lb[0] - la[0]) * t, la[1] + (lb[1] - la[1]) * t, la[2] + (lb[2] - la[2]) * t]);
    const enc = (v: number) => oetf(Math.min(1, Math.max(0, v)));
    return [enc(lin[0]), enc(lin[1]), enc(lin[2]), alpha];
  }
  const ca = hexToRgb(a.color), cb = hexToRgb(b.color);
  return [ca[0] + (cb[0] - ca[0]) * t, ca[1] + (cb[1] - ca[1]) * t, ca[2] + (cb[2] - ca[2]) * t, alpha];
}


/** sRGB (encoded) → Display P3 (encoded): the same colour in the tone pass's encoding. */
export function srgbToP3Encoded(r: number, g: number, b: number): [number, number, number] {
  const R = eotf(r), G = eotf(g), B = eotf(b);
  return [oetf(0.8224621 * R + 0.177538 * G), oetf(0.0331942 * R + 0.9668058 * G), oetf(0.0170827 * R + 0.0723974 * G + 0.9105199 * B)];
}

/** `n` samples (rgba) over 0…1 for the GPU. */
export function gradientTable(g: Gradient, n = 1024): Float32Array {
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const [r, gg, b, a] = gradientAt(g, i / (n - 1));
    const p = srgbToP3Encoded(r, gg, b);
    out.set([p[0], p[1], p[2], a], i * 4);
  }
  return out;
}

/** CSS linear-gradient (left → right) for swatches and the editor bar. */
export function gradientCss(g: Gradient, reverse = false): string {
  if (g.space === "oklab" && g.stops.length > 1) {
    // CSS interpolates in sRGB: the OkLab path is drawn as many short sRGB steps.
    const N = 32;
    const parts: string[] = [];
    for (let i = 0; i <= N; i++) {
      const x = i / N;
      const [r, gg, b, a] = gradientAt(g, reverse ? 1 - x : x);
      parts.push(`rgba(${Math.round(r * 255)},${Math.round(gg * 255)},${Math.round(b * 255)},${a.toFixed(3)}) ${(x * 100).toFixed(1)}%`);
    }
    return `linear-gradient(90deg, ${parts.join(", ")})`;
  }
  const parts = sorted(g).map((s) => {
    const [r, gg, b] = hexToRgb(s.color);
    const pos = reverse ? 1 - s.pos : s.pos;
    return { pos, css: `rgba(${Math.round(r * 255)},${Math.round(gg * 255)},${Math.round(b * 255)},${s.alpha}) ${(pos * 100).toFixed(1)}%` };
  }).sort((a, b) => a.pos - b.pos);
  return `linear-gradient(90deg, ${parts.map((p) => p.css).join(", ")})`;
}

// ---------------------------------------------------------------- harmonies

/**
 * Colour-harmony rules. Hues follow the rule from shadows to highlights (the
 * cooler hue in the shadows, the warmer in the light, as film grades do),
 * lightness rises evenly from dark to light, chroma arcs (quieter at both
 * ends, where strong colour looks dirty or neon).
 */
export type HarmonyRule = "analogous" | "complementary" | "split" | "triadic" | "mono" | "warmCool";
export const HARMONY_RULES: HarmonyRule[] = ["analogous", "complementary", "split", "triadic", "mono", "warmCool"];

/** How cool a hue is (1 = blue ≈ 250°, −1 = orange ≈ 70°). */
const coolness = (h: number) => Math.cos(((h - 250) * Math.PI) / 180);
const wrap = (h: number) => ((h % 360) + 360) % 360;
/** Hue between `a` and `b` along the shorter arc. */
function mixHue(a: number, b: number, t: number): number {
  const d = ((b - a + 540) % 360) - 180;
  return wrap(a + d * t);
}
/** Small deterministic generator: the same seed gives the same variation. */
function rand(seed: number): () => number {
  let x = seed >>> 0 || 1;
  return () => {
    x = (x + 0x6d2b79f5) >>> 0;
    let t = x;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A palette of `n` (3–5) colours from `base` by `rule`, dark → light. Every colour
 * has its own hue and real colour in it (the ends are not near-black / near-white),
 * so it reads as a set of colours that belong together, not one ramp. `seed` 0 is
 * the canonical palette; other seeds vary the lightness range, hues and chroma.
 */
export function harmonyPalette(base: string, rule: HarmonyRule, seed = 0, n = 5): string[] {
  n = Math.min(5, Math.max(3, Math.round(n)));
  const [, C0, h0raw] = hexToOklch(base);
  const grey = C0 < 0.02;
  const h0 = grey ? 250 : h0raw;
  const hash = [...base.toUpperCase()].reduce((a, c) => Math.imul(a ^ c.charCodeAt(0), 16777619), 2166136261);
  const r = rand(hash ^ Math.imul(seed + 1, 2654435761) ^ (HARMONY_RULES.indexOf(rule) * 7919) ^ (n * 104729));
  const j = (amp: number) => (seed === 0 ? 0 : (r() * 2 - 1) * amp);
  const Lmin = Math.min(0.4, Math.max(0.24, 0.3 + j(0.05)));
  const Lmax = Math.min(0.94, Math.max(0.84, 0.9 + j(0.03)));
  const Cpk = grey ? 0.06 : Math.min(0.2, Math.max(0.1, C0 * (1.1 + j(0.25))));
  // Hue anchors, shadows → highlights (the cooler hue in the shadows).
  const coolFirst = (hs: number[]) => [...hs].sort((a, b) => coolness(b) - coolness(a));
  let anchors: number[];
  switch (rule) {
    case "mono": anchors = [wrap(h0 + 14 + j(6)), wrap(h0 - 14 + j(6))]; break;
    case "analogous": {
      // Wide enough that neighbours are different colours, not shades of one.
      const span = 24 * (n - 1) + j(15);
      anchors = coolFirst([wrap(h0 - span / 2), wrap(h0 + span / 2)]);
      anchors.splice(1, 0, h0);
      break;
    }
    case "complementary": anchors = coolFirst([h0, wrap(h0 + 180 + j(15))]); break;
    case "split": {
      const sp = 150 + j(12);
      const [a, b] = coolFirst([wrap(h0 + sp), wrap(h0 - sp)]);
      anchors = [a, h0, b];
      break;
    }
    case "triadic": anchors = coolFirst([h0, wrap(h0 + 120 + j(10)), wrap(h0 + 240 + j(10))]); break;
    case "warmCool": anchors = [mixHue(255 + j(15), h0, 0.25), mixHue(70 + j(15), h0, 0.25)]; break;
  }
  const out: string[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const L = Lmin + (Lmax - Lmin) * t;
    // One hue per stop, along the anchors (the arc between opposite hues passes through
    // real colours on the way, not grey).
    const f = t * (anchors.length - 1);
    const k = Math.min(anchors.length - 2, Math.floor(f));
    const hue = mixHue(anchors[k], anchors[k + 1], f - k);
    // Strong colour everywhere; a little quieter at the light end, where full chroma looks neon.
    const C = Cpk * (0.85 + 0.15 * Math.sin(Math.PI * t)) * (1 - 0.25 * Math.max(0, t - 0.75) * 4);
    out.push(oklchToHex(L, C, hue));
  }
  return out;
}

/** CSS of a palette as separate blocks (how many colours, and which, at a glance). */
export function paletteCss(colors: string[]): string {
  const n = colors.length;
  return `linear-gradient(90deg, ${colors.map((c, i) => `${c} ${((i / n) * 100).toFixed(2)}% ${(((i + 1) / n) * 100).toFixed(2)}%`).join(", ")})`;
}

/**
 * A gradient-map palette from a set of colours (e.g. a photo's dominant ones): near
 * duplicates dropped, sorted dark → light, up to five spread over the lightness range,
 * then stretched to a full dark → light run with every step at least 0.08 lighter.
 */
export function paletteFromColors(colors: string[], max = 5): string[] {
  const labs = colors.map(hexToOklab).sort((a, b) => a[0] - b[0]);
  const kept: Array<[number, number, number]> = [];
  for (const c of labs) if (!kept.some((k) => Math.hypot(k[0] - c[0], k[1] - c[1], k[2] - c[2]) < 0.05)) kept.push(c);
  if (!kept.length) return ["#000000", "#FFFFFF"];
  let pick = kept;
  if (kept.length > max) pick = Array.from({ length: max }, (_, i) => kept[Math.round((i * (kept.length - 1)) / (max - 1))]);
  if (pick.length === 1) pick = [[0.18, pick[0][1] * 0.6, pick[0][2] * 0.6], pick[0], [0.95, pick[0][1] * 0.3, pick[0][2] * 0.3]];
  const lo = pick[0][0], hi = pick[pick.length - 1][0];
  const n = pick.length;
  return pick.map(([L, a, b], i) => {
    // Stretched to 0.16…0.94, halfway between the colours' own spacing and an even
    // one (so a photo of mostly mid-tones still gives a usable map); hue and chroma kept.
    const own = hi - lo > 1e-3 ? (L - lo) / (hi - lo) : i / (n - 1);
    const Lt = 0.16 + 0.78 * (0.5 * own + 0.5 * (i / (n - 1)));
    return [Lt, a, b] as [number, number, number];
  }).map((c, i, arr) => {
    if (i > 0) c[0] = Math.max(c[0], arr[i - 1][0] + 0.08);
    return oklabToHex(Math.min(0.97, c[0]), c[1], c[2]);
  });
}
