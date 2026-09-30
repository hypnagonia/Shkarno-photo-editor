/**
 * Adjustment layers (Photopea-style), applied on the developed image in the
 * tone pass: each layer computes an adjusted colour, blends it with what is
 * below by its blend mode, and mixes the result in by opacity × its mask.
 *
 * Masks are "smart": computed per pixel from what the analysis already knows —
 * a region's soft probability (or skin), a distance band, a region at a
 * distance, or a brightness range — so they follow real edges and cost nothing
 * to store. Stack order: first in the array = bottom (applied first).
 */
import type { CurvePoint, Curves, DepthBand, Region } from "../decision/params.ts";
import { presetGradient, type Gradient } from "./gradient.ts";

export type LayerType = "curves" | "hueSat" | "brightContrast" | "exposure" | "basic" | "gradientMap" | "gradientFill" | "blur" | "fog" | "light";
/** GPU type index = position here (layers.wgsl). */
export const LAYER_TYPES: LayerType[] = ["curves", "hueSat", "brightContrast", "exposure", "basic", "gradientMap", "gradientFill", "blur", "fog", "light"];

export type BlendMode = "normal" | "multiply" | "screen" | "overlay" | "softLight" | "hardLight" | "darken" | "lighten"
  | "hue" | "saturation" | "color" | "luminosity"
  | "colorBurn" | "linearBurn" | "colorDodge" | "linearDodge" | "vividLight" | "linearLight" | "pinLight" | "hardMix"
  | "difference" | "exclusion" | "subtract" | "divide" | "darkerColor" | "lighterColor";
/** GPU index = position here (layers.wgsl blend_modes): new modes go at the end. */
export const BLEND_MODES: BlendMode[] = ["normal", "multiply", "screen", "overlay", "softLight", "hardLight", "darken", "lighten", "hue", "saturation", "color", "luminosity",
  "colorBurn", "linearBurn", "colorDodge", "linearDodge", "vividLight", "linearLight", "pinLight", "hardMix",
  "difference", "exclusion", "subtract", "divide", "darkerColor", "lighterColor"];
/** The modes as Photoshop and Photopea group them in their menus. */
export const BLEND_GROUPS: Array<{ id: string; modes: BlendMode[] }> = [
  { id: "basic", modes: ["normal"] },
  { id: "darken", modes: ["darken", "multiply", "colorBurn", "linearBurn", "darkerColor"] },
  { id: "lighten", modes: ["lighten", "screen", "colorDodge", "linearDodge", "lighterColor"] },
  { id: "contrast", modes: ["overlay", "softLight", "hardLight", "vividLight", "linearLight", "pinLight", "hardMix"] },
  { id: "compare", modes: ["difference", "exclusion", "subtract", "divide"] },
  { id: "component", modes: ["hue", "saturation", "color", "luminosity"] },
];

/**
 * Blend If (Photoshop's Blending Options): the brightness range where the layer
 * shows, judged on the layer's own result ("this") and on what is under it.
 * low / high: 0…1 (0 and 1 = no limit); soft: how gradually it fades in or out there.
 */
export interface BlendRange { low: number; high: number; soft: number }
export interface BlendIf { this: BlendRange; under: BlendRange }
export const fullRange = (): BlendRange => ({ low: 0, high: 1, soft: 0.2 });
/** The four points (hidden below, full from, full to, hidden above) the shader ramps between. */
export function blendRamp(r: BlendRange | undefined): [number, number, number, number] {
  if (!r) return [0, 0, 1, 1];
  const s = Math.max(0, r.soft) / 2;
  const lo = r.low > 0 ? [Math.max(0, r.low - s), Math.min(1, r.low + s)] : [0, 0];
  const hi = r.high < 1 ? [Math.max(0, r.high - s), Math.min(1, r.high + s)] : [1, 1];
  return [lo[0], lo[1], hi[0], hi[1]];
}

export type MaskKind = "all" | "region" | "distance" | "cell" | "luminance" | "color" | "depth" | "object" | "select" | "shape";

/** A tap in a selection (tap-to-select, src/neural/sam.ts): x, y (0…1 of the photo) and 1 = part of it, 0 = not. */
export type SelectPoint = [number, number, 0 | 1];
/** A selection's identity: its taps and level (the engine caches the mask under it). */
export const selectKey = (m: MaskShape): string => JSON.stringify([m.points ?? [], m.level ?? "auto"]);

/** What a mask part selects, and how softly (the fields its kind uses). */
export interface MaskShape {
  kind: MaskKind;
  region?: Region;
  band?: DepthBand;
  /** Brightness range (display-encoded luma): low, high, softness. */
  lum?: [number, number, number];
  /** A colour (OkLab of the picked pixel before the layers) … */
  color?: [number, number, number];
  /** … and how far from it still counts (OkLab distance, lightness at half weight). */
  tol?: number;
  /** Distance range (0 = nearest … 1 = farthest): low, high, softness. "object" = `region` within it. */
  depth?: [number, number, number];
  /**
   * Shape (a graduated or radial filter), laid out as the Gradient Fill layer:
   * linear = on at the start side of a line at `angle` (degrees, 90 = from the top)
   * through (x, y), fading over `soft`; radial = on inside a circle of `scale` around
   * (x, y) (1 = reaching the picture's corners), fading out over `soft`.
   */
  shape?: { style: "linear" | "radial"; angle: number; scale: number; x: number; y: number; soft: number };
  /** Selection: the tap that made it, and which of SAM's readings (0 whole, 1 part, 2 detail; none = SAM's most confident). */
  points?: SelectPoint[];
  level?: 0 | 1 | 2;
  invert: boolean;
  /** 1 = the mask's natural soft edge, 0 = a hard edge at its 50 % point. */
  feather: number;
}

/** How an extra part combines with the mask so far (as Lightroom's add / subtract / intersect). */
export type MaskOp = "add" | "subtract" | "intersect";
export const MASK_OPS: MaskOp[] = ["add", "subtract", "intersect"];
export interface MaskPart extends MaskShape { kind: Exclude<MaskKind, "all">; op: MaskOp }
/** At most this many extra parts per layer (the GPU record has room for them). */
export const MAX_MASK_PARTS = 4;

export interface SmartMask extends MaskShape {
  /** Extra parts, applied in order after the main one. */
  parts?: MaskPart[];
  /** Mask strength 0…1 (multiplies opacity). */
  density: number;
  /** Leaves skin out (region and distance colour: faces keep their own correction). */
  exceptSkin?: boolean;
}

/** Hue/Saturation: master and six colour ranges (as Photoshop: reds, yellows, greens, cyans, blues, magentas). */
export type HueRange = "master" | "reds" | "yellows" | "greens" | "cyans" | "blues" | "magentas";
export const HUE_RANGES: HueRange[] = ["master", "reds", "yellows", "greens", "cyans", "blues", "magentas"];
/** Centre hue (degrees) of each colour range. */
export const RANGE_CENTRE: Record<Exclude<HueRange, "master">, number> = { reds: 0, yellows: 60, greens: 120, cyans: 180, blues: 240, magentas: 300 };
export interface HueSatAdjust { hue: number; sat: number; light: number; /** Range limits (degrees): inner half-width (full effect) and outer (fade). */ inner?: number; outer?: number }

export interface LayerParams {
  curves: Curves;
  hueSat: { ranges: Partial<Record<HueRange, HueSatAdjust>>; colorize: boolean; cHue: number; cSat: number; cLight: number };
  brightContrast: { brightness: number; contrast: number };
  exposure: { exposure: number; offset: number; gamma: number };
  /** Local light & colour (as Lightroom's local adjustments): exposure EV, temperature, tint, saturation, vibrance, hue (degrees). */
  basic: { exposure: number; temp: number; tint: number; saturation: number; vibrance: number; hue: number };
  /** Gradient Map: the pixel's brightness picks a colour (left = shadows). `preset`: the palette it came from. */
  gradientMap: { gradient: Gradient; reverse: boolean; preset?: string };
  /**
   * Gradient Fill: a gradient laid over the photo. Linear: across the picture at
   * `angle` (degrees, 90 = top → bottom); radial: from the centre (`x`, `y`, 0…1) out.
   * `scale` 1 = the gradient spans the picture.
   */
  gradientFill: { gradient: Gradient; style: "linear" | "radial"; angle: number; scale: number; x: number; y: number; reverse: boolean; preset?: string };
  /**
   * Blur of what the mask covers. Lens (default): the depth-of-field gather, so nearer,
   * sharper things keep clean edges; `amount` 1 = a radius of 3 % of the picture's long
   * side. Motion: streaks along `angle` (degrees, 0 = horizontal, 90 = vertical), as a
   * moving subject or a panned camera; `amount` 1 = a streak of 6 % of the long side.
   * `depth`: into the depth — streaks run away from the vanishing point (the centre of
   * the farthest depth), scaled by nearness (near things up to 1.6×, the far distance
   * hardly at all), as when moving into the scene; `angle` is then unused. `through`:
   * the streaks run through the mask's edge instead of stopping at it (the mask only
   * mixes the result in), so what it keeps sharp trails into the streaks around it.
   * `vanish` (0…1): where the masked part recedes to — the far end of what the mask
   * covers, worked out by the engine from the mask and the depth map (absent: the
   * photo's own vanishing point). `range`: the masked part's own depth (near, far) and
   * `reach` its farthest extent from `vanish` (share of the height): into the depth,
   * a streak is longest at the part's near end and far from `vanish`, shortest at its
   * deep end — the flow of a forward motion, ∝ r / Z — at `falloff` −1; 0 the same
   * streak all along; +1 the deep end streaked most (a zoom burst, the part dissolving
   * into the depth). Absent: +0.5.
   * `object`: the mask is a moving object (a vehicle, a horse, a runner), not the scene —
   * it is smeared along `angle` (its travel direction) past its own outline over the
   * background, which the engine inpaints where the object was; `trail` 0 (streaked
   * both ways) … 1 (all behind it), `sharp` 0…1 the object itself sharp on top. With
   * `depth`, it moves along the depth instead of `angle`: into it, or toward the camera
   * (`arriving`) — a scale about `vanish`.
   */
  blur: { amount: number; motion?: boolean; angle?: number; depth?: boolean; through?: boolean; vanish?: [number, number]; range?: [number, number]; reach?: number; falloff?: number; object?: boolean; trail?: number; sharp?: number; arriving?: boolean };
  /**
   * Fog: air thickening with distance (the depth map) — from `start` (0 = the nearest,
   * 1 = the farthest) on, toward `color` (hex), by the real distance through the air
   * (the depth map is disparity: distance ∝ 1 / (1 − d)), so it thickens all the way to
   * the horizon; `amount` 0…3 its density. Mixed in linear light.
   */
  fog: { amount: number; start: number; color: string };
  /**
   * Light by nearness: a fill that is strongest at the front and fades with distance
   * (a flash, a reflector), `amount` −2…4 EV; `reach` 0…1.5 how far into the depth it
   * carries (beyond 1 the far distance is lit too). Highlights roll off into white.
   */
  light: { amount: number; reach: number };
}

export interface Layer<T extends LayerType = LayerType> {
  id: string;
  type: T;
  name: string;
  visible: boolean;
  opacity: number;
  blend: BlendMode;
  /** Blend If: shown only where this layer / the image under it is in a brightness range. */
  blendIf?: BlendIf;
  mask: SmartMask;
  /** Generated by the automatic grading (the rule id); edits keep it, "reset" restores it. */
  auto?: string;
  /** Part of a lens flare (src/layers/flare.ts): its set, its role, and a ghost's place on the axis. */
  flare?: { set: string; role: "veil" | "glow" | "streak" | "ghost"; k?: number };
  params: LayerParams[T];
}

const FLAT: CurvePoint[] = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
export const flatCurves = (): Curves => ({ l: [...FLAT], r: [...FLAT], g: [...FLAT], b: [...FLAT] });
export const allMask = (): SmartMask => ({ kind: "all", invert: false, feather: 1, density: 1 });

export function defaultParams<T extends LayerType>(type: T): LayerParams[T] {
  const d: LayerParams = {
    curves: flatCurves(),
    hueSat: { ranges: {}, colorize: false, cHue: 30, cSat: 0.25, cLight: 0 },
    brightContrast: { brightness: 0, contrast: 0 },
    exposure: { exposure: 0, offset: 0, gamma: 1 },
    basic: { exposure: 0, temp: 0, tint: 0, saturation: 0, vibrance: 0, hue: 0 },
    gradientMap: { gradient: presetGradient("tealGold"), reverse: false, preset: "tealGold" },
    // Foreground to transparent from the top: a graduated filter (darker sky).
    gradientFill: { gradient: { stops: [{ pos: 0, color: "#101820", alpha: 0.75 }, { pos: 0.55, color: "#101820", alpha: 0 }] }, style: "linear", angle: 90, scale: 1, x: 0.5, y: 0.5, reverse: false },
    blur: { amount: 0.4 },
    fog: { amount: 0.5, start: 0.45, color: "#cfd8e0" },
    light: { amount: 1, reach: 0.6 },
  };
  return structuredClone(d[type]) as LayerParams[T];
}

let counter = 0;
export function newId(): string { return `l${Date.now().toString(36)}${(counter++).toString(36)}`; }

export function makeLayer<T extends LayerType>(type: T, name: string, over: Partial<Omit<Layer<T>, "type">> = {}): Layer<T> {
  return { id: newId(), type, name, visible: true, opacity: 1, blend: "normal", mask: allMask(), params: defaultParams(type), ...over };
}

/** How a new layer of this type starts (Photopea starts every layer Normal 100 %; a colour grade reads better softer). */
export function newLayerDefaults(type: LayerType): Partial<Layer> {
  if (type === "gradientMap") return { blend: "softLight", opacity: 0.7 };
  return {};
}

/** A layer that changes nothing (all parameters at their neutral value). */
export function isNeutralLayer(l: Layer): boolean {
  switch (l.type) {
    case "curves": { const c = l.params as Curves; return (["l", "r", "g", "b"] as const).every((k) => (c[k] ?? FLAT).every((q) => Math.abs(q.x - q.y) < 1e-4)); }
    case "hueSat": { const h = l.params as LayerParams["hueSat"]; return !h.colorize && Object.values(h.ranges).every((r) => !r || (Math.abs(r.hue) < 1e-4 && Math.abs(r.sat) < 1e-4 && Math.abs(r.light) < 1e-4)); }
    case "brightContrast": { const b = l.params as LayerParams["brightContrast"]; return Math.abs(b.brightness) < 1e-4 && Math.abs(b.contrast) < 1e-4; }
    case "exposure": { const e = l.params as LayerParams["exposure"]; return Math.abs(e.exposure) < 1e-4 && Math.abs(e.offset) < 1e-4 && Math.abs(e.gamma - 1) < 1e-4; }
    case "basic": { const b = l.params as LayerParams["basic"]; return Object.values(b).every((v) => Math.abs(v) < 1e-4); }
    case "blur": return (l.params as LayerParams["blur"]).amount < 1e-4;
    case "fog": return (l.params as LayerParams["fog"]).amount < 1e-4;
    case "light": return Math.abs((l.params as LayerParams["light"]).amount) < 1e-4;
  }
  return false;
}

/**
 * Per-hue response of a Hue/Saturation layer, sampled at `n` hues over 0…360°:
 * [hue shift (deg), saturation, lightness] per sample, from the master setting
 * plus every colour range weighted by a trapezoid (full inside `inner`, fading
 * to zero at `outer` degrees from the range's centre).
 */
export function hueSatTable(h: LayerParams["hueSat"], n = 360): Float32Array {
  const out = new Float32Array(n * 3);
  const m = h.ranges.master ?? { hue: 0, sat: 0, light: 0 };
  for (let i = 0; i < n; i++) {
    const deg = (i / n) * 360;
    let dh = m.hue, ds = m.sat, dl = m.light;
    for (const [k, c] of Object.entries(RANGE_CENTRE) as Array<[Exclude<HueRange, "master">, number]>) {
      const r = h.ranges[k];
      if (!r) continue;
      const inner = r.inner ?? 15, outer = Math.max(inner + 1, r.outer ?? 45);
      let d = Math.abs(deg - c) % 360; if (d > 180) d = 360 - d;
      const w = d <= inner ? 1 : d >= outer ? 0 : 1 - (d - inner) / (outer - inner);
      dh += w * r.hue; ds += w * r.sat; dl += w * r.light;
    }
    out[i * 3] = dh; out[i * 3 + 1] = ds; out[i * 3 + 2] = dl;
  }
  return out;
}
