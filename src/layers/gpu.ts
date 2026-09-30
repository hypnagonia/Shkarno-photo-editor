/**
 * Layers → GPU: one fixed 32-float record per layer (read-only storage buffer)
 * and one row per layer that needs a table in the LUT atlas (1024 samples × 4
 * channels, rgba16float). Mirrors `layers.wgsl`.
 *
 * Record: [0] type  [1] blend  [2] opacity (0 = hidden)  [3] atlas row (−1 none)
 *         [4] mask kind  [5] region (0…10 groups, 11 skin)  [6] band  [7] invert
 *         [8…10] mask values 0–2  [11] feather  [12] density  [13] except skin  [14…15] mask values 3–4
 *         [16…31] type parameters
 *         [32…79] extra mask parts, 12 floats each (kind 0 = none):
 *                 kind, region, band, op | values 0–3 | value 4, invert, feather, _
 * Mask values by kind — luminance: low, high, soft; color: L, a, b, tolerance; depth: low, high, soft; object: region within depth low, high, soft;
 *   shape: angle (rad), size, x, y, softness (the style in the region field: 0 linear, 1 radial);
 *   select: layer of the selection texture (−1 = not ready: nothing).
 */
import { GROUPS } from "../neural/scene.ts";
import { DEPTH_BANDS, type Curves } from "../decision/params.ts";
import { curveLUT, CURVE_LUT_SIZE } from "../render/curves.ts";
import { BLEND_MODES, blendRamp, hueSatTable, LAYER_TYPES, MAX_MASK_PARTS, type Layer, type LayerParams, type MaskKind, type MaskOp, type MaskShape } from "./model.ts";
import { gradientTable } from "./gradient.ts";

export const RECORD = 80;

/**
 * Atlas rows by the layer settings that make them: a slider drag changes one
 * layer, so the other layers' tables (curve splines, hue and gradient tables)
 * are not rebuilt on every frame.
 */
const rowCache = new Map<string, Float32Array>();
function cachedRow(key: string, make: () => Float32Array): Float32Array {
  let r = rowCache.get(key);
  if (r) { rowCache.delete(key); rowCache.set(key, r); return r; } // most recently used last
  r = make();
  rowCache.set(key, r);
  if (rowCache.size > 64) rowCache.delete(rowCache.keys().next().value!);
  return r;
}
export const ATLAS_W = CURVE_LUT_SIZE; // 1024
const MASK_KIND: Record<MaskKind, number> = { all: 0, region: 1, distance: 2, cell: 3, luminance: 4, color: 5, depth: 6, object: 7, select: 8, shape: 9 };
const MASK_OP: Record<MaskOp, number> = { add: 0, subtract: 1, intersect: 2 };

/** The five numbers a mask kind reads (see the record layout). `slotOf`: a selection's layer in the selection texture. */
function maskValues(m: MaskShape, slotOf?: (m: MaskShape) => number): number[] {
  if (m.kind === "select") return [slotOf?.(m) ?? -1, 0, 0, 0, 0];
  if (m.kind === "shape") { const g = m.shape ?? defaultShape("linear"); return [(g.angle * Math.PI) / 180, g.scale, g.x, g.y, g.soft]; }
  if (m.kind === "color") return [...(m.color ?? [0.5, 0, 0]), m.tol ?? 0.08, 0];
  if (m.kind === "depth" || m.kind === "object") return [...(m.depth ?? [0, 0.3, 0.05]), 0, 0];
  return [m.lum?.[0] ?? 0, m.lum?.[1] ?? 1, m.lum?.[2] ?? 0.08, 0, 0];
}
const regionIndex = (m: MaskShape) => (m.kind === "shape" ? (m.shape?.style === "radial" ? 1 : 0) : m.region === "skin" ? 11 : m.region ? GROUPS.indexOf(m.region) : 0);
/** A new shape mask: a graduated filter over the top half, or a spotlight in the middle. */
export const defaultShape = (style: "linear" | "radial"): NonNullable<MaskShape["shape"]> =>
  style === "linear" ? { style, angle: 90, scale: 1, x: 0.5, y: 0.5, soft: 0.6 } : { style, angle: 0, scale: 0.6, x: 0.5, y: 0.5, soft: 0.5 };

export interface PackedLayers { records: Float32Array; count: number; atlas: Float32Array; rows: number }

/**
 * The module an automatic layer belongs to (for the Regions / Curves switches):
 * the photo's tone curve and the distance curves are "curves", the rest of the
 * automatic grade (region colour, skin, subject, region curves) is "semantic".
 * The user's own layers follow "curves".
 */
export function layerModule(l: Layer): "curves" | "semantic" {
  if (!l.auto) return "curves";
  const [kind, what] = l.auto.split(".");
  if (kind === "curves" && (what === "photo" || what === "black" || (DEPTH_BANDS as string[]).includes(what))) return "curves";
  return "semantic";
}

/** The layers that reach the GPU, in order (the mask view's index counts these). */
export function liveLayers(layers: Layer[], autoStrength = 1, enable?: { curves?: boolean; semantic?: boolean }): Layer[] {
  const on = (l: Layer) => (enable?.[layerModule(l)] ?? true) !== false;
  return layers.filter((l) => l.visible && l.opacity > 0 && on(l) && !(l.auto && autoStrength <= 0));
}

/**
 * `autoStrength` (0…1) scales the opacity of every automatic layer (the "Auto strength" control);
 * `enable` (the module switches) turns layers off with their module.
 */
export function packLayers(layers: Layer[], autoStrength = 1, enable?: { curves?: boolean; semantic?: boolean }, slotOf?: (m: MaskShape) => number): PackedLayers {
  const live = liveLayers(layers, autoStrength, enable);
  const records = new Float32Array(Math.max(1, live.length) * RECORD);
  const rows: Float32Array[] = [];
  live.forEach((l, i) => {
    const r = records.subarray(i * RECORD, (i + 1) * RECORD);
    r[0] = LAYER_TYPES.indexOf(l.type);
    r[1] = Math.max(0, BLEND_MODES.indexOf(l.blend));
    r[2] = Math.min(1, Math.max(0, l.opacity * (l.auto ? Math.min(1, Math.max(0, autoStrength)) : 1)));
    r[3] = -1;
    // Blend If ramps (p2 = this layer, p3 = under it); type parameters use p0–p1 only.
    r.set(blendRamp(l.blendIf?.this), 24); r.set(blendRamp(l.blendIf?.under), 28);
    const m = l.mask;
    r[4] = MASK_KIND[m.kind];
    r[5] = regionIndex(m);
    r[6] = m.band ? DEPTH_BANDS.indexOf(m.band) : 0;
    r[7] = m.invert ? 1 : 0;
    const v = maskValues(m, slotOf);
    r[8] = v[0]; r[9] = v[1]; r[10] = v[2]; r[14] = v[3]; r[15] = v[4];
    r[11] = m.feather; r[12] = m.density; r[13] = m.exceptSkin ? 1 : 0;
    (m.parts ?? []).slice(0, MAX_MASK_PARTS).forEach((part, k) => {
      const q = r.subarray(32 + k * 12, 44 + k * 12);
      const pv = maskValues(part, slotOf);
      q[0] = MASK_KIND[part.kind]; q[1] = regionIndex(part); q[2] = part.band ? DEPTH_BANDS.indexOf(part.band) : 0; q[3] = MASK_OP[part.op];
      q[4] = pv[0]; q[5] = pv[1]; q[6] = pv[2]; q[7] = pv[3];
      q[8] = pv[4]; q[9] = part.invert ? 1 : 0; q[10] = part.feather;
    });
    const p = r.subarray(16);
    switch (l.type) {
      case "curves": {
        r[3] = rows.length;
        rows.push(cachedRow("c" + JSON.stringify(l.params), () => curveLUT(l.params as Curves)));
        break;
      }
      case "hueSat": {
        const h = l.params as LayerParams["hueSat"];
        p[0] = h.colorize ? 1 : 0; p[1] = (h.cHue * Math.PI) / 180; p[2] = h.cSat; p[3] = h.cLight;
        // Per-hue table: r = hue shift (radians), g = saturation, b = lightness; 1024 samples over 0…360°.
        r[3] = rows.length;
        rows.push(cachedRow("h" + JSON.stringify(h.ranges), () => {
          const t = hueSatTable(h, ATLAS_W), row = new Float32Array(ATLAS_W * 4);
          for (let k = 0; k < ATLAS_W; k++) { row[k * 4] = (t[k * 3] * Math.PI) / 180; row[k * 4 + 1] = t[k * 3 + 1]; row[k * 4 + 2] = t[k * 3 + 2]; row[k * 4 + 3] = 1; }
          return row;
        }));
        break;
      }
      case "gradientMap": {
        const g = l.params as LayerParams["gradientMap"];
        p[0] = g.reverse ? 1 : 0;
        r[3] = rows.length;
        rows.push(cachedRow("g" + JSON.stringify(g.gradient), () => gradientTable(g.gradient, ATLAS_W)));
        break;
      }
      case "gradientFill": {
        const g = l.params as LayerParams["gradientFill"];
        p[0] = g.style === "radial" ? 1 : 0; p[1] = (g.angle * Math.PI) / 180; p[2] = Math.max(0.02, g.scale); p[3] = g.reverse ? 1 : 0;
        p[4] = g.x; p[5] = g.y;
        r[3] = rows.length;
        rows.push(cachedRow("g" + JSON.stringify(g.gradient), () => gradientTable(g.gradient, ATLAS_W)));
        break;
      }
      case "blur": { const b = l.params as LayerParams["blur"]; p[0] = Math.max(0, b.amount); p[1] = b.motion ? 1 : 0; p[2] = ((b.angle ?? 0) * Math.PI) / 180; p[3] = b.depth ? 1 : 0; break; }
      case "brightContrast": { const b = l.params as LayerParams["brightContrast"]; p[0] = b.brightness; p[1] = b.contrast; break; }
      case "exposure": { const e = l.params as LayerParams["exposure"]; p[0] = e.exposure; p[1] = e.offset; p[2] = e.gamma; break; }
      case "basic": {
        const b = l.params as LayerParams["basic"];
        p[0] = b.exposure; p[1] = b.temp; p[2] = b.tint; p[3] = b.saturation; p[4] = b.vibrance; p[5] = (b.hue * Math.PI) / 180;
        break;
      }
    }
  });
  const atlas = new Float32Array(Math.max(1, rows.length) * ATLAS_W * 4);
  rows.forEach((row, i) => atlas.set(row, i * ATLAS_W * 4));
  return { records, count: live.length, atlas, rows: Math.max(1, rows.length) };
}

/** Whether any visible lens Blur layer blurs something (the renderer then runs the blur pass). */
export function hasBlurLayers(layers: Layer[], autoStrength = 1, enable?: { curves?: boolean; semantic?: boolean }): boolean {
  return liveLayers(layers, autoStrength, enable).some((l) => l.type === "blur" && (l.params as LayerParams["blur"]).amount > 0 && !(l.params as LayerParams["blur"]).motion);
}
/** Whether any visible motion Blur layer blurs something (the renderer then runs the motion pass). */
export function hasMotionLayers(layers: Layer[], autoStrength = 1, enable?: { curves?: boolean; semantic?: boolean }): boolean {
  return liveLayers(layers, autoStrength, enable).some((l) => l.type === "blur" && (l.params as LayerParams["blur"]).amount > 0 && !!(l.params as LayerParams["blur"]).motion);
}
