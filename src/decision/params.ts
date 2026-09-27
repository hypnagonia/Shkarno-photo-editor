/**
 * Processing parameters — the single contract between the decision engine
 * (which produces them from measurements) and the renderer (which applies
 * them). Everything here is plain data so that it can be logged, diffed,
 * edited in the UI and serialised with an export.
 */
import { GROUPS, type Group } from "../neural/scene.ts";
import type { LookProfile } from "../looks/profile.ts";
import type { Layer } from "../layers/model.ts";
import { BUILTIN_PROFILES, DEFAULT_PROFILE_ID } from "../looks/builtin.ts";

export interface CurvePoint { x: number; y: number }

/** Point curves per channel: l (luminance), r, g, b — display-encoded 0…1. */
export interface Curves { l: CurvePoint[]; r: CurvePoint[]; g: CurvePoint[]; b: CurvePoint[] }

/** A region with its own adjustments: a semantic group, or skin (a layer across people). */
export type Region = Group | "skin";

/** Distance bands with their own curves: soft thirds of this photo's depth range. */
export type DepthBand = "near" | "middle" | "far";
export const DEPTH_BANDS: DepthBand[] = ["near", "middle", "far"];

/**
 * A region at a distance — "building.near", "building.far": semantic and depth
 * maps together, so two buildings at different distances are separate. Cell
 * index = group index × 3 + band index (render_tone.wgsl).
 */
export type CellKey = `${Group}.${DepthBand}`;
export const cellKey = (g: Group, b: DepthBand): CellKey => `${g}.${b}`;
export const CELLS: CellKey[] = GROUPS.flatMap((g) => DEPTH_BANDS.map((b) => cellKey(g, b)));

/** A user-chosen focus point: position in the image (0..1) and the refined distance there. */
export interface FocusPoint {
  x: number; y: number; dist: number;
  /** The automatic subject, kept as a point once manual points are added (until it is moved). */
  auto?: boolean;
  /**
   * The depth range of the object under the point (refined distance, lo…hi): all of
   * it stays sharp — a person turned three-quarters, a long table, a car at an angle.
   */
  range?: [number, number];
}

export const MAX_FOCUS_POINTS = 8;

export interface SemanticAdjust {
  /** Local exposure offset (EV), e.g. a backlit person. */
  exposure: number;
  /** Extra highlight compression for this region (0..1). */
  highlights: number;
  /** Saturation multiplier delta (0 = unchanged, +0.1 = ×1.1). */
  saturation: number;
  vibrance: number;
  /** OkLab hue rotation in degrees. */
  hue: number;
  /** Multipliers applied to the global value (1 = unchanged). */
  clarity: number;
  texture: number;
  sharpen: number;
  denoise: number;
  dehaze: number;
  /** Colour temperature (−1 cool … +1 warm) and tint (−1 green … +1 magenta) of this region. */
  warmth: number;
  tint: number;
}

export interface Params {
  /** Master switch per stage (for A/B debugging). */
  enable: {
    denoise: boolean; wb: boolean; exposure: boolean; localTone: boolean;
    curves: boolean; lut: boolean; semantic: boolean; dehaze: boolean; sharpen: boolean; dof: boolean;
  };
  exposure: number;
  /** Absolute rendering white: temperature (K) and tint. */
  wb: { temp: number; tint: number };
  tone: {
    highlights: number; // −1..1  (negative recovers/compresses)
    shadows: number; // −1..1  (positive lifts)
    whites: number; // −1..1
    blacks: number; // −1..1
    contrast: number; // −1..1
    rolloff: number; // 0..1 shoulder softness
  };
  local: {
    /** Compression of the large-scale (coarse base) log range, 0..0.8. */
    compression: number;
    /** Medium-scale detail gain (clarity), −0.5..1. */
    clarity: number;
    /** Fine detail gain (texture), −0.5..1. */
    texture: number;
    /** Pivot of range compression and shadows/highlights: the scene's own key
     * (log2 luminance, before exposure) — not a fixed middle grey. */
    anchorEV: number;
  };
  color: { saturation: number; vibrance: number };
  curves: Curves;
  /** Curves per region, blended by the soft masks (only regions that have them). */
  regionCurves: Partial<Record<Region, Curves>>;
  /** Curves per distance band, blended by soft band weights (only bands that have them). */
  depthCurves: Partial<Record<DepthBand, Curves>>;
  /**
   * HDR: how far highlights may rise above SDR white on HDR screens (stops,
   * 0 … 3; 0 = SDR only). The SDR rendering does not depend on it.
   */
  hdr: { headroom: number };
  /**
   * Adjustment layers over the developed image (src/layers), bottom first: the
   * automatic grade (generated from the decisions, `auto` set) and the user's own.
   */
  layers: Layer[];
  /** Version of the automatic layers (raised when their conversion changes; older saved ones are replaced). */
  autoLayersVersion?: number;
  /** Strength of the automatic curves (0 … 1.5; 1 = as measured, 0 = none). */
  autoCurves: number;
  /**
   * Highlight protection: the layers (automatic and the user's) may not push a
   * colour that was below clipping into it; they approach white on a soft shoulder.
   * Undefined = on (sessions saved before it existed).
   */
  protectHighlights?: boolean;
  /** Where near ends and far begins, in refined distance (0 near … 1 far): this photo's own depth layers. */
  depthBands: [number, number];
  /** The creative layer: one look profile on top of the technical base. */
  profile: LookProfile;
  denoise: { luma: number; chroma: number; shadowBoost: number };
  sharpen: { amount: number; radius: number; threshold: number };
  dehaze: { strength: number; light: [number, number, number]; beta: number; minT: number };
  semantic: Record<Group, SemanticAdjust>;
  /**
   * Skin: the same adjustments as a region, applied on top of whichever region
   * the pixel belongs to, weighted by skin likelihood (Apple's skin matte on
   * ProRAW, else the person mask × a skin-colour likelihood).
   */
  skin: SemanticAdjust;
  /**
   * Adjustments by distance (everything near / middle / far) and by region at a
   * distance (cells). Both are *relative*, layered on the region's own settings:
   * exposure, highlights, warmth, tint, saturation, vibrance and hue add;
   * clarity, texture, sharpening, noise reduction and dehaze multiply.
   * Neutral = no change. Order: region → distance → cell → skin.
   */
  distance: Record<DepthBand, SemanticAdjust>;
  cells: Partial<Record<CellKey, SemanticAdjust>>;
  /** Curves per cell, after the region and distance curves. */
  cellCurves: Partial<Record<CellKey, Curves>>;
  depth: { near: number; far: number };
  /**
   * Vignette, applied in linear light as an exposure falloff (like a lens):
   * amount −1 … 1 (negative darkens the edges, ≈ −2 EV in the corners at −1;
   * positive lightens, up to +1 EV), midpoint 0 … 1 (where the falloff starts),
   * feather 0 … 1 (how gradual), roundness 0 … 1 (0 follows the frame's
   * aspect, 1 is a circle), highlights 0 … 1 (bright light sources keep their
   * brightness when darkening).
   */
  vignette: { amount: number; midpoint: number; feather: number; roundness: number; highlights: number };
  /**
   * Film grain, added last (after sharpening and depth of field): amount 0 … 1,
   * size 0 … 1 (fine … coarse, relative to the image, so every resolution
   * gets the same film), roughness 0 … 1 (clumping), colour 0 … 1
   * (0 = monochrome grain, 1 = dye-cloud colour grain).
   */
  grain: { amount: number; size: number; roughness: number; color: number };
  /**
   * Depth of field. With no `points`, `focus` (automatic) is the single focal
   * distance. With points, a pixel stays sharp if it is near the distance of
   * *any* point — several subjects at different distances can all be sharp,
   * which no single focal plane can do.
   */
  dof: {
    focus: number; strength: number; points: FocusPoint[]; auto: boolean;
    /** Depth extent of the automatic subject around `focus` (below, above): all of it stays sharp. */
    focusSpan?: [number, number];
    /** "focus": blur from the focus distance / points. "zones": blur set per depth zone by hand. */
    mode?: "focus" | "zones";
    /** Blur 0..1 for each depth zone (near → far). */
    zones?: number[];
    /** The 4 inner zone boundaries in distance units (natural breaks of this photo's depth). */
    zoneBounds?: number[];
  };
}

export function neutralSemantic(): SemanticAdjust {
  return { exposure: 0, highlights: 0, saturation: 0, vibrance: 0, hue: 0, clarity: 1, texture: 1, sharpen: 1, denoise: 1, dehaze: 1, warmth: 0, tint: 0 };
}

export function defaultParams(): Params {
  const flat: CurvePoint[] = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
  return {
    enable: { denoise: true, wb: true, exposure: true, localTone: true, curves: true, lut: true, semantic: true, dehaze: true, sharpen: true, dof: false },
    exposure: 0,
    wb: { temp: 6504, tint: 0 },
    tone: { highlights: 0, shadows: 0, whites: 0, blacks: 0, contrast: 0, rolloff: 0.5 },
    local: { compression: 0, clarity: 0, texture: 0, anchorEV: Math.log2(0.18) },
    color: { saturation: 0, vibrance: 0 },
    curves: { l: [...flat], r: [...flat], g: [...flat], b: [...flat] },
    profile: structuredClone(BUILTIN_PROFILES.find((q) => q.id === DEFAULT_PROFILE_ID)!),
    denoise: { luma: 0, chroma: 0, shadowBoost: 0 },
    sharpen: { amount: 0, radius: 1, threshold: 0.01 },
    dehaze: { strength: 0, light: [1, 1, 1], beta: 1, minT: 0.45 },
    semantic: Object.fromEntries(GROUPS.map((g) => [g, neutralSemantic()])) as Record<Group, SemanticAdjust>,
    skin: neutralSemantic(),
    distance: { near: neutralSemantic(), middle: neutralSemantic(), far: neutralSemantic() },
    cells: {},
    cellCurves: {},
    regionCurves: {},
    depthCurves: {},
    depthBands: [0.33, 0.66],
    autoCurves: 1,
    protectHighlights: true,
    layers: [],
    hdr: { headroom: 0 },
    depth: { near: 1, far: 1 },
    vignette: { amount: 0, midpoint: 0.5, feather: 0.6, roundness: 0.3, highlights: 0.5 },
    grain: { amount: 0, size: 0.35, roughness: 0.5, color: 0 },
    dof: { focus: 0.3, strength: 0, points: [], auto: false },
  };
}

/** One inspectable automatic decision. */
export interface Decision {
  id: string;
  value: number | string | boolean | number[];
  reason: string;
  inputs: Record<string, number | string>;
}
