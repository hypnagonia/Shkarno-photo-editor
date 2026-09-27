/**
 * Lens flare, built from ordinary layers (nothing new in the renderer):
 *
 *   veil    Light & colour, a radial mask on the light: the washed-out haze
 *   glow    Gradient Fill, radial, Screen: the bright bloom around the light
 *   streak  Gradient Fill, a thin linear band through the light, Screen, fading out
 *           sideways under a radial mask
 *   ghosts  small tinted radial Gradient Fills, Screen, on the line from the light
 *           through the frame's centre (where a lens's internal reflections land)
 *
 * The layers of one flare share `flare.set`, so the whole flare moves with its
 * light (moveFlare) and each stays editable on its own.
 */
import { defaultShape } from "./gpu.ts";
import { allMask, defaultParams, makeLayer, newId, type Layer, type LayerParams } from "./model.ts";

export interface Point { x: number; y: number }
/** Ghosts: position along the axis (C + k·(C − L)), size, colour. */
const GHOSTS: Array<{ k: number; size: number; color: string }> = [
  { k: 0.4, size: 0.08, color: "#7FD6FF" },
  { k: 0.9, size: 0.05, color: "#B58CFF" },
  { k: 1.3, size: 0.12, color: "#9BFF9B" },
];
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
/** Where a ghost lands for a light at `L`: on the line from the light through the centre. */
export function ghostAt(L: Point, k: number): Point {
  return { x: clamp01(0.5 + k * (0.5 - L.x)), y: clamp01(0.5 + k * (0.5 - L.y)) };
}

const fill = (over: Partial<LayerParams["gradientFill"]>): LayerParams["gradientFill"] => ({ ...defaultParams("gradientFill"), preset: undefined, ...over });
const radialMask = (L: Point, scale: number) => ({ ...allMask(), kind: "shape" as const, shape: { ...defaultShape("radial"), x: L.x, y: L.y, scale, soft: 1 } });

/** The layers of a new flare with its light at `L`, bottom first (the order they stack in). */
export function flareLayers(L: Point, names: { veil: string; glow: string; streak: string; ghost: string }): Layer[] {
  const set = newId();
  const tag = (role: NonNullable<Layer["flare"]>["role"], k?: number) => ({ set, role, ...(k !== undefined ? { k } : {}) });
  const veil = makeLayer("basic", names.veil, {
    params: { ...defaultParams("basic"), exposure: 0.3, saturation: -0.2 },
    mask: radialMask(L, 0.9), flare: tag("veil"),
  });
  const glow = makeLayer("gradientFill", names.glow, {
    blend: "screen", opacity: 0.7, flare: tag("glow"),
    params: fill({ style: "radial", scale: 0.35, x: L.x, y: L.y, gradient: { space: "oklab", stops: [
      { pos: 0, color: "#FFF1D6", alpha: 0.9 }, { pos: 0.25, color: "#FFB060", alpha: 0.35 }, { pos: 0.6, color: "#FFB060", alpha: 0 },
    ] } }),
  });
  // Angle 90°: the gradient runs top → bottom, so its narrow middle is a horizontal band.
  const streak = makeLayer("gradientFill", names.streak, {
    blend: "screen", opacity: 0.8, flare: tag("streak"), mask: radialMask(L, 0.5),
    params: fill({ style: "linear", angle: 90, scale: 0.35, x: L.x, y: L.y, gradient: { space: "oklab", stops: [
      { pos: 0.47, color: "#FFE9C4", alpha: 0 }, { pos: 0.5, color: "#FFE9C4", alpha: 0.8 }, { pos: 0.53, color: "#FFE9C4", alpha: 0 },
    ] } }),
  });
  const ghosts = GHOSTS.map((g, i) => {
    const at = ghostAt(L, g.k);
    return makeLayer("gradientFill", `${names.ghost} ${i + 1}`, {
      blend: "screen", opacity: 0.6, flare: tag("ghost", g.k),
      params: fill({ style: "radial", scale: g.size, x: at.x, y: at.y, gradient: { space: "oklab", stops: [
        { pos: 0, color: g.color, alpha: 0.38 }, { pos: 0.7, color: g.color, alpha: 0.26 }, { pos: 1, color: g.color, alpha: 0 },
      ] } }),
    });
  });
  return [veil, glow, streak, ...ghosts];
}

/** Moves every layer of flare `set` to a light at `L` (the ghosts follow along the axis). */
export function moveFlare(layers: Layer[], set: string, L: Point) {
  for (const l of layers) {
    if (l.flare?.set !== set) continue;
    if (l.type === "gradientFill") {
      const g = l.params as LayerParams["gradientFill"];
      const at = l.flare.role === "ghost" ? ghostAt(L, l.flare.k ?? 1) : L;
      g.x = at.x; g.y = at.y;
    }
    const sh = l.mask.kind === "shape" ? l.mask.shape : undefined;
    if (sh) { sh.x = L.x; sh.y = L.y; }
  }
}

/** The light a flare is at now (its glow's centre). */
export function flareLight(layers: Layer[], set: string): Point | undefined {
  const g = layers.find((l) => l.flare?.set === set && l.flare.role === "glow");
  if (!g) return undefined;
  const p = g.params as LayerParams["gradientFill"];
  return { x: p.x, y: p.y };
}
