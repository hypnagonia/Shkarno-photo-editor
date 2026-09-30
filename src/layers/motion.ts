/**
 * Motion blur's geometry, shared by the page's arrows on the photo and (as numbers
 * written twice) layers.wgsl / render_motion.wgsl: how long a streak is at a distance,
 * and which way it runs.
 */
import type { LayerParams } from "./model.ts";

/** A streak at amount 1 is this share of the picture's long side. */
export const MOTION_STREAK = 0.06;
/** Into the depth, how the deep end streaks when the layer does not say: more than the near. */
export const DEFAULT_FALLOFF = 0.5;

/** Into the depth: the streak's factor at relative depth `rel` (0 near … 1 deep end of the part) and `rn` (distance from the vanishing point over the part's reach). */
export function depthFlow(rel: number, rn: number, falloff: number): number {
  const fo = Math.min(1, Math.max(-1, falloff));
  const sm = (x: number) => { const t = Math.min(1, Math.max(0, x / 0.15)); return t * t * (3 - 2 * t); };
  const mix = (a: number, b: number, t: number) => a + (b - a) * t;
  return fo < 0 ? mix(1, mix(1.4, 0.25, rel) * (0.15 + 0.85 * rn), -fo) : mix(1, mix(0.6, 1.6, rel), fo) * sm(rn);
}

/** Streak length by nearness (distance 0 = nearest … 1 = farthest), as seen from a moving camera. */
export function motionScale(dist: number, deep: boolean): number {
  const d = Math.min(1, Math.max(0, dist));
  return deep ? 1.4 + (0.25 - 1.4) * d : 1.5 + (0.25 - 1.5) * d;
}

export interface Arrow { x: number; y: number; dx: number; dy: number; len: number }

/**
 * Arrows on a cols × rows grid over the photo (0…1 coordinates), where the layer's mask
 * blurs: direction (unit, screen y down) and length (share of the long side) of the
 * streak there — into the depth, away from the layer's own vanishing point.
 */
export function motionArrows(b: LayerParams["blur"], depth: { w: number; h: number; data: number[]; mask?: number[]; vanish: [number, number] }, cols: number, rows: number, aspect: number): Arrow[] {
  const out: Arrow[] = [];
  const a = ((b.angle ?? 0) * Math.PI) / 180;
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    const x = (i + 0.5) / cols, y = (j + 0.5) / rows;
    const k = Math.min(depth.h - 1, Math.floor(y * depth.h)) * depth.w + Math.min(depth.w - 1, Math.floor(x * depth.w));
    // Only where the mask blurs (the layer's mask on the same grid).
    if (depth.mask && (depth.mask[k] ?? 0) < 0.25) continue;
    const d = depth.data[k] ?? 0.5;
    const vanish = b.vanish ?? depth.vanish;
    let dx = Math.cos(a), dy = -Math.sin(a);
    if (b.depth) {
      // Away from the vanishing point, measured in the picture's own proportions.
      const rx = (x - vanish[0]) * aspect, ry = y - vanish[1];
      const r = Math.hypot(rx, ry);
      if (r < 1e-3) continue;
      dx = rx / r; dy = ry / r;
    }
    // Into the depth: ∝ r / Z within the part (as layers.wgsl).
    let len = b.amount * MOTION_STREAK;
    if (b.depth) {
      const rg = b.range ?? [0, 1];
      const rel = Math.min(1, Math.max(0, (d - rg[0]) / Math.max(rg[1] - rg[0], 0.05)));
      const r = Math.hypot((x - vanish[0]) * aspect, y - vanish[1]);
      len *= depthFlow(rel, Math.min(1, r / Math.max(b.reach ?? 0.6, 0.05)), b.falloff ?? DEFAULT_FALLOFF);
    } else len *= motionScale(d, false);
    out.push({ x, y, dx, dy, len });
  }
  return out;
}
