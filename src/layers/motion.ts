/**
 * Motion blur's geometry, shared by the page's arrows on the photo and (as numbers
 * written twice) layers.wgsl / render_motion.wgsl: how long a streak is at a distance,
 * and which way it runs.
 */
import type { LayerParams } from "./model.ts";

/** A streak at amount 1 is this share of the picture's long side. */
export const MOTION_STREAK = 0.06;

/** Streak length by nearness (distance 0 = nearest … 1 = farthest), as seen from a moving camera. */
export function motionScale(dist: number, deep: boolean): number {
  const d = Math.min(1, Math.max(0, dist));
  return deep ? 1.6 + (0.1 - 1.6) * d : 1.5 + (0.25 - 1.5) * d;
}

export interface Arrow { x: number; y: number; dx: number; dy: number; len: number }

/**
 * Arrows on a cols × rows grid over the photo (0…1 coordinates): direction (unit, screen
 * y down) and length (share of the long side) of the streak there.
 */
export function motionArrows(b: LayerParams["blur"], depth: { w: number; h: number; data: number[]; vanish: [number, number] }, cols: number, rows: number, aspect: number): Arrow[] {
  const out: Arrow[] = [];
  const a = ((b.angle ?? 0) * Math.PI) / 180;
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    const x = (i + 0.5) / cols, y = (j + 0.5) / rows;
    const d = depth.data[Math.min(depth.h - 1, Math.floor(y * depth.h)) * depth.w + Math.min(depth.w - 1, Math.floor(x * depth.w))] ?? 0.5;
    let dx = Math.cos(a), dy = -Math.sin(a);
    if (b.depth) {
      // Away from the vanishing point, measured in the picture's own proportions.
      const rx = (x - depth.vanish[0]) * aspect, ry = y - depth.vanish[1];
      const r = Math.hypot(rx, ry);
      if (r < 1e-3) continue;
      dx = rx / r; dy = ry / r;
    }
    out.push({ x, y, dx, dy, len: b.amount * MOTION_STREAK * motionScale(d, !!b.depth) });
  }
  return out;
}
