/**
 * Magic brush geometry (pure): the region around a stroke that inpainting sees, the
 * stroke as a mask over that region, and the soft edge its fill is laid in with.
 */
import type { RetouchStroke } from "../decision/params.ts";

export interface Rect { x: number; y: number; w: number; h: number }

/**
 * The pixels to inpaint a stroke in (image W × H): its bounds, grown by context on every
 * side — the network fills the hole from what surrounds it, so the hole should be well
 * under half the region — square where the frame allows, clamped to the frame.
 */
export function strokeRect(s: RetouchStroke, W: number, H: number): Rect {
  const long = Math.max(W, H);
  const rad = s.r * long;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of s.pts) { x0 = Math.min(x0, x * W); y0 = Math.min(y0, y * H); x1 = Math.max(x1, x * W); y1 = Math.max(y1, y * H); }
  x0 -= rad; y0 -= rad; x1 += rad; y1 += rad;
  const ctx = Math.max(3 * rad, 0.06 * long, 48);
  // Square around the grown bounds where the frame allows (a network trained on squares).
  const side = Math.max(x1 - x0, y1 - y0) + 2 * ctx;
  const rw = Math.round(Math.min(W, side)), rh = Math.round(Math.min(H, side));
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const rx = Math.round(Math.min(W - rw, Math.max(0, cx - rw / 2)));
  const ry = Math.round(Math.min(H - rh, Math.max(0, cy - rh / 2)));
  return { x: rx, y: ry, w: rw, h: rh };
}

/**
 * The stroke as a mask over `rect` rasterised at ow × oh (1 inside: capsules of the
 * brush radius between successive points, grown by `grow` output pixels so the fill
 * reaches past the painted edge).
 */
export function strokeMask(s: RetouchStroke, W: number, H: number, rect: Rect, ow: number, oh: number, grow = 0): Uint8Array {
  const m = new Uint8Array(ow * oh);
  const sx = ow / rect.w, sy = oh / rect.h;
  const long = Math.max(W, H);
  // In output pixels (the rect may be stretched: an ellipse radius per axis).
  const pts = s.pts.map(([x, y]) => [(x * W - rect.x) * sx, (y * H - rect.y) * sy] as const);
  const rx = s.r * long * sx + grow, ry = s.r * long * sy + grow;
  for (let k = 0; k < pts.length; k++) {
    const [ax, ay] = pts[k], [bx, by] = pts[Math.min(k + 1, pts.length - 1)];
    const minX = Math.max(0, Math.floor(Math.min(ax, bx) - rx)), maxX = Math.min(ow - 1, Math.ceil(Math.max(ax, bx) + rx));
    const minY = Math.max(0, Math.floor(Math.min(ay, by) - ry)), maxY = Math.min(oh - 1, Math.ceil(Math.max(ay, by) + ry));
    // Distance to the segment in the circle's own units.
    const dx = (bx - ax) / rx, dy = (by - ay) / ry, len2 = dx * dx + dy * dy;
    for (let y = minY; y <= maxY; y++) for (let x = minX; x <= maxX; x++) {
      const px = (x + 0.5 - ax) / rx, py = (y + 0.5 - ay) / ry;
      const t = len2 > 0 ? Math.min(1, Math.max(0, (px * dx + py * dy) / len2)) : 0;
      const qx = px - t * dx, qy = py - t * dy;
      if (qx * qx + qy * qy <= 1) m[y * ow + x] = 1;
    }
  }
  return m;
}

/**
 * A soft edge for laying the fill in (0…1 per pixel): 1 on the mask, falling to 0 over
 * `feather` pixels outside it (a box-blurred mask, twice, then made 1 inside).
 */
export function featherMask(m: Uint8Array, w: number, h: number, feather: number): Float32Array {
  let a: Float32Array = Float32Array.from(m);
  const r = Math.max(1, Math.round(feather / 2));
  for (let pass = 0; pass < 2; pass++) a = boxBlur(a, w, h, r);
  for (let i = 0; i < a.length; i++) a[i] = m[i] ? 1 : Math.min(1, a[i] * 2);
  return a;
}

function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
  const n = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    let acc = 0;
    for (let x = -r; x <= r; x++) acc += src[y * w + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = acc / n;
      acc += src[y * w + Math.min(w - 1, x + r + 1)] - src[y * w + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = -r; y <= r; y++) acc += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = acc / n;
      acc += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
    }
  }
  return out;
}

/** Bilinear resize of an interleaved image (c channels). */
export function resize(src: Float32Array, w: number, h: number, c: number, ow: number, oh: number): Float32Array {
  const out = new Float32Array(ow * oh * c);
  for (let y = 0; y < oh; y++) {
    const fy = Math.min(h - 1, Math.max(0, ((y + 0.5) * h) / oh - 0.5));
    const y0 = Math.floor(fy), y1 = Math.min(h - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < ow; x++) {
      const fx = Math.min(w - 1, Math.max(0, ((x + 0.5) * w) / ow - 0.5));
      const x0 = Math.floor(fx), x1 = Math.min(w - 1, x0 + 1), tx = fx - x0;
      for (let k = 0; k < c; k++) {
        const a = src[(y0 * w + x0) * c + k], b = src[(y0 * w + x1) * c + k];
        const d = src[(y1 * w + x0) * c + k], e = src[(y1 * w + x1) * c + k];
        out[(y * ow + x) * c + k] = (a + (b - a) * tx) * (1 - ty) + (d + (e - d) * tx) * ty;
      }
    }
  }
  return out;
}

/** The keys of a stroke list's prefixes: stroke i's fill stays valid while strokes 0…i are unchanged. */
export function prefixKeys(strokes: RetouchStroke[]): string[] {
  const out: string[] = [];
  let acc = "";
  for (const s of strokes) { acc += JSON.stringify(s) + ";"; out.push(acc); }
  return out;
}
