/**
 * The frame: quarter turns, a mirror, straightening and the crop — one affine map from
 * the finished picture back to the photo. The preview applies it on the GPU
 * (render_frame.wgsl), the export on the CPU after the photo is rendered (resampleFrame),
 * and the page maps taps and overlays through it (toSource / toView).
 *
 * In order, from the photo:
 *   1. mirror (left ↔ right), then `quarter` turns clockwise — the "oriented" photo;
 *   2. turned clockwise by `angle` degrees about its centre, on a canvas of the same size;
 *   3. the crop: a rectangle of that canvas, 0…1.
 */
export interface Frame {
  /** Clockwise quarter turns, 0…3. */
  quarter: number;
  /** Mirrored left ↔ right (before the turns). */
  flip: boolean;
  /** Straightening, degrees clockwise, −45 … 45. */
  angle: number;
  /** The crop of the oriented, straightened photo: x, y, width, height (0…1). */
  crop: [number, number, number, number];
}

/** A 2 × 3 affine map: (u, v) → (a·u + b·v + c, d·u + e·v + f). */
export type Affine = [number, number, number, number, number, number];

export const FULL_CROP: [number, number, number, number] = [0, 0, 1, 1];

export function isIdentityFrame(f: Frame | undefined): boolean {
  if (!f) return true;
  const c = f.crop;
  return (f.quarter & 3) === 0 && !f.flip && Math.abs(f.angle) < 1e-4 && c[0] <= 1e-6 && c[1] <= 1e-6 && c[2] >= 1 - 1e-6 && c[3] >= 1 - 1e-6;
}

/** Size of the oriented photo (turned a quarter: its sides swap). */
export function orientedSize(f: Frame, W: number, H: number): [number, number] {
  return (f.quarter & 1) ? [H, W] : [W, H];
}

/** Size of the finished picture for a photo of W × H px. */
export function frameSize(f: Frame, W: number, H: number): [number, number] {
  const [ow, oh] = orientedSize(f, W, H);
  return [Math.max(1, Math.round(f.crop[2] * ow)), Math.max(1, Math.round(f.crop[3] * oh))];
}

/**
 * The map from the finished picture (0…1) to the photo (0…1) for a photo of W × H
 * (any scale: only its proportions count).
 */
export function frameToSource(f: Frame, W: number, H: number): Affine {
  const [ow, oh] = orientedSize(f, W, H);
  const [cx, cy, cw, ch] = f.crop;
  // Output uv → oriented, straightened canvas px: Q = (ow·(cx + u·cw), oh·(cy + v·ch)).
  // Un-straighten about the centre: S = C + R(−θ)(Q − C), R(φ) clockwise on screen (y down).
  const th = (f.angle * Math.PI) / 180, co = Math.cos(th), si = Math.sin(th);
  // R(−θ) = [[co, si], [−si, co]] (y down: a clockwise turn by θ is [[co, −si], [si, co]]).
  const Cx = ow / 2, Cy = oh / 2;
  // Q(u, v) = (q0 + qu·u, q1 + qv·v)
  const q0 = ow * cx - Cx, qu = ow * cw, q1 = oh * cy - Cy, qv = oh * ch;
  // S = C + [[co, si], [−si, co]]·(Q − C)
  let S: Affine = [co * qu, si * qv, Cx + co * q0 + si * q1, -si * qu, co * qv, Cy - si * q0 + co * q1];
  // Oriented px → source px (undo the turns), then unmirror; then to 0…1.
  const k = f.quarter & 3;
  // (x, y) = (a·X + b·Y + ox, d·X + e·Y + oy), applied to S.
  const lin = (a: number, b: number, d: number, e: number, ox: number, oy: number): Affine =>
    [a * S[0] + b * S[3], a * S[1] + b * S[4], a * S[2] + b * S[5] + ox, d * S[0] + e * S[3], d * S[1] + e * S[4], d * S[2] + e * S[5] + oy];
  if (k === 1) S = lin(0, 1, -1, 0, 0, H);        // x = Y, y = H − X
  else if (k === 2) S = lin(-1, 0, 0, -1, W, H);  // x = W − X, y = H − Y
  else if (k === 3) S = lin(0, -1, 1, 0, W, 0);   // x = W − Y, y = X
  if (f.flip) S = [-S[0], -S[1], W - S[2], S[3], S[4], S[5]];
  return [S[0] / W, S[1] / W, S[2] / W, S[3] / H, S[4] / H, S[5] / H];
}

export function applyAffine(A: Affine, u: number, v: number): [number, number] {
  return [A[0] * u + A[1] * v + A[2], A[3] * u + A[4] * v + A[5]];
}

export function invertAffine(A: Affine): Affine {
  const det = A[0] * A[4] - A[1] * A[3] || 1e-12;
  const a = A[4] / det, b = -A[1] / det, d = -A[3] / det, e = A[0] / det;
  return [a, b, -(a * A[2] + b * A[5]), d, e, -(d * A[2] + e * A[5])];
}

/** The crop's corners all inside the turned photo (straightening leaves empty corners outside it). */
export function cropInside(f: Frame, W: number, H: number, crop = f.crop): boolean {
  const A = frameToSource({ ...f, crop }, W, H);
  const eps = 1e-4;
  return [[0, 0], [1, 0], [0, 1], [1, 1]].every(([u, v]) => {
    const [x, y] = applyAffine(A, u, v);
    return x >= -eps && y >= -eps && x <= 1 + eps && y <= 1 + eps;
  });
}

/** The crop shrunk about its centre (its proportions kept) until it lies inside the turned photo, and within 0…1. */
export function fitCrop(f: Frame, W: number, H: number): [number, number, number, number] {
  let [x, y, w, h] = f.crop;
  w = Math.min(1, Math.max(0.02, w)); h = Math.min(1, Math.max(0.02, h));
  x = Math.min(1 - w, Math.max(0, x)); y = Math.min(1 - h, Math.max(0, y));
  const at = (s: number): [number, number, number, number] => [x + (w * (1 - s)) / 2, y + (h * (1 - s)) / 2, w * s, h * s];
  if (cropInside(f, W, H, at(1))) return at(1);
  let lo = 0, hi = 1;
  for (let i = 0; i < 30; i++) { const m = (lo + hi) / 2; if (cropInside(f, W, H, at(m))) lo = m; else hi = m; }
  return at(lo);
}

/** The largest crop of proportions `aspect` (width / height in px, of the oriented photo) centred where the crop is now. */
export function cropOfAspect(f: Frame, W: number, H: number, aspect: number): [number, number, number, number] {
  const [ow, oh] = orientedSize(f, W, H);
  // In 0…1 units: w·ow / (h·oh) = aspect.
  let w = 1, h = (ow / oh) / aspect;
  if (h > 1) { w = 1 / h; h = 1; }
  const cx = f.crop[0] + f.crop[2] / 2, cy = f.crop[1] + f.crop[3] / 2;
  return fitCrop({ ...f, crop: [cx - w / 2, cy - h / 2, w, h] }, W, H);
}

type Pixels = Uint8ClampedArray<ArrayBuffer> | Float32Array<ArrayBuffer>;

/**
 * The finished picture from a rendered photo (W × H, 4 channels): bilinear, `outW × outH`.
 * Outside the photo: transparent black (a crop fitted by fitCrop never reaches there).
 */
export function resampleFrame<T extends Pixels>(src: T, W: number, H: number, A: Affine, outW: number, outH: number): T {
  const out = new (src.constructor as { new (n: number): T })(outW * outH * 4);
  for (let j = 0; j < outH; j++) {
    const v = (j + 0.5) / outH;
    for (let i = 0; i < outW; i++) {
      const u = (i + 0.5) / outW;
      const sx = (A[0] * u + A[1] * v + A[2]) * W - 0.5, sy = (A[3] * u + A[4] * v + A[5]) * H - 0.5;
      const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
      if (x0 < -1 || y0 < -1 || x0 >= W || y0 >= H) continue;
      const xa = Math.max(0, x0), xb = Math.min(W - 1, x0 + 1), ya = Math.max(0, y0), yb = Math.min(H - 1, y0 + 1);
      const p00 = (ya * W + xa) * 4, p10 = (ya * W + xb) * 4, p01 = (yb * W + xa) * 4, p11 = (yb * W + xb) * 4;
      const o = (j * outW + i) * 4;
      for (let c = 0; c < 4; c++) {
        const top = src[p00 + c] + (src[p10 + c] - src[p00 + c]) * fx;
        const bot = src[p01 + c] + (src[p11 + c] - src[p01 + c]) * fx;
        out[o + c] = top + (bot - top) * fy;
      }
    }
  }
  return out;
}
