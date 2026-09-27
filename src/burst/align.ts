/**
 * Burst alignment, coarse part (CPU, on the quarter-size "scan" of every frame):
 * a Gaussian-free box pyramid and HDR+-style tile matching from coarse to fine,
 * then a global motion model fitted to the tile field. The GPU refines the field
 * at full resolution (burst.wgsl, `align`), starting from this.
 *
 * Everything here is plain TS on luma planes, so tests run it in node.
 */

/** A single-channel image. */
export interface Plane { w: number; h: number; d: Float32Array }
/** Tile motion field: offset (dx, dy) of each tile of the reference into the frame, in pixels of its level. */
export interface Flow { tw: number; th: number; tile: number; dx: Float32Array; dy: Float32Array; err: Float32Array }
/** x' = a·x − b·y + tx, y' = b·x + a·y + ty (about the image centre, in scan pixels): shift, rotation, scale. */
export interface Similarity { a: number; b: number; tx: number; ty: number }
/**
 * Where a pixel of one image is in another (possibly of another size, another lens):
 * q − c_dst = [a −b; b a]·(p − c_src) + t, with c = each image's centre.
 */
export type Affine = Similarity;

/** The same mapping in other units: the source measured ×ks, the destination ×kd. */
export const rescale = (m: Affine, ks: number, kd: number): Affine => ({ a: (m.a * kd) / ks, b: (m.b * kd) / ks, tx: m.tx * kd, ty: m.ty * kd });
/** Scale of a mapping (destination pixels per source pixel). */
export const scaleOf = (m: Affine) => Math.hypot(m.a, m.b);

/**
 * `src` resampled into a w×h image: out(p) = src(m(p)). Pixels that fall outside
 * `src` are −1 (not covered: another lens sees less). When `src` is sampled
 * sparsely (a longer lens), 3×3 samples are averaged instead of one.
 */
export function resamplePlane(src: Plane, w: number, h: number, m: Affine): Plane {
  const d = new Float32Array(w * h);
  const sc = scaleOf(m), k = sc > 1.5 ? 1 : 0, step = sc / 3;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const u = x + 0.5 - w / 2, v = y + 0.5 - h / 2;
    const X = src.w / 2 + m.a * u - m.b * v + m.tx, Y = src.h / 2 + m.b * u + m.a * v + m.ty;
    if (X < 0 || Y < 0 || X > src.w || Y > src.h) { d[y * w + x] = -1; continue; }
    let s = 0, n = 0;
    for (let j = -k; j <= k; j++) for (let i = -k; i <= k; i++) { s += bilinear(src, X - 0.5 + i * step, Y - 0.5 + j * step); n++; }
    d[y * w + x] = s / n;
  }
  return { w, h, d };
}

/** Tiles every pixel of which is covered (no −1 from resamplePlane). */
export function coveredTiles(p: Plane, f: Flow): Uint8Array {
  const out = new Uint8Array(f.tw * f.th);
  for (let i = 0; i < out.length; i++) {
    const [cx, cy] = centre(f, i, p.w, p.h);
    let ok = 1;
    for (let y = cy - TILE / 2; y < cy + TILE / 2 && ok; y += 3) for (let x = cx - TILE / 2; x < cx + TILE / 2; x += 3) if (at(p, x, y) < 0) { ok = 0; break; }
    out[i] = ok;
  }
  return out;
}

export const TILE = 16;

/** Half-size box downsample. */
export function half(p: Plane): Plane {
  const w = Math.max(1, p.w >> 1), h = Math.max(1, p.h >> 1), d = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const o = 2 * y * p.w + 2 * x, x1 = Math.min(1, p.w - 1 - 2 * x), y1 = Math.min(1, p.h - 1 - 2 * y) * p.w;
    d[y * w + x] = 0.25 * (p.d[o] + p.d[o + x1] + p.d[o + y1] + p.d[o + y1 + x1]);
  }
  return { w, h, d };
}

/** Pyramid, finest first; stops before a level smaller than 3 tiles. */
export function pyramid(p: Plane, levels = 4): Plane[] {
  const out = [p];
  while (out.length < levels) {
    const l = out[out.length - 1];
    if (l.w < TILE * 8 || l.h < TILE * 8) break;
    out.push(half(l));
  }
  return out;
}

const at = (p: Plane, x: number, y: number) => p.d[Math.min(p.h - 1, Math.max(0, y)) * p.w + Math.min(p.w - 1, Math.max(0, x))];

/** Sum of squared (or absolute) differences of a reference tile against the frame at an integer offset. */
function cost(ref: Plane, frm: Plane, x0: number, y0: number, dx: number, dy: number, l1: boolean): number {
  let s = 0;
  const inside = x0 + dx >= 0 && y0 + dy >= 0 && x0 + dx + TILE <= frm.w && y0 + dy + TILE <= frm.h && x0 + TILE <= ref.w && y0 + TILE <= ref.h;
  for (let y = 0; y < TILE; y++) {
    if (inside) {
      const r = (y0 + y) * ref.w + x0, f = (y0 + y + dy) * frm.w + x0 + dx;
      for (let x = 0; x < TILE; x++) { const e = ref.d[r + x] - frm.d[f + x]; s += l1 ? Math.abs(e) : e * e; }
    } else for (let x = 0; x < TILE; x++) { const e = at(ref, x0 + x, y0 + y) - at(frm, x0 + x + dx, y0 + y + dy); s += l1 ? Math.abs(e) : e * e; }
  }
  return s;
}

/**
 * Tile flow from `ref` to `frm` (same size), coarse to fine: each level searches
 * ±radius around the best of the upsampled guesses of its own and its two nearest
 * coarse tiles; the finest level ends with a parabola fit for sub-pixel offsets.
 */
export function alignTiles(ref: Plane[], frm: Plane[], radius = 6): Flow {
  let prev: Flow | undefined;
  for (let L = ref.length - 1; L >= 0; L--) {
    const R = ref[L], F = frm[L];
    const tw = Math.ceil(R.w / TILE), th = Math.ceil(R.h / TILE);
    const f: Flow = { tw, th, tile: TILE, dx: new Float32Array(tw * th), dy: new Float32Array(tw * th), err: new Float32Array(tw * th) };
    const finest = L === 0, r = L === ref.length - 1 ? radius : 2;
    for (let ty = 0; ty < th; ty++) for (let tx = 0; tx < tw; tx++) {
      const x0 = Math.min(tx * TILE, Math.max(0, R.w - TILE)), y0 = Math.min(ty * TILE, Math.max(0, R.h - TILE));
      // Candidates from the coarser level (its tile and the nearest neighbours), ×2.
      const cands: Array<[number, number]> = [[0, 0]];
      if (prev) {
        const cx = Math.min(prev.tw - 1, tx >> 1), cy = Math.min(prev.th - 1, ty >> 1);
        const nx = Math.min(prev.tw - 1, Math.max(0, cx + (tx & 1 ? 1 : -1))), ny = Math.min(prev.th - 1, Math.max(0, cy + (ty & 1 ? 1 : -1)));
        for (const [ix, iy] of [[cx, cy], [nx, cy], [cx, ny]]) cands.push([Math.round(prev.dx[iy * prev.tw + ix] * 2), Math.round(prev.dy[iy * prev.tw + ix] * 2)]);
      }
      let bx = 0, by = 0, best = Infinity;
      for (const [gx, gy] of cands) { const c = cost(R, F, x0, y0, gx, gy, false); if (c < best) { best = c; bx = gx; by = gy; } }
      const cx0 = bx, cy0 = by;
      for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
        const c = cost(R, F, x0, y0, cx0 + dx, cy0 + dy, false);
        if (c < best) { best = c; bx = cx0 + dx; by = cy0 + dy; }
      }
      let sx = bx, sy = by;
      if (finest) { const [ex, ey] = lkStep(R, F, x0, y0, bx, by); sx += ex; sy += ey; }
      const i = ty * tw + tx;
      f.dx[i] = sx; f.dy[i] = sy; f.err[i] = best / (TILE * TILE);
    }
    prev = f;
  }
  return prev!;
}

/**
 * Sub-pixel part: one Lucas–Kanade step from the best integer offset (the cost
 * parabola is biased toward whole pixels). Clamped to ±0.5 px.
 */
export function lkStep(R: Plane, F: Plane, x0: number, y0: number, bx: number, by: number): [number, number] {
  let a = 0, b = 0, c = 0, p = 0, q = 0;
  for (let y = 0; y < TILE; y++) for (let x = 0; x < TILE; x++) {
    const fx = x0 + x + bx, fy = y0 + y + by;
    const gx = (at(F, fx + 1, fy) - at(F, fx - 1, fy)) / 2, gy = (at(F, fx, fy + 1) - at(F, fx, fy - 1)) / 2;
    const e = at(F, fx, fy) - at(R, x0 + x, y0 + y);
    a += gx * gx; b += gx * gy; c += gy * gy; p += gx * e; q += gy * e;
  }
  const det = a * c - b * b;
  if (det <= 1e-12) return [0, 0];
  const cl = (v: number) => Math.max(-0.5, Math.min(0.5, v));
  return [cl(-(c * p - b * q) / det), cl(-(a * q - b * p) / det)];
}

/** Tile centre in its level's pixels. */
const centre = (f: Flow, i: number, w: number, h: number): [number, number] => {
  const tx = i % f.tw, ty = Math.floor(i / f.tw);
  return [Math.min(tx * f.tile, Math.max(0, w - f.tile)) + f.tile / 2, Math.min(ty * f.tile, Math.max(0, h - f.tile)) + f.tile / 2];
};

/**
 * Global similarity fitted to the tile field by iteratively reweighted least
 * squares (moving things and flat tiles are voted out). Coordinates relative to
 * the image centre; `w`, `h` = the field's level size.
 */
export function fitGlobal(f: Flow, w: number, h: number, textured?: Uint8Array): Similarity {
  const n = f.tw * f.th;
  const wt = new Float32Array(n).fill(1);
  if (textured) for (let i = 0; i < n; i++) if (!textured[i]) wt[i] = 0;
  let m: Similarity = { a: 1, b: 0, tx: 0, ty: 0 };
  for (let it = 0; it < 6; it++) {
    // Least squares for (a, b, tx, ty) with x' − x = (a−1)x − b y + tx … : normal equations.
    const A = new Float64Array(16), B = new Float64Array(4);
    for (let i = 0; i < n; i++) {
      if (!wt[i]) continue;
      const [cx, cy] = centre(f, i, w, h);
      const x = cx - w / 2, y = cy - h / 2, u = x + f.dx[i], v = y + f.dy[i];
      // rows: [x, −y, 1, 0]·p = u ; [y, x, 0, 1]·p = v
      const rows: Array<[number[], number]> = [[[x, -y, 1, 0], u], [[y, x, 0, 1], v]];
      for (const [r, t] of rows) for (let j = 0; j < 4; j++) { B[j] += wt[i] * r[j] * t; for (let k = 0; k < 4; k++) A[j * 4 + k] += wt[i] * r[j] * r[k]; }
    }
    const p = solve4(A, B);
    if (!p) break;
    m = { a: p[0], b: p[1], tx: p[2], ty: p[3] };
    // Reweight: Cauchy on the residual (1 px scale).
    const res: number[] = [];
    for (let i = 0; i < n; i++) {
      const [cx, cy] = centre(f, i, w, h);
      const x = cx - w / 2, y = cy - h / 2;
      const e = Math.hypot(m.a * x - m.b * y + m.tx - x - f.dx[i], m.b * x + m.a * y + m.ty - y - f.dy[i]);
      res.push(e);
      if (!textured || textured[i]) wt[i] = 1 / (1 + (e / 0.75) ** 2);
    }
  }
  return m;
}

function solve4(A: Float64Array, b: Float64Array): number[] | undefined {
  const M = Array.from({ length: 4 }, (_, i) => [...A.slice(i * 4, i * 4 + 4), b[i]]);
  for (let c = 0; c < 4; c++) {
    let p = c;
    for (let r = c + 1; r < 4; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-9) return undefined;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < 4; r++) if (r !== c) { const k = M[r][c] / M[c][c]; for (let j = c; j < 5; j++) M[r][j] -= k * M[c][j]; }
  }
  return M.map((r, i) => r[4] / r[i]);
}

/** The model's offset at scan position (x, y). */
export function modelAt(m: Similarity, x: number, y: number, w: number, h: number): [number, number] {
  const u = x - w / 2, v = y - h / 2;
  return [m.a * u - m.b * v + m.tx - u, m.b * u + m.a * v + m.ty - v];
}

/** Tiles with enough texture to align on (standard deviation above `minStd`). */
export function texturedTiles(p: Plane, f: Flow, minStd: number): Uint8Array {
  const out = new Uint8Array(f.tw * f.th);
  for (let i = 0; i < out.length; i++) {
    const [cx, cy] = centre(f, i, p.w, p.h);
    let s = 0, s2 = 0;
    for (let y = cy - TILE / 2; y < cy + TILE / 2; y++) for (let x = cx - TILE / 2; x < cx + TILE / 2; x++) { const v = at(p, x, y); s += v; s2 += v * v; }
    const n = TILE * TILE, mean = s / n;
    out[i] = Math.sqrt(Math.max(0, s2 / n - mean * mean)) > minStd ? 1 : 0;
  }
  return out;
}

/**
 * Clamps each tile's offset to within `maxDev` pixels of the global model (a tile
 * that "aligned" onto something else is pulled back), and fills flat tiles with it.
 */
export function regularise(f: Flow, m: Similarity, w: number, h: number, textured: Uint8Array, maxDev: number, globalOnly = false): Flow {
  const out: Flow = { ...f, dx: f.dx.slice(), dy: f.dy.slice() };
  for (let i = 0; i < f.dx.length; i++) {
    const [cx, cy] = centre(f, i, w, h);
    const [gx, gy] = modelAt(m, cx, cy, w, h);
    if (globalOnly || !textured[i]) { out.dx[i] = gx; out.dy[i] = gy; continue; }
    const ex = f.dx[i] - gx, ey = f.dy[i] - gy, e = Math.hypot(ex, ey);
    if (e > maxDev) { out.dx[i] = gx + (ex * maxDev) / e; out.dy[i] = gy + (ey * maxDev) / e; }
  }
  return out;
}

/** Warps `frm` by the flow (bilinear between tile centres, bilinear sampling). */
export function warp(frm: Plane, f: Flow): Plane {
  const d = new Float32Array(frm.w * frm.h);
  for (let y = 0; y < frm.h; y++) for (let x = 0; x < frm.w; x++) {
    const [dx, dy] = flowAt(f, x + 0.5, y + 0.5, frm.w, frm.h);
    d[y * frm.w + x] = bilinear(frm, x + dx, y + dy);
  }
  return { w: frm.w, h: frm.h, d };
}

export function bilinear(p: Plane, x: number, y: number): number {
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  return (at(p, x0, y0) * (1 - fx) + at(p, x0 + 1, y0) * fx) * (1 - fy) + (at(p, x0, y0 + 1) * (1 - fx) + at(p, x0 + 1, y0 + 1) * fx) * fy;
}

/** Flow at a pixel position (pixel centres at +0.5), bilinear between tile centres. */
export function flowAt(f: Flow, px: number, py: number, w: number, h: number): [number, number] {
  // Tiles are laid on a regular grid (the last one clamped inside): use the regular centres.
  const gx = Math.min(f.tw - 1, Math.max(0, px / f.tile - 0.5)), gy = Math.min(f.th - 1, Math.max(0, py / f.tile - 0.5));
  void w; void h;
  const x0 = Math.floor(gx), y0 = Math.floor(gy), x1 = Math.min(f.tw - 1, x0 + 1), y1 = Math.min(f.th - 1, y0 + 1), fx = gx - x0, fy = gy - y0;
  const g = (a: Float32Array) => (a[y0 * f.tw + x0] * (1 - fx) + a[y0 * f.tw + x1] * fx) * (1 - fy) + (a[y1 * f.tw + x0] * (1 - fx) + a[y1 * f.tw + x1] * fx) * fy;
  return [g(f.dx), g(f.dy)];
}

/**
 * Exposure ratio of an aligned frame to the reference: the median of ref/frame over
 * mid-tones both see unclipped (bracketed series, or auto exposure drifting).
 */
export function exposureRatio(ref: Plane, aligned: Plane, clipAt = 0.9): number {
  const r: number[] = [];
  for (let i = 0; i < ref.d.length; i += 3) {
    const a = ref.d[i], b = aligned.d[i];
    if (a > 0.02 && b > 0.02 && a < clipAt && b < clipAt) r.push(a / b);
  }
  if (r.length < 50) return 1;
  r.sort((x, y) => x - y);
  return r[r.length >> 1];
}

/**
 * How a frame's brightness ratio to the reference changes from the centre out
 * (vignetting differs between lenses, and between a lens's own corrections):
 * ref / (g · frame) ≈ 1 + v1·r² + v2·r⁴, r = distance from the centre ÷ half the
 * diagonal. Fitted on mid-tones both see unclipped; `frame` aligned to `ref` (−1 = not covered).
 */
export function radialGain(ref: Plane, frame: Plane, g: number, clipAt = 0.9): [number, number] {
  const hd = Math.hypot(ref.w, ref.h) / 2;
  // Normal equations for (v1, v2) on q − 1 = v1 r² + v2 r⁴.
  let a11 = 0, a12 = 0, a22 = 0, b1 = 0, b2 = 0, n = 0, rMax = 0;
  for (let y = 1; y < ref.h; y += 3) for (let x = 1; x < ref.w; x += 3) {
    const i = y * ref.w + x, a = ref.d[i], b = frame.d[i] * g;
    if (!(frame.d[i] > 0.01 / g) || a < 0.01 || a > clipAt || b > clipAt) continue;
    const q = a / b - 1;
    if (Math.abs(q) > 0.6) continue; // not the same thing (moved, misaligned)
    const r2 = ((x + 0.5 - ref.w / 2) ** 2 + (y + 0.5 - ref.h / 2) ** 2) / (hd * hd), r4 = r2 * r2;
    a11 += r2 * r2; a12 += r2 * r4; a22 += r4 * r4; b1 += r2 * q; b2 += r4 * q; n++;
    if (r2 > rMax) rMax = r2;
  }
  // A frame that sees only the middle (a longer lens) tells nothing about the corners.
  if (rMax < 0.7 * 0.7) return [0, 0];
  const det = a11 * a22 - a12 * a12;
  if (n < 200 || Math.abs(det) < 1e-12) return [0, 0];
  const cl = (v: number) => Math.max(-0.8, Math.min(0.8, v));
  return [cl((b1 * a22 - b2 * a12) / det), cl((a11 * b2 - a12 * b1) / det)];
}

/** Laplacian energy over the central 60 %: which frame is sharpest (the reference). */
export function sharpness(p: Plane): number {
  let s = 0, n = 0;
  const x0 = Math.floor(p.w * 0.2), x1 = Math.ceil(p.w * 0.8), y0 = Math.floor(p.h * 0.2), y1 = Math.ceil(p.h * 0.8);
  for (let y = Math.max(1, y0); y < Math.min(p.h - 1, y1); y++) for (let x = Math.max(1, x0); x < Math.min(p.w - 1, x1); x++) {
    const i = y * p.w + x;
    const l = 4 * p.d[i] - p.d[i - 1] - p.d[i + 1] - p.d[i - p.w] - p.d[i + p.w];
    s += l * l; n++;
  }
  return n ? s / n : 0;
}

/**
 * Noise of the series from the scan: σ of (aligned frame − reference) per
 * brightness bin, robust (MAD) so motion does not count. `bins` over √Y in 0…1.
 * Returns σ of one frame's scan pixel (the difference ÷ √2).
 */
export function scanNoise(ref: Plane, aligned: Plane[], bins = 8): number[] {
  const acc: number[][] = Array.from({ length: bins }, () => []);
  for (const a of aligned) for (let i = 0; i < ref.d.length; i += 2) {
    const y = ref.d[i];
    if (y <= 0 || y >= 0.95 || a.d[i] < 0) continue; // (−1: not covered by that frame)
    const b = Math.min(bins - 1, Math.floor(Math.sqrt(y) * bins));
    acc[b].push(Math.abs(a.d[i] - y));
  }
  const out = acc.map((v) => { if (v.length < 30) return NaN; v.sort((x, y) => x - y); return (1.4826 * v[v.length >> 1]) / Math.SQRT2; });
  // Empty bins take the nearest known one.
  for (let i = 0; i < bins; i++) if (!Number.isFinite(out[i])) {
    let k = 1;
    while (k < bins && !Number.isFinite(out[i - k] ?? NaN) && !Number.isFinite(out[i + k] ?? NaN)) k++;
    out[i] = Number.isFinite(out[i - k] ?? NaN) ? out[i - k] : Number.isFinite(out[i + k] ?? NaN) ? out[i + k] : 0.002;
  }
  // Noise never falls with brightness (photon noise grows with it). A dark bin above a
  // brighter one is misalignment or parallax at dark edges, not noise: capped, or the
  // merge would take those differences for noise and blur them in.
  for (let i = bins - 2; i >= 0; i--) out[i] = Math.min(out[i], out[i + 1]);
  return out;
}

const sqrtPlane = (p: Plane, g: number): Plane => ({ w: p.w, h: p.h, d: p.d.map((v) => Math.sqrt(Math.max(0, v * g))) });
const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
const both = (a: Uint8Array, b: Uint8Array) => a.map((v, i) => v & b[i]);

/**
 * The field-of-view ratio of a frame to the reference when EXIF does not say which
 * lens (`sF`, as the ratio of 35 mm focal lengths). Scales from ¼× to 4× are tried
 * on half-size scans: each gets tile alignment and a global fit (which also corrects
 * a scale that is only roughly right), and is judged by how well the whole covered
 * view then matches — per-tile errors alone cannot tell, since every tile of smooth
 * content matches at some shift.
 */
export function searchScale(ref: Plane, frame: Plane, g: number, diagRatio: number): number {
  // `ref`: √-luma scan; searched at half its size (a half-size pixel spans 2 of the scan's).
  const refHalf = half(ref);
  const refPyr = pyramid(refHalf, 3);
  const grid = { tw: Math.ceil(refHalf.w / TILE), th: Math.ceil(refHalf.h / TILE), tile: TILE, dx: new Float32Array(0), dy: new Float32Array(0), err: new Float32Array(0) };
  const tex = texturedTiles(refHalf, grid, 0.01);
  let best = 1, bestErr = Infinity;
  for (let k = -8; k <= 8; k++) {
    const s0 = 2 * 2 ** (k / 4) * diagRatio;
    const base = resamplePlane(frame, refHalf.w, refHalf.h, { a: s0, b: 0, tx: 0, ty: 0 });
    const f = alignTiles(refPyr, pyramid(sqrtPlane(base, g), 3));
    const m = both(tex, coveredTiles(base, f));
    if (m.reduce((a, v) => a + v, 0) < m.length * 0.05) continue;
    const g1 = fitGlobal(f, refHalf.w, refHalf.h, m);
    const total: Affine = { a: s0 * g1.a, b: s0 * g1.b, tx: s0 * g1.tx, ty: s0 * g1.ty };
    const check = resamplePlane(frame, refHalf.w, refHalf.h, total);
    let e = 0, n = 0;
    for (let i = 0; i < check.d.length; i += 3) if (check.d[i] >= 0) { e += Math.abs(Math.sqrt(Math.max(0, check.d[i] * g)) - refHalf.d[i]); n++; }
    if (n < check.d.length / 3 * 0.05) continue;
    if (e / n < bestErr) { bestErr = e / n; best = scaleOf(total) / (2 * diagRatio); }
  }
  return best;
}
