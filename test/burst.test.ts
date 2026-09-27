import { test } from "node:test";
import assert from "node:assert/strict";
import { alignTiles, bilinear, exposureRatio, fitGlobal, flowAt, pyramid, regularise, scanNoise, sharpness, texturedTiles, warp, type Plane } from "../src/burst/align.ts";

/** Smooth random texture (value noise over several scales), continuous in (u, v): not periodic. */
const GRID = 64;
const lattice = (() => { let s = 7; const r = new Float32Array(GRID * GRID * 4); for (let i = 0; i < r.length; i++) { s = (s * 16807) % 2147483647; r[i] = s / 2147483647; } return r; })();
function valueNoise(u: number, v: number, o: number): number {
  const x0 = Math.floor(u), y0 = Math.floor(v), fx = u - x0, fy = v - y0;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const g = (x: number, y: number) => lattice[o + (((y % GRID) + GRID) % GRID) * GRID + (((x % GRID) + GRID) % GRID)];
  return (g(x0, y0) * (1 - sx) + g(x0 + 1, y0) * sx) * (1 - sy) + (g(x0, y0 + 1) * (1 - sx) + g(x0 + 1, y0 + 1) * sx) * sy;
}
const texture = (u: number, v: number) => 0.15 + 0.35 * valueNoise(u / 97 + 3.1, v / 97 + 1.7, 0) + 0.25 * valueNoise(u / 23, v / 23, GRID * GRID) + 0.15 * valueNoise(u / 6.3, v / 6.3, 2 * GRID * GRID);

/** A textured test scene sampled through an optional coordinate map, with optional noise. */
function scene(w: number, h: number, map: (x: number, y: number) => [number, number] = (x, y) => [x, y], noise = 0, seed = 1): Plane {
  let s = seed;
  const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const d = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const [u, v] = map(x, y);
    d[y * w + x] = Math.max(0, texture(u, v) + noise * gauss());
  }
  return { w, h, d };
}

test("burst align: a sub-pixel shift is found within 0.1 px", () => {
  const ref = scene(256, 192);
  const frm = scene(256, 192, (x, y) => [x - 3.3, y + 1.7]); // content moved by (+3.3, −1.7)
  const f = alignTiles(pyramid(ref), pyramid(frm));
  const [dx, dy] = flowAt(f, 128, 96, 256, 192);
  assert.ok(Math.abs(dx - 3.3) < 0.1 && Math.abs(dy + 1.7) < 0.1, `found (${dx.toFixed(2)}, ${dy.toFixed(2)})`);
  // Warped back, the frame matches the reference.
  const back = warp(frm, f);
  let e = 0, n = 0;
  for (let y = 20; y < 172; y++) for (let x = 20; x < 236; x++) { e += Math.abs(back.d[y * 256 + x] - ref.d[y * 256 + x]); n++; }
  assert.ok(e / n < 0.01, `mean error ${(e / n).toFixed(4)}`);
});

test("burst align: a large shift is found through the pyramid", () => {
  // The scan's size (≈1024 px): four levels, ±48 px reach.
  const ref = scene(1024, 768);
  const frm = scene(1024, 768, (x, y) => [x + 41, y - 23]);
  const f = alignTiles(pyramid(ref), pyramid(frm));
  const [dx, dy] = flowAt(f, 512, 384, 1024, 768);
  assert.ok(Math.abs(dx + 41) < 0.3 && Math.abs(dy - 23) < 0.3, `found (${dx.toFixed(2)}, ${dy.toFixed(2)})`);
});

test("burst align: the global model recovers a small rotation and ignores a moving patch", () => {
  const w = 384, h = 288, a = (0.5 * Math.PI) / 180;
  const ref = scene(w, h);
  const frm = scene(w, h, (x, y) => {
    const u = x - w / 2, v = y - h / 2;
    // frame(x) = ref(R⁻¹ x): content rotated by +a.
    return [Math.cos(a) * u + Math.sin(a) * v + w / 2, -Math.sin(a) * u + Math.cos(a) * v + h / 2];
  });
  // Something moving in one corner of the frame.
  for (let y = 20; y < 70; y++) for (let x = 20; x < 70; x++) frm.d[y * w + x] = 0.9;
  const pr = pyramid(ref);
  const f = alignTiles(pr, pyramid(frm));
  const tex = texturedTiles(pr[0], f, 0.01);
  const m = fitGlobal(f, w, h, tex);
  const rot = Math.atan2(m.b, m.a);
  assert.ok(Math.abs(rot - a) < 0.0015, `rotation ${(rot * 180 / Math.PI).toFixed(3)}°`);
  const r = regularise(f, m, w, h, tex, 1);
  const [cx, cy] = flowAt(r, 45, 45, w, h);
  assert.ok(Math.hypot(cx, cy) < 3, "the moving patch follows the camera's motion, not the object");
});

test("burst: exposure ratio, noise and sharpness from scans", () => {
  const ref = scene(200, 150, undefined, 0.01, 3);
  const dark = { ...ref, d: scene(200, 150, undefined, 0.01, 5).d.map((v) => v / 4) };
  const g = exposureRatio(ref, dark);
  assert.ok(Math.abs(g - 4) / 4 < 0.02, `ratio ${g.toFixed(3)}`);
  const other = scene(200, 150, undefined, 0.01, 9);
  const sig = scanNoise(ref, [other]);
  const mid = sig.filter((v) => v > 0).sort()[3];
  assert.ok(Math.abs(mid - 0.01) < 0.003, `σ ${mid.toFixed(4)}`);
  const blurred: Plane = { ...ref, d: ref.d.map((_, i) => { const x = i % 200, y = Math.floor(i / 200); return (bilinear(ref, x - 0.5, y) + bilinear(ref, x + 0.5, y) + bilinear(ref, x, y - 0.5) + bilinear(ref, x, y + 0.5)) / 4; }) };
  assert.ok(sharpness(ref) > sharpness(blurred), "the unblurred frame is sharper");
});
