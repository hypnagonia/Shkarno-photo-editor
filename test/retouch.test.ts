import { test } from "node:test";
import assert from "node:assert/strict";
import { featherMask, prefixKeys, resize, strokeMask, strokeRect } from "../src/retouch/geometry.ts";

const W = 4000, H = 3000;

test("the region holds the stroke with context, square, inside the frame", () => {
  const s = { pts: [[0.5, 0.5], [0.55, 0.52]] as Array<[number, number]>, r: 0.01 };
  const r = strokeRect(s, W, H);
  assert.equal(r.w, r.h);
  assert.ok(r.x <= 0.5 * W - 40 && r.x + r.w >= 0.55 * W + 40);
  assert.ok(r.y <= 0.5 * H - 40 && r.y + r.h >= 0.52 * H + 40);
  // At a corner: clamped, still inside.
  const c = strokeRect({ pts: [[0.001, 0.999]], r: 0.02 }, W, H);
  assert.ok(c.x === 0 && c.y + c.h === H && c.w <= W && c.h <= H);
});

test("the mask covers the brush, not beyond it", () => {
  const s = { pts: [[0.5, 0.5]] as Array<[number, number]>, r: 0.01 };
  const rect = { x: 1800, y: 1300, w: 400, h: 400 };
  const m = strokeMask(s, W, H, rect, 400, 400);
  // Radius 40 px around (200, 200).
  assert.equal(m[200 * 400 + 200], 1);
  assert.equal(m[200 * 400 + 238], 1);
  assert.equal(m[200 * 400 + 245], 0);
  // Grown by 5: now reaches 244.
  assert.equal(strokeMask(s, W, H, rect, 400, 400, 5)[200 * 400 + 244], 1);
  // Scaled to 512: the same brush, in the new pixels.
  const m2 = strokeMask(s, W, H, rect, 512, 512);
  assert.equal(m2[256 * 512 + 256], 1);
  assert.equal(m2[256 * 512 + 256 + 55], 0);
});

test("the feather is 1 inside and fades to 0", () => {
  const m = new Uint8Array(64 * 64);
  for (let y = 24; y < 40; y++) for (let x = 24; x < 40; x++) m[y * 64 + x] = 1;
  const a = featherMask(m, 64, 64, 6);
  assert.equal(a[32 * 64 + 32], 1);
  assert.ok(a[32 * 64 + 42] > 0 && a[32 * 64 + 42] < 1);
  assert.equal(a[2 * 64 + 2], 0);
});

test("resize keeps a flat image flat and the stroke keys are prefixes", () => {
  const src = new Float32Array(10 * 10 * 3).fill(0.4);
  assert.ok(resize(src, 10, 10, 3, 17, 23).every((v) => Math.abs(v - 0.4) < 1e-6));
  const a = { pts: [[0.1, 0.1]] as Array<[number, number]>, r: 0.01 }, b = { pts: [[0.2, 0.2]] as Array<[number, number]>, r: 0.02 };
  const k = prefixKeys([a, b]);
  assert.ok(k[1].startsWith(k[0]));
  assert.equal(prefixKeys([a])[0], k[0]);
});
