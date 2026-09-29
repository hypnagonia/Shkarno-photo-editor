import { test } from "node:test";
import assert from "node:assert/strict";
import { findPreview } from "../src/decode/preview.ts";

/** A minimal JPEG header: SOI, an APP0 segment, then a frame header (SOFn). */
function jpeg(sof: number, w: number, h: number, comps: number): number[] {
  return [0xff, 0xd8, 0xff, 0xe0, 0x00, 0x04, 0x00, 0x00, 0xff, sof, 0x00, 0x11, 0x08, h >> 8, h & 255, w >> 8, w & 255, comps, 0, 0, 0, 0];
}

test("the camera's rendering: the largest colour picture JPEG, never the raw data or a matte", () => {
  const bytes = new Uint8Array([
    0, 1, 2,
    ...jpeg(0xc0, 4032, 3024, 3), 9, 9,
    ...jpeg(0xc3, 8064, 6048, 3), 9,   // lossless: the raw sensor data
    ...jpeg(0xc0, 8064, 6048, 1), 9,   // single channel: a semantic matte
    ...jpeg(0xc0, 160, 120, 3),        // a thumbnail
  ]);
  const p = findPreview(bytes);
  assert.ok(p);
  assert.deepEqual([p.width, p.height, p.components, p.offset], [4032, 3024, 3, 3]);
  assert.equal(findPreview(new Uint8Array(jpeg(0xc3, 4000, 3000, 3))), null);
});

import { shadowMatch } from "../src/decode/preview.ts";
test("black point from the camera: lifted shadows go deeper, crushed ones open, bounded, midtones kept", () => {
  const v = (a: number[]) => a.map((x) => x / 255);
  const deeper = shadowMatch(v([35, 48, 67, 78, 104, 135]), v([18, 32, 50, 63, 87, 139]))!;
  assert.ok(deeper[1].y < deeper[1].x, JSON.stringify(deeper));
  const open = shadowMatch(v([4, 7, 10, 15, 41, 86]), v([5, 13, 22, 33, 63, 94]))!;
  assert.ok(open.some((p) => p.y > p.x + 5 / 255), JSON.stringify(open));
  for (const c of [deeper, open]) {
    for (let i = 1; i < c.length; i++) assert.ok(c[i].y > c[i - 1].y && c[i].x > c[i - 1].x, "monotone");
    for (const p of c) assert.ok(Math.abs(p.y - p.x) <= 20 / 255 + 1e-3, "bounded");
    assert.deepEqual(c[c.length - 1], { x: 1, y: 1 });
  }
  assert.equal(shadowMatch(v([20, 30, 40, 50, 80, 120]), v([21, 30, 41, 50, 81, 120])), undefined, "already matching");
});

test("tone match: highlights short of the camera's are lifted toward it, the median stays", async () => {
  const { shadowMatch } = await import("../src/decode/preview.ts");
  const v = (a: number[]) => a.map((x) => x / 255);
  const pts = shadowMatch(v([10, 20, 30, 45, 80, 120]), v([10, 20, 30, 45, 80, 120]), v([150, 170, 185, 195]), v([165, 200, 235, 250]))!;
  assert.ok(pts, "a curve");
  const at = (x: number) => { for (let i = 1; i < pts.length; i++) if (x <= pts[i].x) { const a = pts[i - 1], b = pts[i]; return a.y + ((x - a.x) / (b.x - a.x)) * (b.y - a.y); } return 1; };
  assert.ok(Math.abs(at(120 / 255) - 120 / 255) < 1e-3, "median unchanged");
  assert.ok(at(185 / 255) > 185 / 255 + 10 / 255, `highlights lifted: ${Math.round(at(185 / 255) * 255)}`);
  for (let i = 1; i < pts.length; i++) assert.ok(pts[i].y >= pts[i - 1].y, "monotone");
});
