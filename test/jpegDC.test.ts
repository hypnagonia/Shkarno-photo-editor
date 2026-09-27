import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { jpegDC } from "../src/decode/jpegDC.ts";
import { findPreview } from "../src/decode/preview.ts";

test("a 48 MP camera preview at 1/8 size from its DC coefficients (when the sample is present)", { skip: !existsSync(".samples/IMG_1489.DNG") }, () => {
  const b = readFileSync(".samples/IMG_1489.DNG");
  const all = new Uint8Array(b.buffer, b.byteOffset, b.length);
  const p = findPreview(all.subarray(0, 24 << 20))!;
  const r = jpegDC(all.subarray(p.offset))!;
  assert.equal(r.w, Math.ceil(p.width / 8));
  assert.equal(r.h, Math.ceil(p.height / 8));
  // Means of this preview, from a full decode (ImageMagick): ≈ 0.272 / 0.292 / 0.331.
  const mean = [0, 1, 2].map((c) => { let s = 0; for (let i = c; i < r.rgba.length; i += 4) s += r.rgba[i]; return s / (r.w * r.h) / 255; });
  assert.ok(Math.abs(mean[0] - 0.272) < 0.01 && Math.abs(mean[1] - 0.292) < 0.01 && Math.abs(mean[2] - 0.331) < 0.01, mean.join(" "));
});

test("not a JPEG, or one it cannot read: undefined, never a throw", () => {
  assert.equal(jpegDC(new Uint8Array([1, 2, 3, 4])), undefined);
  assert.equal(jpegDC(new Uint8Array([0xff, 0xd8, 0xff, 0xc2, 0, 4, 0, 0])), undefined); // progressive
});
