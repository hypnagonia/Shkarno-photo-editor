import { test } from "node:test";
import assert from "node:assert/strict";
import { applyAffine, cropInside, cropOfAspect, fitCrop, frameSize, frameToSource, invertAffine, isIdentityFrame, resampleFrame, type Frame } from "../src/geometry/frame.ts";

const F = (o: Partial<Frame> = {}): Frame => ({ quarter: 0, flip: false, angle: 0, crop: [0, 0, 1, 1], ...o });
const near = (a: number[], b: number[], eps = 1e-9) => a.forEach((x, i) => assert.ok(Math.abs(x - b[i]) < eps, `${a} vs ${b}`));

test("identity maps the picture onto the photo", () => {
  assert.ok(isIdentityFrame(F()));
  near(frameToSource(F(), 400, 300), [1, 0, 0, 0, 1, 0]);
});

test("crop: the finished picture is that part of the photo", () => {
  const A = frameToSource(F({ crop: [0.25, 0.5, 0.5, 0.5] }), 400, 300);
  near(applyAffine(A, 0, 0), [0.25, 0.5]);
  near(applyAffine(A, 1, 1), [0.75, 1]);
  assert.deepEqual(frameSize(F({ crop: [0.25, 0.5, 0.5, 0.5] }), 400, 300), [200, 150]);
});

test("a clockwise quarter turn: the picture's top-left is the photo's bottom-left", () => {
  const f = F({ quarter: 1 });
  const A = frameToSource(f, 400, 300);
  near(applyAffine(A, 0, 0), [0, 1]);
  near(applyAffine(A, 1, 0), [0, 0]);
  near(applyAffine(A, 1, 1), [1, 0]);
  assert.deepEqual(frameSize(f, 400, 300), [300, 400]);
});

test("half and three-quarter turns, and the mirror", () => {
  near(applyAffine(frameToSource(F({ quarter: 2 }), 400, 300), 0, 0), [1, 1]);
  near(applyAffine(frameToSource(F({ quarter: 3 }), 400, 300), 0, 0), [1, 0]);
  near(applyAffine(frameToSource(F({ flip: true }), 400, 300), 0, 0.5), [1, 0.5]);
  // Mirrored, then turned a quarter: the picture's top-left is the photo's bottom-right.
  near(applyAffine(frameToSource(F({ flip: true, quarter: 1 }), 400, 300), 0, 0), [1, 1]);
});

test("straightening keeps the centre and turns about it", () => {
  const A = frameToSource(F({ angle: 10 }), 400, 400);
  near(applyAffine(A, 0.5, 0.5), [0.5, 0.5]);
  // Content turned clockwise: the picture's point right of centre shows the photo above-right of it.
  const [x, y] = applyAffine(A, 0.75, 0.5);
  assert.ok(x < 0.75 && y < 0.5);
});

test("inverse, fitting and proportions", () => {
  const A = frameToSource(F({ angle: 7, quarter: 3, flip: true, crop: [0.1, 0.2, 0.6, 0.5] }), 640, 480);
  const B = invertAffine(A);
  near(applyAffine(B, ...applyAffine(A, 0.3, 0.8)), [0.3, 0.8]);
  const f = F({ angle: 12 });
  assert.ok(!cropInside(f, 400, 300));
  const c = fitCrop(f, 400, 300);
  assert.ok(cropInside({ ...f, crop: c }, 400, 300));
  assert.ok(Math.abs(c[2] / c[3] - 1) < 1e-9 && c[2] > 0.6);
  const sq = cropOfAspect(F(), 400, 300, 1);
  near([sq[2] * 400, sq[3] * 300], [300, 300], 1e-6);
});

test("resample: a crop copies the pixels", () => {
  const src = new Uint8ClampedArray(4 * 4 * 4).map((_, i) => i);
  const out = resampleFrame(src, 4, 4, frameToSource(F({ crop: [0.5, 0.5, 0.5, 0.5] }), 4, 4), 2, 2);
  assert.deepEqual([...out.subarray(0, 4)], [...src.subarray((2 * 4 + 2) * 4, (2 * 4 + 2) * 4 + 4)]);
});
