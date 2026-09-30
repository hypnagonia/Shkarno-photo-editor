import { test } from "node:test";
import assert from "node:assert/strict";
import { chromaStats, renderedQuantiles, REF_QS, HI_QS } from "../src/decode/preview.ts";
import { srgbEotf as eotf, srgbOetf as oetf } from "../src/color/transfer.ts";
import { linSrgbToOklab } from "../src/color/oklab.ts";

/** A photo-like RGBA image: smooth gradients, a dark and a bright patch, noise. */
function image(w = 640, h = 480): Uint8Array {
  const px = new Uint8Array(w * h * 4);
  let s = 7;
  const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = (y * w + x) * 4;
    const base = x < w / 4 ? 8 : x > (3 * w) / 4 ? 235 : 40 + (180 * y) / h;
    px[i] = Math.min(255, Math.max(0, base + 30 * Math.sin(x / 37) + 12 * rnd()));
    px[i + 1] = Math.min(255, Math.max(0, base * 0.9 + 20 * rnd()));
    px[i + 2] = Math.min(255, Math.max(0, base * 0.7 + 40 * Math.cos(y / 23)));
    px[i + 3] = 255;
  }
  return px;
}

/** The exact computation the histogram replaced: every sample, sorted. */
function sortedQuantiles(px: Uint8Array, qs: number[], stride: number): number[] {
  const ys: number[] = [];
  for (let k = 0; k < px.length; k += 4 * stride) ys.push(oetf(0.229 * eotf(px[k] / 255) + 0.6917 * eotf(px[k + 1] / 255) + 0.0793 * eotf(px[k + 2] / 255)));
  ys.sort((a, b) => a - b);
  return qs.map((q) => ys[Math.min(ys.length - 1, Math.floor(q * ys.length))]);
}

test("luminance quantiles by histogram match sorting within 0.25/255", () => {
  const px = image();
  for (const qs of [REF_QS, HI_QS]) {
    const a = renderedQuantiles(px, qs), b = sortedQuantiles(px, qs, 7);
    a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < 0.25 / 255, `q${qs[i]}: ${v * 255} vs ${b[i] * 255}`));
  }
});

test("chroma p95 by histogram matches sorting within 5e-4", () => {
  const px = image();
  const cs: number[] = [];
  for (let k = 0; k < px.length; k += 28) {
    const r = eotf(px[k] / 255), g = eotf(px[k + 1] / 255), b = eotf(px[k + 2] / 255);
    const lab = linSrgbToOklab([1.2249401 * r - 0.2249404 * g, -0.0420569 * r + 1.0420571 * g, -0.0196376 * r - 0.0786361 * g + 1.0982735 * b]);
    if (lab[0] >= 0.25 && lab[0] <= 0.92) cs.push(Math.hypot(lab[1], lab[2]));
  }
  cs.sort((a, b) => a - b);
  const c = chromaStats(px);
  assert.ok(Math.abs(c.p95 - cs[Math.floor(cs.length * 0.95)]) < 5e-4);
  assert.ok(Math.abs(c.mean - cs.reduce((a, b) => a + b, 0) / cs.length) < 1e-6);
});
