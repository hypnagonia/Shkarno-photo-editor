import { test } from "node:test";
import assert from "node:assert/strict";
import { checkPhoto, type CheckImage } from "../src/analysis/check.ts";

const W = 128, H = 96;
/** A soft scene: a horizontal ramp with a darker band and a small bright spot (not clipped). */
function scene(f: (x: number, y: number) => [number, number, number] = () => [0, 0, 0]): CheckImage {
  const rgba = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const base = 40 + (x / (W - 1)) * 170 - (y > 60 ? 30 : 0);
    const d = f(x, y);
    const i = (y * W + x) * 4;
    rgba[i] = Math.max(0, Math.min(255, base + 6 + d[0]));
    rgba[i + 1] = Math.max(0, Math.min(255, base + d[1]));
    rgba[i + 2] = Math.max(0, Math.min(255, base - 4 + d[2]));
    rgba[i + 3] = 255;
  }
  return { rgba, w: W, h: H };
}
const byId = (items: ReturnType<typeof checkPhoto>) => Object.fromEntries(items.map((i) => [i.id, i]));

test("an unedited photo passes every check", () => {
  const img = scene();
  const r = byId(checkPhoto({ final: img, before: img }));
  for (const id of ["highlights", "shadows", "colorClip", "saturation", "cast", "noise", "halos", "banding", "vignette"]) assert.equal(r[id].level, "ok", `${id}: ${JSON.stringify(r[id].v)}`);
});

test("highlights blown by the edit are found, with where", () => {
  const before = scene();
  const final = scene((x) => (x > 90 ? [120, 120, 120] : [0, 0, 0]));
  const r = byId(checkPhoto({ final, before }));
  assert.equal(r.highlights.level, "bad");
  assert.ok(r.highlights.mask && r.highlights.mask[40 * W + 120] === 1 && r.highlights.mask[40 * W + 10] === 0);
});

test("a colour cast the edit added is found and named", () => {
  const before = scene();
  const final = scene(() => [-12, 6, 22]); // bluer
  const r = byId(checkPhoto({ final, before }));
  assert.notEqual(r.cast.level, "ok");
  assert.ok(["blue", "cyan"].includes(String(r.cast.v.tint)), String(r.cast.v.tint));
});

test("noise amplified by the edit is found", () => {
  const before = scene();
  let s = 7;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5) * 2;
  const final = scene(() => { const v = rnd() * 14; return [v, v, v]; });
  const r = byId(checkPhoto({ final, before }));
  assert.notEqual(r.noise.level, "ok", JSON.stringify(r.noise.v));
});

test("banding (steps in a smooth gradient) is found", () => {
  const before = scene();
  const final = scene(); // posterize the final to steps of 8 levels
  for (let i = 0; i < final.rgba.length; i++) if (i % 4 !== 3) final.rgba[i] = Math.round(final.rgba[i] / 8) * 8;
  const r = byId(checkPhoto({ final, before }));
  assert.notEqual(r.banding.level, "ok", JSON.stringify(r.banding.v));
});
