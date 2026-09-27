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

test("exposure is judged for the scene: a dark photo is right at night, wrong in daylight", async () => {
  const { sceneKind } = await import("../src/analysis/check.ts");
  assert.equal(sceneKind(3.5, 0.3, 0), "night");
  assert.equal(sceneKind(8, 0.5, 0), "indoor");
  assert.equal(sceneKind(11, 0.5, 0.3), "overcast");
  assert.equal(sceneKind(15, 0.7, 0.2), "bright");
  assert.equal(sceneKind(undefined, 0.2, 0), "night"); // no settings: the camera's rendering decides
  // A dark photo (median ≈ 0.3 L): natural for night, too dark for daylight.
  const dark = scene((x) => [-45 + x * 0 , -45, -45]);
  const night = byId(checkPhoto({ final: dark, before: dark, scene: { ev: 4 } }));
  const day = byId(checkPhoto({ final: dark, before: dark, scene: { ev: 13 } }));
  assert.equal(night.exposure.level, "ok", JSON.stringify(night.exposure.v));
  assert.notEqual(day.exposure.level, "ok", JSON.stringify(day.exposure.v));
  assert.equal(day.exposure.v.dir, "dark");
});

test("saturation: pale is colour the edit lost, loud is loud — never a ratio to a pale RAW development", () => {
  const colourful = scene((x, y) => [x % 32 < 16 ? 60 : -40, y % 24 < 12 ? 40 : -30, x % 20 < 10 ? -50 : 50]);
  const faded = scene((x, y) => [x % 32 < 16 ? 12 : -8, y % 24 < 12 ? 8 : -6, x % 20 < 10 ? -10 : 10]);
  // Faded against the camera's colourful rendering: pale, even though our plain development was paler still.
  const r = byId(checkPhoto({ final: faded, before: scene(), camera: colourful }));
  assert.equal(r.saturation.v.issue, "pale", JSON.stringify(r.saturation.v));
  // A grey scene that stays grey is not pale.
  const g = byId(checkPhoto({ final: scene(), before: scene() }));
  assert.equal(g.saturation.level, "ok", JSON.stringify(g.saturation.v));
});

test("with the camera's own rendering, exposure is judged against it", () => {
  const dark = scene(() => [-45, -45, -45]);
  const bright = scene();
  // The camera itself rendered this scene dark: a dark edit matches it (even in daylight).
  const same = byId(checkPhoto({ final: dark, before: dark, camera: dark, scene: { ev: 13 } }));
  assert.equal(same.exposure.level, "ok", JSON.stringify(same.exposure.v));
  assert.equal(same.exposure.v.basis, "camera");
  // Much darker than the camera's rendering: says so, and which way.
  const off = byId(checkPhoto({ final: dark, before: dark, camera: bright, scene: { ev: 13 } }));
  assert.notEqual(off.exposure.level, "ok");
  assert.equal(off.exposure.v.dir, "darker");
});
