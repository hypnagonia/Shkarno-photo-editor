import { test } from "node:test";
import assert from "node:assert/strict";
import { ALL_PARTS, CAMERA_LOOKS, applyCameraLook, lensBlurStrength } from "../src/looks/cameras.ts";
import { defaultParams } from "../src/decision/params.ts";

test("lens blur follows the optics: bigger sensor, longer lens, wider aperture = more blur", () => {
  const mf80 = lensBlurStrength(80, 1.9, "mf"), ff50 = lensBlurStrength(50, 1.8, "ff"), apsc23 = lensBlurStrength(23, 2, "apsc");
  assert.ok(mf80 > ff50 && ff50 > apsc23, `${mf80} ${ff50} ${apsc23}`);
  assert.ok(lensBlurStrength(85, 1.2, "ff") > lensBlurStrength(85, 2.8, "ff"));
  // Full frame 50 mm f/1.8 at 2 m: a background disc ≈ 2 % of the frame width → ≈ 0.45.
  assert.ok(ff50 > 0.35 && ff50 < 0.55, String(ff50));
});

test("ten looks, unique, each a real profile", () => {
  assert.equal(CAMERA_LOOKS.length, 10);
  assert.equal(new Set(CAMERA_LOOKS.map((l) => l.id)).size, 10);
  for (const l of CAMERA_LOOKS) assert.ok(l.profile.id.startsWith("camera-") && l.lens.focal > 0);
});

test("switching looks never stacks, and none restores the automatic rendering", () => {
  const auto = defaultParams();
  const p = structuredClone(auto);
  applyCameraLook(p, auto, CAMERA_LOOKS[0], ALL_PARTS, false);
  assert.equal(p.camera?.id, CAMERA_LOOKS[0].id);
  assert.ok(p.enable.dof && p.dof.strength > 0 && p.grain.amount > 0);
  applyCameraLook(p, auto, CAMERA_LOOKS[9], { colour: true, lens: false, sensor: false }, false);
  assert.equal(p.grain.amount, auto.grain.amount, "sensor part off: grain back to auto");
  assert.equal(p.enable.dof, auto.enable.dof, "lens part off: no blur");
  assert.equal(p.profile.id, CAMERA_LOOKS[9].profile.id);
  applyCameraLook(p, auto, undefined, ALL_PARTS, false);
  assert.equal(p.camera, undefined);
  assert.deepEqual(p.profile, auto.profile);
  // No depth map: no lens blur, the rest still applies.
  const q = structuredClone(auto);
  applyCameraLook(q, auto, CAMERA_LOOKS[1], ALL_PARTS, true);
  assert.equal(q.enable.dof, auto.enable.dof);
  assert.notEqual(q.vignette.amount, auto.vignette.amount);
});
