import { test } from "node:test";
import assert from "node:assert/strict";
import { NEUTRAL_TONE, toneCurve } from "../src/render/curves.ts";

test("display-referred sources: neutral tone is the identity (no second rendering)", () => {
  const f = toneCurve({ ...NEUTRAL_TONE, displayReferred: true });
  for (let y = 0.001; y <= 0.9; y *= 1.3) assert.ok(Math.abs(f(y) - y) < 0.5 / 255 + y * 0.004, `y ${y.toFixed(4)} → ${f(y).toFixed(4)}`);
  // Above white (exposure raised it): rolled into 1, never beyond, monotone.
  let prev = f(0.9);
  for (let y = 0.92; y < 8; y *= 1.1) { const v = f(y); assert.ok(v >= prev - 1e-9 && v <= 1); prev = v; }
});

test("display-referred sources: contrast is a change around the identity", () => {
  const f = toneCurve({ ...NEUTRAL_TONE, contrast: 0.5, displayReferred: true });
  // Mid grey (display) stays close; shadows darker, highlights brighter.
  assert.ok(f(0.05) < 0.05 && f(0.7) > 0.7, `${f(0.05).toFixed(4)} ${f(0.7).toFixed(4)}`);
  // The scene curve is unchanged for RAW (0.18 → 0.29).
  assert.ok(Math.abs(toneCurve(NEUTRAL_TONE)(0.18) - 0.29) < 0.01);
});
