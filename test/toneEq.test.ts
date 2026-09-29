import { test } from "node:test";
import assert from "node:assert/strict";
import { autoFitMask, compensate, histQuantile, LUT_HI, LUT_LO, LUT_N, neutralToneEq, TONE_EQ_PRESETS, toneEqActive, toneEqCurve, toneEqLut, ZONE_EV, type MaskHist } from "../src/tone/toneEq.ts";

test("tone EQ: neutral changes nothing", () => {
  const eq = neutralToneEq();
  assert.equal(toneEqActive(eq), false);
  assert.ok(toneEqLut(eq).every((v) => Math.abs(v) < 1e-9));
});

test("tone EQ: without smoothing the curve passes through every node", () => {
  const eq = { ...neutralToneEq(), gains: [1, 0.8, 0.5, 0.2, 0, -0.3, -0.6, -1, -1.4] };
  const f = toneEqCurve(eq);
  ZONE_EV.forEach((x, i) => assert.ok(Math.abs(f(x) - eq.gains[i]) < 0.02, `zone ${x}: ${f(x).toFixed(3)} want ${eq.gains[i]}`));
  // Beyond the outer zones it holds their values.
  assert.ok(Math.abs(f(-10) - 1) < 0.02 && Math.abs(f(2) + 1.4) < 0.02);
});

test("tone EQ: one node raised makes a smooth bump without overshoot", () => {
  const g = new Array(9).fill(0); g[4] = 1;
  const f = toneEqCurve({ ...neutralToneEq(), gains: g });
  let max = -Infinity, min = Infinity;
  for (let x = -8; x <= 0; x += 0.01) { max = Math.max(max, f(x)); min = Math.min(min, f(x)); }
  assert.ok(max <= 1.05 && min >= -0.08, `max ${max.toFixed(3)} min ${min.toFixed(3)}`);
  // Smoothing lowers and widens it.
  const s = toneEqCurve({ ...neutralToneEq(), gains: g, smoothing: 1 });
  assert.ok(s(-4) < f(-4) && s(-4) > 0.2);
});

test("tone EQ: the GPU table samples the curve", () => {
  const eq = { ...neutralToneEq(), gains: TONE_EQ_PRESETS[1].gains };
  const lut = toneEqLut(eq), f = toneEqCurve(eq);
  assert.equal(lut.length, LUT_N);
  const i = 30, x = LUT_LO + ((LUT_HI - LUT_LO) * i) / (LUT_N - 1);
  assert.ok(Math.abs(lut[i] - f(x)) < 1e-6);
});

test("tone EQ: auto-fit spreads the photo's tones over −7 … −1 EV", () => {
  // Tones between −6 and −2 EV (a flat photo).
  const h: MaskHist = { lo: -14, hi: 4, bins: new Array(180).fill(0) };
  for (let i = 0; i < 180; i++) { const ev = -14 + (i + 0.5) * 0.1; if (ev > -6 && ev < -2) h.bins[i] = 1; }
  const fit = autoFitMask(h);
  const lo = compensate(fit, histQuantile(h, 0.05)), hi = compensate(fit, histQuantile(h, 0.95));
  assert.ok(Math.abs(lo + 7) < 0.1 && Math.abs(hi + 1) < 0.1, `${lo.toFixed(2)} … ${hi.toFixed(2)}`);
});

test("tone EQ: presets stay within ±2 EV", () => {
  for (const p of TONE_EQ_PRESETS) assert.ok(p.gains.length === 9 && p.gains.every((g) => Math.abs(g) <= 2), p.id);
});
