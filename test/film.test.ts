import { test } from "node:test";
import assert from "node:assert/strict";
import { filmOf, filmUniforms } from "../src/film/film.ts";
import { defaultParams } from "../src/decision/params.ts";

test("off, zero strength and absent render nothing", () => {
  assert.equal(filmUniforms({ character: "off", strength: 1, format: 36 }, 4032), undefined);
  assert.equal(filmUniforms({ character: "negative", strength: 0, format: 36 }, 4032), undefined);
  assert.equal(filmOf(defaultParams()), undefined);
});

test("a larger negative shows finer, weaker grain and tighter halation", () => {
  const small = filmUniforms({ character: "negative", strength: 1, format: 36 }, 4032)!;
  const large = filmUniforms({ character: "negative", strength: 1, format: 125 }, 4032)!;
  assert.ok(large.grain[1] < small.grain[1], "cloud size in pixels");
  assert.ok(large.grain[0] < small.grain[0], "grain RMS per pixel");
  assert.ok(large.halationR < small.halationR, "halation radius");
  assert.equal(large.bloomR, small.bloomR, "bloom is optical: the same share of the frame");
});

test("a higher-resolution photo shows the same grain in more, noisier pixels", () => {
  const lo = filmUniforms({ character: "cinema", strength: 1, format: 36 }, 4032)!;
  const hi = filmUniforms({ character: "cinema", strength: 1, format: 36 }, 8064)!;
  assert.ok(Math.abs(hi.grain[1] / lo.grain[1] - 2) < 1e-9, "twice the pixels across a cloud");
  assert.ok(hi.grain[0] >= lo.grain[0]);
});

test("the characters are ordered: clean < negative < cinema", () => {
  const u = (c: "clean" | "negative" | "cinema") => filmUniforms({ character: c, strength: 1, format: 36 }, 4032)!;
  assert.ok(u("clean").grain[0] < u("negative").grain[0] && u("negative").grain[0] < u("cinema").grain[0]);
  assert.ok(u("clean").halation < u("negative").halation && u("negative").halation < u("cinema").halation);
});
