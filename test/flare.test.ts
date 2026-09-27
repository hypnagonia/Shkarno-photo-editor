import { test } from "node:test";
import assert from "node:assert/strict";
import { flareLayers, flareLight, ghostAt, moveFlare } from "../src/layers/flare.ts";
import { packLayers } from "../src/layers/gpu.ts";
import type { LayerParams } from "../src/layers/model.ts";

const names = { veil: "Flare · veil", glow: "Flare · glow", streak: "Flare · streak", ghost: "Flare · ghost" };

test("ghosts sit on the line from the light through the centre", () => {
  const g = ghostAt({ x: 0.2, y: 0.3 }, 1); // the mirror point through the centre
  assert.ok(Math.abs(g.x - 0.8) < 1e-9 && Math.abs(g.y - 0.7) < 1e-9);
  const half = ghostAt({ x: 0.2, y: 0.3 }, 0.5);
  assert.ok(Math.abs(half.x - 0.65) < 1e-9 && Math.abs(half.y - 0.6) < 1e-9);
});

test("a flare is six ordinary layers of one set, glow and streak on the light", () => {
  const ls = flareLayers({ x: 0.3, y: 0.2 }, names);
  assert.equal(ls.length, 6);
  assert.equal(new Set(ls.map((l) => l.flare!.set)).size, 1);
  assert.deepEqual(ls.map((l) => l.flare!.role), ["veil", "glow", "streak", "ghost", "ghost", "ghost"]);
  const glow = ls[1].params as LayerParams["gradientFill"];
  assert.equal(glow.style, "radial"); assert.equal(ls[1].blend, "screen");
  assert.ok(glow.x === 0.3 && glow.y === 0.2);
  assert.equal(ls[2].mask.kind, "shape");
  assert.equal(packLayers(ls).count, 6, "all render");
});

test("moving the light moves the whole flare, ghosts along the axis", () => {
  const ls = flareLayers({ x: 0.3, y: 0.2 }, names);
  const set = ls[0].flare!.set;
  moveFlare(ls, set, { x: 0.7, y: 0.25 });
  assert.deepEqual(flareLight(ls, set), { x: 0.7, y: 0.25 });
  assert.ok(ls[0].mask.shape!.x === 0.7 && ls[2].mask.shape!.y === 0.25);
  const g = ls[3].params as LayerParams["gradientFill"];
  const want = ghostAt({ x: 0.7, y: 0.25 }, ls[3].flare!.k!);
  assert.ok(Math.abs(g.x - want.x) < 1e-9 && Math.abs(g.y - want.y) < 1e-9);
});
