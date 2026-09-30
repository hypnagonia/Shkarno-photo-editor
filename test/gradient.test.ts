import { test } from "node:test";
import assert from "node:assert/strict";
import {
  GRADIENT_GROUPS, GRADIENT_PRESETS, HARMONY_RULES, gradientAt, gradientCss, gradientFrom, harmonyPalette, hexToOklab, hexToOklch,
  oklchToHex, paletteFromColors,
} from "../src/layers/gradient.ts";

const Ls = (cs: string[]) => cs.map((c) => hexToOklab(c)[0]);
const rising = (cs: string[]) => Ls(cs).every((L, i, a) => i === 0 || L > a[i - 1]);

test("every preset runs strictly dark → light in OkLab lightness", () => {
  for (const p of GRADIENT_PRESETS) assert.ok(rising(p.colors), `${p.id}: ${Ls(p.colors).map((l) => l.toFixed(3)).join(" ")}`);
  const ids = GRADIENT_PRESETS.map((p) => p.id);
  assert.equal(new Set(ids).size, ids.length, "preset ids are unique");
  for (const g of GRADIENT_GROUPS) assert.ok(GRADIENT_PRESETS.some((p) => p.group === g), `group ${g} has palettes`);
});

test("OkLCH → hex keeps lightness and hue, reducing chroma to fit sRGB", () => {
  const hex = oklchToHex(0.9, 0.3, 30); // far outside sRGB
  const [L, C, h] = hexToOklch(hex);
  assert.ok(Math.abs(L - 0.9) < 0.01, `L ${L}`);
  assert.ok(C < 0.3 && C > 0.02, `C ${C}`);
  assert.ok(Math.abs(((h - 30 + 540) % 360) - 180) < 6, `h ${h}`);
});

test("harmonies: 3–5 distinct colours, strictly dark → light, deterministic, for every rule and seed", () => {
  for (const base of ["#1D6A73", "#D9894A", "#7A2C6E", "#808080", "#FFD700", "#0000FF"]) {
    for (const rule of HARMONY_RULES) {
      for (let seed = 0; seed < 25; seed++) {
        for (const n of [3, 4, 5]) {
          const q = harmonyPalette(base, rule, seed, n);
          assert.equal(q.length, n, `${base} ${rule} ${seed}: ${q.length} colours`);
          assert.ok(rising(q), `${base} ${rule} ${seed} n${n}: ${q.join(" ")}`);
        }
        const p = harmonyPalette(base, rule, seed);
        assert.ok(rising(p), `${base} ${rule} ${seed}: ${p.join(" ")}`);
        assert.ok(p.every((c) => /^#[0-9A-F]{6}$/.test(c)));
      }
      assert.deepEqual(harmonyPalette(base, rule, 3), harmonyPalette(base, rule, 3), "same seed, same palette");
      assert.equal(harmonyPalette(base, rule, 0).length, 5);
    }
  }
  assert.notDeepEqual(harmonyPalette("#1D6A73", "triadic", 1), harmonyPalette("#1D6A73", "triadic", 2), "seeds vary");
});

test("smooth (OkLab) interpolation: same ends, lightness evenly between", () => {
  const classic = gradientFrom(["#0000FF", "#FFFF00"]);
  const smooth = gradientFrom(["#0000FF", "#FFFF00"], "oklab");
  assert.deepEqual(gradientAt(smooth, 0).slice(0, 3).map((v) => Math.round(v * 255)), [0, 0, 255]);
  assert.deepEqual(gradientAt(smooth, 1).slice(0, 3).map((v) => Math.round(v * 255)), [255, 255, 0]);
  const hex = (c: number[]) => "#" + c.slice(0, 3).map((v) => Math.round(v * 255).toString(16).padStart(2, "0")).join("");
  const Lmid = hexToOklab(hex(gradientAt(smooth, 0.5)))[0];
  const [L0, L1] = [hexToOklab("#0000FF")[0], hexToOklab("#FFFF00")[0]];
  assert.ok(Math.abs(Lmid - (L0 + L1) / 2) < 0.02, `OkLab middle L ${Lmid}`);
  // Classic sRGB blue → yellow passes through grey: much less chroma in the middle.
  const chroma = (c: number[]) => hexToOklch(hex(c))[1];
  assert.ok(chroma(gradientAt(smooth, 0.5)) > chroma(gradientAt(classic, 0.5)));
  assert.equal(gradientCss(smooth).split("rgba(").length - 1, 33, "CSS draws the OkLab path in steps");
});

test("palette from colours: deduplicated, dark → light, full range", () => {
  const p = paletteFromColors(["#6B8E23", "#6C8F24", "#87CEEB", "#F5DEB3", "#2F4F4F", "#8B4513", "#D2B48C", "#556B2F"]);
  assert.ok(p.length >= 2 && p.length <= 5);
  assert.ok(rising(p), p.join(" "));
  const L = Ls(p);
  assert.ok(L[0] < 0.25 && L[L.length - 1] > 0.85, `range ${L[0]}…${L[L.length - 1]}`);
  assert.ok(rising(paletteFromColors(["#777777"])), "a single colour still makes a dark → light run");
});

test("palettes are colours, not a ramp to black and white", () => {
  // Every colour of a harmony (other than mono / grey bases) carries real chroma, ends included.
  for (const rule of HARMONY_RULES.filter((r) => r !== "mono")) {
    const p = harmonyPalette("#3A7BD5", rule, 0, 5);
    for (const c of p) assert.ok(hexToOklch(c)[1] > 0.05, `${rule}: ${c} is almost grey`);
    // Neighbours are different colours (hue or lightness clearly apart).
    for (let i = 1; i < p.length; i++) {
      const a = hexToOklab(p[i - 1]), b = hexToOklab(p[i]);
      assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) > 0.08, `${rule}: ${p[i - 1]} ≈ ${p[i]}`);
    }
  }
  // The Palettes group: 3–5 colours each.
  for (const pr of GRADIENT_PRESETS.filter((q) => q.group === "palettes")) assert.ok(pr.colors.length >= 3 && pr.colors.length <= 5, pr.id);
});

test("a palette over a fading gradient keeps the fade", async () => {
  const { withAlphaOf, gradientFrom, gradientAt, recolor } = await import("../src/layers/gradient.ts");
  const ghost = { space: "oklab" as const, stops: [{ pos: 0, color: "#7FD6FF", alpha: 0.38 }, { pos: 0.7, color: "#7FD6FF", alpha: 0.26 }, { pos: 1, color: "#7FD6FF", alpha: 0 }] };
  const g = withAlphaOf(gradientFrom(["#FF0000", "#00FF00", "#0000FF"], "oklab"), ghost);
  for (const x of [0, 0.3, 0.7, 0.85, 1]) assert.ok(Math.abs(gradientAt(g, x)[3] - gradientAt(ghost, x)[3]) < 0.002, `alpha at ${x}`);
  assert.equal(gradientAt(g, 1)[3], 0);
  assert.equal(g.stops.find((s) => s.pos === 0)?.color, "#FF0000");
  // Opaque before: the palette as it is.
  const opaque = gradientFrom(["#000000", "#FFFFFF"]);
  assert.equal(withAlphaOf(gradientFrom(["#FF0000", "#0000FF"]), opaque).stops.length, 2);
  // One colour: every stop, opacities kept.
  const r = recolor(ghost, "#FFAA00");
  assert.deepEqual(r.stops.map((s) => [s.color, s.alpha]), [["#FFAA00", 0.38], ["#FFAA00", 0.26], ["#FFAA00", 0]]);
});
