/** The Check: findings on the finished photo against the camera's rendering, and their fixes. */
import type { Engine } from "./engine.ts";
import { embeddedPreviewPixels } from "../decode/preview.ts";
import type { Params } from "../decision/params.ts";
import { GROUPS } from "../neural/scene.ts";
import { liveLayers } from "../layers/gpu.ts";
import type { CheckItem, CheckInput } from "../analysis/check.ts";
import type { FixChange } from "../analysis/checkFix.ts";
import { checkCode, sceneEV, type Session, isMobile, StaleCheck } from "./session.ts";

/**
 * Check: the edit as exported and the camera's rendering, both at 1024 px, looked
 * at for technical mistakes (src/analysis/check.ts). Returns the findings and the
 * edit's pixels (to show where a finding is).
 */
export async function check(eng: Engine) : Promise<{ items: CheckItem[]; rgba: Uint8Array; w: number; h: number }> {
  const s = eng.s;
  if (!s) throw new Error("No photo open");
  await eng.ensureSelections(s, s.params);
  const t = await eng.ensureThumb(1024);
  const src = { base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width };
  const read = async (p: Params, dof: boolean) => {
    const r = await eng.renderer.render(src, s.maps, p, { wb: eng.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false }, dof);
    return new Uint8Array(await eng.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4));
  };
  // Judged without the film: its grain and glow are a chosen look, not noise or haze.
  const p = s.params.film ? { ...s.params, film: { ...s.params.film, character: "off" as const } } : s.params;
  const final = await read(p, p.enable.dof && p.dof.strength > 0);
  const before = await read(eng.cameraParams(), false);
  const seg = { ...s.scene.seg, groups: GROUPS };
  const [{ checkPhoto }] = await checkCode();
  // The camera's own rendering (Apple's JPEG inside a ProRAW DNG): what "camera" means for
  // the whole-photo figures. Our plain development of the RAW is far paler than what the
  // phone shows, and would make any edit look "twice as colourful".
  if (s.cameraRef === undefined) s.cameraRef = /\.dng$/i.test(s.name) ? (await embeddedPreviewPixels(s.file, 256, isMobile() ? 24 : Infinity)) ?? null : null;
  // A JPEG / HEIC is itself the camera's rendering (and "before" shows it unchanged).
  const camera = s.cameraRef ?? (s.work.referred === "display" ? { rgba: before, w: t.w, h: t.h } : undefined);
  const items = checkPhoto({ final: { rgba: final, w: t.w, h: t.h }, before: { rgba: before, w: t.w, h: t.h }, camera, seg, scene: { ev: sceneEV(s.decoded.meta) } });
  // The fixes are worked out next, as their own job (solveCheckFixes): the findings show at once.
  eng.lastCheck = { s, items: items.map((i) => ({ ...i, mask: undefined })), seg, scene: { ev: sceneEV(s.decoded.meta) }, camera };
  return { items, rgba: final, w: t.w, h: t.h };
}

/** The fixes for the last check's findings, each reported as soon as it is worked out. */
export async function solveCheckFixes(eng: Engine, onFix: (id: CheckItem["id"], fix: FixChange[], partial: boolean) => void) {
  const c = eng.lastCheck;
  eng.lastCheck = undefined;
  if (!c || c.s !== eng.s) return; // another photo since
  try {
    await eng.solveFixes(c.s, c.items, c.seg, c.scene, c.camera, (it) => onFix(it.id, it.fix!, !!it.fixPartial));
  } catch (e) {
    if (!(e instanceof StaleCheck)) throw e; // edited or another photo: the answers would be for old settings
  }
}

/**
 * For every finding that is not fine: which control, set to what, fixes it. Each
 * finding measures how far it is from a comfortable result (CheckItem.err, > 0 =
 * needs fixing); one control at a time (in the order leversFor gives) is bisected,
 * on renders at 768 px, to where that crosses zero — a comfortable value, not the
 * edge of "fine". If no control gets there, the layer whose opacity does; failing
 * that, the control that helps most (a partial fix).
 *
 * Each render is its own short exclusive job, so slider edits and previews are
 * served in between; an edit (new params) or another photo stops the solving.
 */
export async function solveFixes(eng: Engine, s: Session, items: CheckItem[], seg: NonNullable<CheckInput["seg"]>, scene: CheckInput["scene"], camera: CheckInput["camera"], onFix: (it: CheckItem) => void) {
  const todo = items.filter((i) => i.level !== "ok" && i.err !== undefined);
  if (!todo.length) return;
  const [{ checkPhoto, planesOf }, { getPath, leversFor, setPath, stepOf }] = await checkCode();
  const t0 = performance.now();
  let renders = 0, size = "";
  const base = s.params;
  const render = (p: Params) => eng.exclusive(async () => {
    if (eng.s !== s || s.params !== base) throw new StaleCheck();
    renders++;
    const t = await eng.ensureThumb(768);
    size = `${t.w}×${t.h}`;
    const src = { base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width };
    const r = await eng.renderer.render(src, s.maps, p, { wb: eng.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false }, p.enable.dof && p.dof.strength > 0);
    return { rgba: new Uint8Array(await eng.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4)), w: t.w, h: t.h };
  });
  const beforeImg = await render(eng.cameraParams());
  const beforePlanes = planesOf(beforeImg);
  const errWith = async (id: CheckItem["id"], p: Params) =>
    checkPhoto({ final: await render(p), before: beforeImg, seg, scene, camera }, id, beforePlanes).find((i) => i.id === id)?.err ?? -1;
  for (const it of todo) {
    const e0 = await errWith(it.id, base);
    if (e0 <= 0) continue; // comfortable at this size already
    let best: { change: FixChange; err: number } | undefined;
    for (const lv of leversFor(it, base)) {
      const v0 = getPath(base, lv.path);
      const step = stepOf(lv.path);
      const at = (v: number) => { const p = structuredClone(base); setPath(p, lv.path, v); return errWith(it.id, p); };
      // Walk from the current value towards the end in 6 steps to the first comfortable
      // value (a two-sided target like exposure is passed, not just approached), then
      // bisect between it and the step before.
      let prev = v0, prevE = e0, before2 = v0, found: number | undefined, minE = e0, minV = v0;
      for (let k = 1; k <= 6; k++) {
        const v = v0 + ((lv.bound - v0) * k) / 6;
        const e = await at(v);
        if (e < minE) { minE = e; minV = v; }
        if (e <= 0) { found = v; break; }
        if (e > prevE && k > 1) {
          // Worse again after getting better: a narrow comfortable window may lie between
          // (stepped over). Look for the best value in (before2, v) by golden-section search.
          let a = before2, b = v;
          for (let g = 0; g < 5 && Math.abs(b - a) > step; g++) {
            const m1 = b - (b - a) * 0.618, m2 = a + (b - a) * 0.618;
            const [e1, e2] = [await at(m1), await at(m2)];
            if (e1 < minE) { minE = e1; minV = m1; }
            if (e2 < minE) { minE = e2; minV = m2; }
            if (Math.min(e1, e2) <= 0) break;
            if (e1 < e2) b = m2; else a = m1;
          }
          if (minE <= 0) { found = minV; prev = before2; }
          break;
        }
        before2 = prev; prev = v; prevE = e;
      }
      if (found === undefined) {
        // Not all the way: remember the value that helps most.
        if (minE < e0 * 0.6 && (!best || minE < best.err)) best = { change: { path: lv.path, from: v0, to: Math.round((Math.round(minV / step) * step) * 1000) / 1000 }, err: minE };
        continue;
      }
      let lo = prev, hi = found;
      for (let k = 0; k < 5 && Math.abs(hi - lo) > step; k++) { const mid = (lo + hi) / 2; if ((await at(mid)) <= 0) hi = mid; else lo = mid; }
      let to = Math.round(hi / step) * step;
      if (Math.sign(to - hi) === Math.sign(v0 - lv.bound)) to -= Math.sign(v0 - lv.bound) * step; // round towards the fixing side
      it.fix = [{ path: lv.path, from: v0, to: Math.round(to * 1000) / 1000 }];
      break;
    }
    if (it.fix) { onFix(it); continue; }
    // No control does it: the layer that does (its opacity; 0 = hide it).
    const live = liveLayers(base.layers ?? [], base.autoCurves ?? 1, base.enable);
    for (const L of [...live].reverse()) {
      const at = (o: number) => { const p = structuredClone(base); p.layers!.find((x) => x.id === L.id)!.opacity = o; return errWith(it.id, p); };
      if ((await at(0)) > 0) continue;
      let lo = L.opacity, hi = 0;
      for (let k = 0; k < 6; k++) { const mid = (lo + hi) / 2; if ((await at(mid)) <= 0) hi = mid; else lo = mid; }
      it.fix = [{ layer: L.id, name: L.name, from: L.opacity, to: Math.floor(hi * 20) / 20 }];
      break;
    }
    if (!it.fix && best) { it.fix = [best.change]; it.fixPartial = true; }
    if (it.fix) onFix(it);
  }
  eng.log(`check fixes: ${todo.length} findings, ${renders} renders at ${size}, ${Math.round(performance.now() - t0)} ms`);
}
