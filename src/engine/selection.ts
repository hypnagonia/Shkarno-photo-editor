/** Taps on the photo and tap-to-select: what is under a tap, and the SAM masks layers use. */
import type { Engine } from "./engine.ts";
import { halvesToFloats } from "../gpu/half.ts";
import { downsample } from "../refine/refine.ts";
import type { Params, Region } from "../decision/params.ts";
import { GROUPS } from "../neural/scene.ts";
import type { PickInfo } from "./protocol.ts";
import { srgbEotf } from "../color/transfer.ts";
import { linSrgbToOklab } from "../color/oklab.ts";
import { SamSelector } from "../neural/sam.ts";
import { phoneForced } from "../device.ts";
import { levelsByArea, selectionMask } from "../refine/selection.ts";
import { selectKey, type MaskShape } from "../layers/model.ts";
import { liveLayers } from "../layers/gpu.ts";
import { mulVec } from "../color/mat3.ts";
import { P3_TO_SRGB, type Session, type Selections } from "./session.ts";

/**
 * What is under a tap, for a mask built from it: the region (network probabilities,
 * 3×3 around the tap), the distance and the tapped object's depth range (as focus
 * points use), and the colour before the layers (median of 5×5 at 384 px) in OkLab.
 */
export async function pickAt(eng: Engine, x: number, y: number, layer?: number, object = false) : Promise<PickInfo | undefined> {
  const s = eng.s;
  const f = eng.focusRangeAt(x, y);
  if (!s || !f) return undefined;
  const seg = s.scene.seg, plane = seg.width * seg.height;
  const cx = Math.round(x * (seg.width - 1)), cy = Math.round(y * (seg.height - 1));
  const score = new Float64Array(GROUPS.length);
  for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
    const k = Math.min(seg.height - 1, Math.max(0, cy + j)) * seg.width + Math.min(seg.width - 1, Math.max(0, cx + i));
    for (let g = 0; g < GROUPS.length; g++) score[g] += seg.probs[g * plane + k] / 9;
  }
  let best = 0;
  for (let g = 1; g < GROUPS.length; g++) if (score[g] > score[best]) best = g;
  let region: Region = GROUPS[best];
  // People: Apple's skin matte (ProRAW) tells skin from clothes.
  const skin = s.decoded.masks?.find((m) => m.kind === "skin");
  if (region === "person" && skin && skin.data[Math.min(skin.height - 1, Math.round(y * (skin.height - 1))) * skin.width + Math.min(skin.width - 1, Math.round(x * (skin.width - 1)))] > 127) region = "skin";
  // The colour before the layers (the layers' masks compare against exactly that).
  const t = await eng.ensureThumb(384);
  // (View 8: that colour itself — no look, sharpening or grain after it.)
  const p: Params = { ...s.params, enable: { ...s.params.enable, dof: false } };
  const r = await eng.renderer.render({ base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width }, s.maps, p, { wb: eng.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false, debugView: 8 }, false);
  const px = new Uint8Array(await eng.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4));
  const tx = Math.round(x * (t.w - 1)), ty = Math.round(y * (t.h - 1));
  const labs: Array<[number, number, number]> = [];
  for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++) {
    const k = (Math.min(t.h - 1, Math.max(0, ty + j)) * t.w + Math.min(t.w - 1, Math.max(0, tx + i))) * 4;
    const lin = mulVec(P3_TO_SRGB, [srgbEotf(px[k] / 255), srgbEotf(px[k + 1] / 255), srgbEotf(px[k + 2] / 255)]);
    labs.push(linSrgbToOklab(lin));
  }
  const med = (c: 0 | 1 | 2) => labs.map((l) => l[c]).sort((a, b) => a - b)[labs.length >> 1];
  // Surfaces running from near to far (ground, sky, a wall over a third of the frame) get
  // only a thin depth slice for focus; as a mask, "this object" is then the whole region.
  const range: [number, number] = f.range[1] - f.range[0] <= 0.0401 ? [0, 1] : f.range;
  const info: PickInfo = { x, y, region, prob: score[GROUPS.indexOf(region === "skin" ? "person" : region)], dist: f.dist, range, color: [med(0), med(1), med(2)] };
  if (layer !== undefined && layer >= 0) Object.assign(info, await eng.maskUnderTap(s, x, y, layer, object));
  return info;
}

/**
 * Is the tap on something the layer's mask already covers? Read from the mask
 * itself (view 7 at 384 px, median of 3×3). If so and the tap selects objects:
 * is the tapped object one of the layer's own selections (its mask overlaps the
 * new tap's by IoU > 0.5)? Then that selection is what a tap removes.
 */
export async function maskUnderTap(eng: Engine, s: Session, x: number, y: number, layer: number, object: boolean) : Promise<{ inMask: number; sameAs?: string }> {
  await eng.ensureSelections(s, s.params);
  const t = await eng.ensureThumb(384);
  const r = await eng.renderer.render({ base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width }, s.maps, s.params,
    { wb: eng.wbFor(s.params), gain: s.gain, lightLinear: s.lightLinear, output: "p38", dither: false, debugView: 7, region: layer }, false);
  const px = new Uint8Array(await eng.gpu.readTexture(r.tex, 0, 0, t.w, t.h, 4));
  const tx = Math.round(x * (t.w - 1)), ty = Math.round(y * (t.h - 1));
  const v: number[] = [];
  for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) v.push(px[(Math.min(t.h - 1, Math.max(0, ty + j)) * t.w + Math.min(t.w - 1, Math.max(0, tx + i))) * 4 + 1]);
  let inMask = v.sort((a, b) => a - b)[4] / 255;
  const st = s.sel;
  if (!object || !st) return { inMask };
  // An object tap is judged by the whole object, not the pixel under the finger (thin
  // things and soft mask edges read half-selected there): how much of the tapped
  // object the mask already covers.
  const { w: gw, h: gh } = s.maps;
  const tapObj = await eng.selectionMask(s, st, { kind: "select", points: [[x, y, 1]], invert: false, feather: 1 });
  let n = 0, covered = 0;
  for (let gy = 0; gy < gh; gy += 2) for (let gx = 0; gx < gw; gx += 2) {
    if (tapObj[gy * gw + gx] <= 127) continue;
    n++;
    const k = (Math.min(t.h - 1, Math.round((gy / (gh - 1)) * (t.h - 1))) * t.w + Math.min(t.w - 1, Math.round((gx / (gw - 1)) * (t.w - 1)))) * 4 + 1;
    covered += px[k] / 255;
  }
  if (n > 8) inMask = covered / n;
  if (inMask <= 0.5) return { inMask };
  // The layer's own added selections that cover the tap.
  const L = liveLayers(s.params.layers ?? [], s.params.autoCurves ?? 1, s.params.enable)[layer];
  const w = gw, h = gh;
  const at = (mask: Uint8Array) => mask[Math.min(h - 1, Math.round(y * (h - 1))) * w + Math.min(w - 1, Math.round(x * (w - 1)))];
  const cands = L ? [L.mask, ...(L.mask.parts ?? []).filter((q) => q.op === "add")].filter((m) => m.kind === "select" && !m.invert && st.cache.get(selectKey(m)) && at(st.cache.get(selectKey(m))!) > 127) : [];
  if (!cands.length) return { inMask };
  const tap = tapObj;
  let best: string | undefined, bestIou = 0.5;
  for (const m of cands) {
    const c = st.cache.get(selectKey(m))!;
    let inter = 0, uni = 0;
    for (let i = 0; i < c.length; i++) { const a = c[i] > 127, b = tap[i] > 127; if (a && b) inter++; if (a || b) uni++; }
    const iou = uni ? inter / uni : 0;
    if (iou > bestIou) { bestIou = iou; best = selectKey(m); }
  }
  return { inMask, sameAs: best };
}

/**
 * Makes sure every selection the layers use has its mask in the selection
 * texture (encoding the photo on first use: MobileSAM, a few seconds on a
 * phone), then points the renderer at it. Cheap when nothing changed.
 */
export async function ensureSelections(eng: Engine, s: Session, p: Params) {
  await eng.ensureRetouch(s, p); // (the photo itself first: masks are read from it)
  const want: MaskShape[] = [];
  for (const l of p.layers ?? []) for (const m of [l.mask, ...(l.mask.parts ?? [])]) if (m.kind === "select" && m.points?.length) want.push(m);
  const keys = [...new Set(want.map(selectKey))];
  const sel = s.sel;
  if (!keys.length && !sel?.keys.length) return;
  if (sel && keys.length === sel.keys.length && keys.every((k, i) => k === sel.keys[i])) return;
  const st: Selections = (s.sel ??= { sam: new SamSelector(eng.base, s.work.width, s.work.height, phoneForced()), cache: new Map(), failed: new Set(), keys: [], version: 0 });
  for (const m of want) {
    const k = selectKey(m);
    if (st.cache.has(k) || st.failed.has(k)) continue;
    try { st.cache.set(k, await eng.selectionMask(s, st, m)); }
    catch (e) { st.failed.add(k); eng.log(`selection unavailable: ${e instanceof Error ? e.message : e}`); }
  }
  // Keep the last 16 masks (≈ 0.4 MB each): undo and switching readings are instant.
  for (const k of [...st.cache.keys()]) if (st.cache.size > 16 && !keys.includes(k)) st.cache.delete(k);
  const ready = keys.filter((k) => st.cache.has(k));
  const { w, h } = s.maps;
  eng.gpu.release(st.tex);
  st.tex = eng.gpu.tex("selection", w, h, "r8unorm", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, "2d", Math.max(1, ready.length));
  ready.forEach((k, z) => eng.gpu.device.queue.writeTexture({ texture: st.tex!, origin: { x: 0, y: 0, z } }, st.cache.get(k)! as Uint8Array<ArrayBuffer>, { bytesPerRow: w, rowsPerImage: h }, { width: w, height: h }));
  st.keys = keys;
  st.version++;
  eng.renderer.selection = { tex: st.tex, slotOf: (m) => ready.indexOf(selectKey(m)), version: st.version };
}

/** One selection's mask at the guide resolution: SAM's reading for its taps, snapped to the photo's edges. */
export async function selectionMaskFor(eng: Engine, s: Session, st: Selections, m: MaskShape) : Promise<Uint8Array> {
  const { w, h } = s.maps;
  if (!st.sam.encoded) {
    eng.progress("selection", "encode");
    await st.sam.encode(async () => {
      // The photo as SAM sees it: display-encoded, long side 1024, HWC 0…255.
      const [pw, ph] = st.sam.dims;
      const t = await downsample(eng.gpu, s.work.tex, s.work.width, s.work.height, pw, ph, true, s.gain, "sam.input");
      const px = halvesToFloats(new Uint16Array(await eng.gpu.readTexture(t, 0, 0, pw, ph, 8)));
      eng.gpu.release(t);
      const img = new Float32Array(pw * ph * 3);
      for (let i = 0, j = 0; i < pw * ph; i++, j += 4) for (let c = 0; c < 3; c++) img[i * 3 + c] = 255 * Math.min(1, Math.max(0, px[j + c]));
      return img;
    });
    eng.progress("");
  }
  if (!st.guide) {
    const g = halvesToFloats(new Uint16Array(await eng.gpu.readTexture(s.maps.guide, 0, 0, w, h, 8)));
    st.guide = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) st.guide[i] = Math.min(1, Math.max(0, 0.2126 * g[i * 4] + 0.7152 * g[i * 4 + 1] + 0.0722 * g[i * 4 + 2]));
  }
  eng.progress("selection", "decode");
  const { low, iou } = await st.sam.decode(m.points!);
  eng.progress("selection", "filter");
  // No level chosen: SAM's own pick, its most confident of the three readings.
  const k = m.level === undefined ? [1, 2, 3].reduce((a, b) => (iou[b] > iou[a] ? b : a)) : levelsByArea(low)[m.level];
  return selectionMask(low, k, st.sam.dims, w, h, st.guide);
}
