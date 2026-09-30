/**
 * A moving object's background (Motion Blur set to Object): where the object is, the photo
 * shows the object — so what its streak should reveal behind it is inpainted, once per
 * mask, from a 1024 px render of the photo without the layer and without film. Kept as a
 * small texture over the object's box (render_motion_object.wgsl lays it in). Edits made
 * later leave it as it was until the mask changes: it only shows through the streak.
 */
import type { Engine } from "./engine.ts";
import type { Params } from "../decision/params.ts";
import { objectMotionLayer } from "../layers/gpu.ts";
import type { Session } from "./session.ts";
import { painterFor } from "./retouch.ts";

export async function ensureMotionPlate(eng: Engine, s: Session, p: Params) {
  const obj = objectMotionLayer(p.layers ?? [], p.autoCurves ?? 1, p.enable);
  if (!obj) { dropMotionPlate(eng, s); return; }
  const key = JSON.stringify([obj.layer.id, obj.layer.mask]);
  if (s.motionPlate?.key === key) return;
  dropMotionPlate(eng, s);
  s.motionPlate = { key };
  eng.progress("motion object");
  const t = await eng.ensureThumb(1024);
  const src = { base: t.base, denoised: t.denoised, width: t.w, height: t.h, fullWidth: s.work.width };
  const o = { wb: eng.wbFor(p), gain: s.gain, lightLinear: s.lightLinear, output: "p38" as const, dither: false };
  // The object's mask (view 7: the layer's mask as grey), then the photo without the layer.
  const rm = await eng.renderer.render(src, s.maps, p, { ...o, debugView: 7, region: obj.index }, false);
  const mk = new Uint8Array(await eng.gpu.readTexture(rm.tex, 0, rm.top, t.w, t.h, 4));
  const bare: Params = { ...p, layers: (p.layers ?? []).filter((l) => l.id !== obj.layer.id), film: p.film ? { ...p.film, character: "off" } : p.film };
  const r = await eng.renderer.render(src, s.maps, bare, o, bare.enable.dof && bare.dof.strength > 0);
  const img = new Uint8Array(await eng.gpu.readTexture(r.tex, 0, r.top, t.w, t.h, 4));
  // The object's box, grown so the network sees its surroundings.
  let x0 = t.w, y0 = t.h, x1 = -1, y1 = -1;
  for (let y = 0; y < t.h; y++) for (let x = 0; x < t.w; x++) {
    if (mk[(y * t.w + x) * 4 + 1] > 64) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  }
  if (x1 < 0) { eng.log("motion object: its mask covers nothing"); return; }
  const gx = Math.round((x1 - x0) * 0.25) + 12, gy = Math.round((y1 - y0) * 0.25) + 12;
  x0 = Math.max(0, x0 - gx); y0 = Math.max(0, y0 - gy); x1 = Math.min(t.w - 1, x1 + gx); y1 = Math.min(t.h - 1, y1 + gy);
  const w = x1 - x0 + 1, h = y1 - y0 + 1;
  const rgb = new Float32Array(w * h * 3), hole = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = ((y + y0) * t.w + x + x0) * 4, o3 = (y * w + x) * 3;
    rgb[o3] = img[i] / 255; rgb[o3 + 1] = img[i + 1] / 255; rgb[o3 + 2] = img[i + 2] / 255;
  }
  // The hole: the mask, grown 3 px (its soft edge belongs to the object too).
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let on = 0;
    for (let dy = -3; dy <= 3 && !on; dy++) for (let dx = -3; dx <= 3 && !on; dx++) {
      const X = x + x0 + dx, Y = y + y0 + dy;
      if (X >= 0 && Y >= 0 && X < t.w && Y < t.h && mk[(Y * t.w + X) * 4 + 1] > 32) on = 1;
    }
    hole[y * w + x] = on;
  }
  const t0 = performance.now();
  let fill: Float32Array;
  try { fill = await (await painterFor(eng, s)).run(rgb, w, h, hole); }
  catch (e) { eng.log(`motion object: no background (${e instanceof Error ? e.message : e})`); return; }
  if (s.motionPlate?.key !== key) return; // (the mask changed meanwhile)
  const bytes = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    for (let c = 0; c < 3; c++) bytes[i * 4 + c] = Math.round(Math.min(1, Math.max(0, fill[i * 3 + c])) * 255);
    bytes[i * 4 + 3] = 255;
  }
  const tex = eng.gpu.tex("motion.plate", w, h, "rgba8unorm", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST);
  eng.gpu.device.queue.writeTexture({ texture: tex }, bytes, { bytesPerRow: w * 4, rowsPerImage: h }, { width: w, height: h });
  s.motionPlate = { key, tex };
  eng.renderer.motionPlate = { tex, rect: [x0 / t.w, y0 / t.h, (x1 + 1) / t.w, (y1 + 1) / t.h] };
  eng.log(`motion object: background behind it inpainted (${w}×${h}) in ${Math.round(performance.now() - t0)} ms`);
}

export function dropMotionPlate(eng: Engine, s: Session) {
  if (s.motionPlate?.tex) eng.gpu.release(s.motionPlate.tex);
  s.motionPlate = undefined;
  eng.renderer.motionPlate = undefined;
}
