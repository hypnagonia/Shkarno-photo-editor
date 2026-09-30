/** Export at full resolution, in strips (JPEG, Ultra HDR, HEIC, TIFF, linear DNG). */
import type { Engine } from "./engine.ts";
import { ensureMotionPlate } from "./objectMotion.ts";
import { halvesToFloats } from "../gpu/half.ts";
import { encodeGainMapJpeg, encodeHeic, encodeJpeg, encodeLinearDng, encodeTiff16 } from "../output/encoders.ts";
import type { ExportFormat } from "./protocol.ts";
import { contrastEqActive } from "../tone/contrastEq.ts";
import { frameSize, frameToSource, isIdentityFrame, resampleFrame } from "../geometry/frame.ts";

export async function exportPhoto(eng: Engine, format: ExportFormat, quality: number, space: "srgb" | "p3", stripRows = 512) : Promise<{ blob: Blob; name: string; ms: number }> {
  const s = eng.s;
  if (!s) throw new Error("No photo open");
  const t0 = performance.now();
  const gpu = eng.gpu;
  const src = eng.renderSource(true);
  const p = s.params;
  await eng.ensureSelections(s, p);
  await ensureMotionPlate(eng, s, p);
  const W = src.width, H = src.height;
  // The frame (turns, mirror, straightening, crop): on the rendered photo, before encoding.
  const fr = isIdentityFrame(p.frame) ? undefined : p.frame!;
  const [FW, FH] = fr ? frameSize(fr, W, H) : [W, H];
  const framed = <T extends Uint8ClampedArray<ArrayBuffer> | Float32Array<ArrayBuffer>>(buf: T, w = W, h = H, ow = FW, oh = FH): T => {
    if (!fr) return buf;
    eng.progress("export", "framing");
    return resampleFrame(buf, w, h, frameToSource(fr, W, H), ow, oh);
  };
  const base = s.name.replace(/\.[^.]+$/, "");
  const P = eng.profiler;
  const o = { wb: eng.wbFor(p), gain: s.gain, lightLinear: s.lightLinear };
  const dof = p.enable.dof && p.dof.strength > 0;
  // Full-resolution exports render in strips: peak extra GPU memory stays at a
  // few tens of MB instead of several full-frame textures.
  // The contrast equalizer needs ≈ 254 rows of margin around each strip at full size: with
  // it, strips are half as tall, so the margin does not double every strip buffer.
  const STRIP = Math.max(64, Math.round(contrastEqActive(eng.s?.params.contrastEq) ? Math.min(stripRows, 256) : stripRows));
  const strips = async (each: (y0: number, rows: number) => Promise<void>) => {
    for (let y0 = 0; y0 < H; y0 += STRIP) {
      eng.progress("export", `rendering ${Math.round((y0 / H) * 100)}%`, y0 / H);
      await each(y0, Math.min(STRIP, H - y0));
    }
  };
  try {
    if (format === "dng") {
      const f = new Float32Array(W * H * 4);
      await P.time("export render (linear)", () => strips(async (y0, rows) => {
        const r = await eng.renderer.renderLinear(src, s.maps, p, { ...o, output: "p3f16" }, { y0, rows });
        await eng.readHalfRows(r.tex, r.top, W, rows, f.subarray(y0 * W * 4, (y0 + rows) * W * 4));
      }));
      eng.progress("export", "writing DNG");
      const blob = await P.time("encode DNG", () => encodeLinearDng(framed(f), FW, FH, s.decoded.meta));
      eng.post({ type: "profile", stages: P.stages });
      return { blob, name: `${base}-processed-linear.dng`, ms: performance.now() - t0 };
    }
    if (format === "tiff16") {
      const f = new Float32Array(W * H * 4);
      await P.time("export render (16-bit)", () => strips(async (y0, rows) => {
        const r = await eng.renderer.render(src, s.maps, p, { ...o, output: "p3f16" }, dof, { y0, rows });
        await eng.readHalfRows(r.tex, r.top, W, rows, f.subarray(y0 * W * 4, (y0 + rows) * W * 4));
      }));
      eng.progress("export", "writing TIFF");
      const blob = await P.time("encode TIFF", () => encodeTiff16(framed(f), FW, FH, s.decoded.meta));
      eng.post({ type: "profile", stages: P.stages });
      return { blob, name: `${base}-edit.tif`, ms: performance.now() - t0 };
    }
    if (format === "jpeg-hdr") {
      // SDR image + gain map, strip by strip. The gain map is ½ size (¼ above 24 MP);
      // strips start on multiples of the block size so its rows line up.
      const s2 = W * H > 24e6 ? 4 : 2;
      const stops = p.hdr?.headroom || 2;
      const gw = Math.ceil(W / s2), gh = Math.ceil(H / s2);
      const rgba = new Uint8ClampedArray(W * H * 4);
      const gain = new Uint8ClampedArray(gw * gh * 4);
      const HS = Math.max(64, Math.round(STRIP / s2) * s2);
      await P.time("export render (HDR)", async () => {
        for (let y0 = 0; y0 < H; y0 += HS) {
          eng.progress("export", `rendering ${Math.round((y0 / H) * 100)}%`, y0 / H);
          const rows = Math.min(HS, H - y0);
          const r = await eng.renderer.render(src, s.maps, { ...p, hdr: { headroom: stops } }, { ...o, output: space === "p3" ? "p38" : "srgb8", hdr: true, gainMap: { scale: s2, stops } }, dof, { y0, rows });
          rgba.set(new Uint8Array(await gpu.readTexture(r.tex, 0, r.top, W, rows, 4)), y0 * W * 4);
          if (r.gm) gain.set(new Uint8Array(await gpu.readTexture(r.gm, 0, 0, r.gmW!, r.gmRows!, 4)), (y0 / s2) * gw * 4);
        }
      }, () => `${W}×${H} + gain map ${gw}×${gh}, +${stops} EV`);
      eng.progress("export", "encoding JPEG (HDR)");
      const fgw = Math.ceil(FW / s2), fgh = Math.ceil(FH / s2);
      const blob = await P.time("encode JPEG (HDR)", () => encodeGainMapJpeg(framed(rgba), FW, FH, framed(gain, gw, gh, fgw, fgh), fgw, fgh, stops, space, quality, s.decoded.meta));
      eng.post({ type: "profile", stages: P.stages });
      return { blob, name: `${base}-edit-hdr.jpg`, ms: performance.now() - t0 };
    }
    const rgba = new Uint8ClampedArray(W * H * 4);
    await P.time("export render", () => strips(async (y0, rows) => {
      const r = await eng.renderer.render(src, s.maps, p, { ...o, output: space === "p3" ? "p38" : "srgb8" }, dof, { y0, rows });
      rgba.set(new Uint8Array(await gpu.readTexture(r.tex, 0, r.top, W, rows, 4)), y0 * W * 4);
    }), () => `${W}×${H} in ${Math.ceil(H / STRIP)} strips`);
    eng.progress("export", `encoding ${format.toUpperCase()}`);
    const out = framed(rgba);
    const blob = await P.time(`encode ${format}`, () => format === "heic" ? encodeHeic(out, FW, FH, space, quality) : encodeJpeg(out, FW, FH, space, quality, s.decoded.meta));
    eng.post({ type: "profile", stages: P.stages });
    return { blob, name: `${base}-edit.${format === "heic" ? "heic" : "jpg"}`, ms: performance.now() - t0 };
  } finally {
    // Strip targets and readback buffers are export-sized; the next preview re-creates its own.
    eng.renderer.releaseTargets();
    eng.gpu.flushStaging();
  }
}

export async function readHalfRows(eng: Engine, tex: GPUTexture, top: number, W: number, rows: number, out: Float32Array<ArrayBuffer>) {
  const half = new Uint16Array(await eng.gpu.readTexture(tex, 0, top, W, rows, 8));
  halvesToFloats(half, out);
}
