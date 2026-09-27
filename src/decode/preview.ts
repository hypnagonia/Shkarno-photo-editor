import { srgbEotf as eotf, srgbOetf as oetf } from "../color/transfer.ts";
import { jpegDC } from "./jpegDC.ts";
/**
 * The camera's own rendering, embedded in a DNG: Apple ProRAW (and most camera
 * DNGs) carry a full-size JPEG preview of the photo as the phone showed it. Its
 * brightness is the reference for automatic exposure ("as bright as the
 * original"), measured here without decoding it at full size.
 *
 * The file is scanned for JPEG streams; each one's frame header (SOF) gives its
 * size and component count without decoding (Apple's semantic mattes are
 * single-component and are skipped). The largest colour JPEG is decoded at a
 * small size and its luminance quantiles returned (display-encoded, 0…1).
 */

interface JpegInfo { offset: number; width: number; height: number; components: number }

/** Frame header of the JPEG starting at `o` (null if not a plausible JPEG). */
export function jpegInfo(b: Uint8Array, o: number): JpegInfo | null {
  let i = o + 2;
  const end = Math.min(b.length - 9, o + (1 << 20)); // the SOF sits in the first KB of real files; 1 MB is generous
  while (i < end) {
    if (b[i] !== 0xff) return null;
    const m = b[i + 1];
    if (m === 0xd8 || m === 0x01 || (m >= 0xd0 && m <= 0xd7)) { i += 2; continue; }
    const len = (b[i + 2] << 8) | b[i + 3];
    if (len < 2) return null;
    // Frame headers. Only baseline / extended / progressive (C0–C2) are pictures: the
    // raw sensor data itself is stored as lossless JPEG (C3 …) and must not be read as one.
    if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) {
      if (m > 0xc2) return null;
      return { offset: o, height: (b[i + 5] << 8) | b[i + 6], width: (b[i + 7] << 8) | b[i + 8], components: b[i + 9] };
    }
    if (m === 0xda) return null; // scan started without a frame header
    i += 2 + len;
  }
  return null;
}

/** The largest colour picture JPEG in the bytes (the camera's rendering), if any. */
export function findPreview(head: Uint8Array): JpegInfo | null {
  let best: JpegInfo | null = null;
  for (let i = 0; i < head.length - 4; i++) {
    if (head[i] !== 0xff || head[i + 1] !== 0xd8 || head[i + 2] !== 0xff) continue;
    const info = jpegInfo(head, i);
    if (info && info.components === 3 && Math.max(info.width, info.height) >= 256 && (!best || info.width * info.height > best.width * best.height)) best = info;
  }
  return best;
}


/**
 * The levels compared with the camera's rendering: the shadows (black point) and
 * the median (exposure, last). Both sides are measured at exactly these.
 */
export const REF_QS = [0.005, 0.02, 0.05, 0.1, 0.25, 0.5];
export const MEDIAN = REF_QS.length - 1;

/** Display-encoded luminance quantiles of rendered 8-bit P3 pixels (rgba), subsampled. */
export function renderedQuantiles(px: Uint8Array, qs: number[] = REF_QS, stride = 7): number[] {
  const ys: number[] = [];
  for (let k = 0; k < px.length; k += 4 * stride) {
    const Y = 0.229 * eotf(px[k] / 255) + 0.6917 * eotf(px[k + 1] / 255) + 0.0793 * eotf(px[k + 2] / 255);
    ys.push(oetf(Y));
  }
  ys.sort((a, b) => a - b);
  return qs.map((q) => (ys.length ? ys[Math.min(ys.length - 1, Math.floor(q * ys.length))] : 0));
}

/**
 * Black point from the camera's rendering: a curve taking our rendered shadow
 * levels (`ours`, at REF_QS) to the camera's (`ref`) — deeper where ours are
 * lifted, opened where ours are crushed — and identity from the median up.
 * The deep tones only (to the darkest 10 %: lower midtones and the exposure stay),
 * 70 % of the way (embedded previews can be deeper than the camera's own full
 * rendering), each level at most 20/255; the result is monotone with a sane slope.
 * Undefined when the shadows already agree (within 3/255).
 */
export function shadowMatch(ours: number[], ref: number[]): Array<{ x: number; y: number }> | undefined {
  const MAX = 20 / 255, SHARE = 0.7, DEEP = REF_QS.indexOf(0.1);
  const pts: Array<{ x: number; y: number }> = [{ x: 0, y: 0 }];
  let moved = 0;
  for (let i = 0; i <= DEEP; i++) {
    const x = ours[i];
    if (x <= pts[pts.length - 1].x + 0.01 || x >= ours[MEDIAN] - 0.02) continue;
    const prev = pts[pts.length - 1];
    let y = x + Math.max(-MAX, Math.min(MAX, SHARE * (ref[i] - x)));
    // Monotone, no flat or near-vertical stretch between points.
    y = Math.max(y, prev.y + (x - prev.x) * 0.35, 1 / 255);
    y = Math.min(y, prev.y + (x - prev.x) * 2.5);
    moved = Math.max(moved, Math.abs(y - x));
    pts.push({ x, y });
  }
  if (moved < 3 / 255) return undefined;
  const m = ours[MEDIAN];
  if (m > pts[pts.length - 1].x + 0.02) pts.push({ x: m, y: m });
  pts.push({ x: 1, y: 1 });
  return pts.map((p) => ({ x: Math.round(p.x * 1000) / 1000, y: Math.round(p.y * 1000) / 1000 }));
}

export interface PreviewStats {
  width: number; height: number;
  /** Display-encoded luminance at the requested quantiles. */
  q: number[];
}

/**
 * Luminance quantiles of the embedded camera rendering, or undefined when the
 * file has none (or only a thumbnail: under 256 px on the long side).
 */
/**
 * The embedded camera rendering's pixels (RGBA, display-encoded Display P3) at `long`
 * px on the long side, or undefined when the file has none (or only a thumbnail).
 * For whole-photo figures (Check's "camera" colour and brightness): not aligned
 * pixel for pixel with the development.
 */
export async function embeddedPreviewPixels(file: Blob, long = 256, maxMP = Infinity): Promise<{ rgba: Uint8Array; w: number; h: number } | undefined> {
  if (typeof createImageBitmap !== "function" || typeof OffscreenCanvas === "undefined") return undefined;
  let head: Uint8Array;
  try { head = new Uint8Array(await file.slice(0, Math.min(file.size, (maxMP < Infinity ? 24 : 48) << 20)).arrayBuffer()); }
  catch { return undefined; }
  const best = findPreview(head);
  if (!best) return undefined;
  // A big preview (a 48 MP ProRAW's is 8064×6048): its DC coefficients alone give it at
  // 1/8 size with ≈ 1 MB, where a browser decode would hold ≈ 200 MB on a phone.
  if (Math.max(best.width, best.height) / 8 >= long) {
    const dc = jpegDC(head.subarray(best.offset))
      ?? jpegDC(new Uint8Array(await file.slice(best.offset, best.offset + Math.min(file.size - best.offset, 48 << 20)).arrayBuffer()));
    if (dc) return boxDown(dc, long);
  }
  if ((best.width * best.height) / 1e6 > maxMP) return undefined;
  try {
    const s = long / Math.max(best.width, best.height);
    const w = Math.max(1, Math.round(best.width * s)), h = Math.max(1, Math.round(best.height * s));
    const bmp = await createImageBitmap(file.slice(best.offset, best.offset + Math.max(4 << 20, best.width * best.height * 2)), { resizeWidth: w, resizeHeight: h, resizeQuality: "medium" });
    const c = new OffscreenCanvas(bmp.width, bmp.height);
    const g = c.getContext("2d", { willReadFrequently: true })!;
    g.drawImage(bmp, 0, 0);
    bmp.close();
    return { rgba: new Uint8Array(g.getImageData(0, 0, c.width, c.height).data.buffer), w: c.width, h: c.height };
  } catch {
    return undefined;
  }
}

/** Box-average down to `long` px on the long side. */
function boxDown(img: { rgba: Uint8Array; w: number; h: number }, long: number): { rgba: Uint8Array; w: number; h: number } {
  const k = Math.max(1, Math.floor(Math.max(img.w, img.h) / long));
  const w = Math.floor(img.w / k), h = Math.floor(img.h / k);
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const acc = [0, 0, 0];
    for (let j = 0; j < k; j++) for (let i = 0; i < k; i++) { const o = ((y * k + j) * img.w + x * k + i) * 4; acc[0] += img.rgba[o]; acc[1] += img.rgba[o + 1]; acc[2] += img.rgba[o + 2]; }
    const o = (y * w + x) * 4;
    out[o] = acc[0] / (k * k); out[o + 1] = acc[1] / (k * k); out[o + 2] = acc[2] / (k * k); out[o + 3] = 255;
  }
  return { rgba: out, w, h };
}

export async function embeddedPreviewStats(file: Blob, qs: number[], maxMP = Infinity): Promise<PreviewStats | undefined> {
  if (typeof createImageBitmap !== "function" || typeof OffscreenCanvas === "undefined") return undefined;
  // Previews sit in the first part of the file (before the raw data in Apple's layout,
  // but not always): read up to 48 MB, which also covers most cameras' layouts.
  let head: Uint8Array;
  try { head = new Uint8Array(await file.slice(0, Math.min(file.size, (maxMP < Infinity ? 24 : 48) << 20)).arrayBuffer()); }
  catch { return undefined; } // no memory for it: the photo opens without the reference
  const best = findPreview(head);
  // A phone's browser may decode the whole picture before shrinking it (48 MP ≈ 200 MB).
  if (!best || (best.width * best.height) / 1e6 > maxMP) return undefined;
  try {
    const w = 256, h = Math.max(1, Math.round((w * best.height) / best.width));
    // The JPEG only (not the raw data after it): ≈ 2 bytes per pixel is a generous bound.
    const bmp = await createImageBitmap(file.slice(best.offset, best.offset + Math.max(4 << 20, best.width * best.height * 2)), { resizeWidth: w, resizeHeight: h, resizeQuality: "medium" });
    const c = new OffscreenCanvas(bmp.width, bmp.height);
    const g = c.getContext("2d", { willReadFrequently: true })!;
    g.drawImage(bmp, 0, 0);
    bmp.close();
    const px = g.getImageData(0, 0, c.width, c.height).data;
    const ys: number[] = [];
    for (let k = 0; k < px.length; k += 4) {
      // Display P3 luminance (Apple's previews are P3); encoded like our own display levels.
      const Y = 0.229 * eotf(px[k] / 255) + 0.6917 * eotf(px[k + 1] / 255) + 0.0793 * eotf(px[k + 2] / 255);
      ys.push(oetf(Y));
    }
    ys.sort((a, b) => a - b);
    return { width: best.width, height: best.height, q: qs.map((q) => ys[Math.min(ys.length - 1, Math.floor(q * ys.length))]) };
  } catch {
    return undefined;
  }
}
