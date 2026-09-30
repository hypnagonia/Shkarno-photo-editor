/**
 * EXIF metadata for exports. Pixels are always written upright, so
 * Orientation is 1. Capture metadata from the source (DNG via LibRaw, HEIC/
 * JPEG via their Exif block) is carried over.
 */
import type { PhotoMetadata } from "../decode/types.ts";
import { IfdWriter, rational, T, tiffHeader, concat, type Entry } from "./tiffWriter.ts";

export const SOFTWARE = "Shkarno";

function exifDate(d: Date): string {
  const p = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${p(d.getFullYear(), 4)}:${p(d.getMonth() + 1)}:${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function dms(v: number): number[] {
  const a = Math.abs(v);
  const d = Math.floor(a), mf = (a - d) * 60, m = Math.floor(mf), s = (mf - m) * 60;
  return [d, 1, m, 1, Math.round(s * 1000), 1000];
}

/** Tag entries describing the capture (usable in IFD0 of a TIFF/DNG too). */
export function captureEntries(m: PhotoMetadata, colorSpace: "srgb" | "p3") {
  const ifd0: Entry[] = [
    { tag: 0x0112, type: T.SHORT, values: [1] },
    { tag: 0x0131, type: T.ASCII, values: SOFTWARE },
  ];
  if (m.make) ifd0.push({ tag: 0x010f, type: T.ASCII, values: m.make });
  if (m.model) ifd0.push({ tag: 0x0110, type: T.ASCII, values: m.model });
  if (m.artist) ifd0.push({ tag: 0x013b, type: T.ASCII, values: m.artist });
  const now = exifDate(new Date());
  ifd0.push({ tag: 0x0132, type: T.ASCII, values: now });
  const exif: Entry[] = [
    { tag: 0x9000, type: T.UNDEFINED, values: new TextEncoder().encode("0232") },
    { tag: 0xa001, type: T.SHORT, values: [colorSpace === "srgb" ? 1 : 0xffff] },
  ];
  if (m.exposureTime) exif.push({ tag: 0x829a, type: T.RATIONAL, values: rational(m.exposureTime, 1000000) });
  if (m.fNumber) exif.push({ tag: 0x829d, type: T.RATIONAL, values: rational(m.fNumber, 100) });
  if (m.iso) exif.push({ tag: 0x8827, type: T.SHORT, values: [Math.min(65535, Math.round(m.iso))] });
  if (m.dateTime && !isNaN(m.dateTime.getTime())) {
    exif.push({ tag: 0x9003, type: T.ASCII, values: exifDate(m.dateTime) });
    exif.push({ tag: 0x9004, type: T.ASCII, values: exifDate(m.dateTime) });
  }
  if (m.focalLength) exif.push({ tag: 0x920a, type: T.RATIONAL, values: rational(m.focalLength, 100) });
  if (m.focalLength35) exif.push({ tag: 0xa405, type: T.SHORT, values: [Math.round(m.focalLength35)] });
  if (m.lensMake) exif.push({ tag: 0xa433, type: T.ASCII, values: m.lensMake });
  if (m.lensModel) exif.push({ tag: 0xa434, type: T.ASCII, values: m.lensModel });
  let gps: Entry[] | undefined;
  if (m.gps && Number.isFinite(m.gps.lat) && Number.isFinite(m.gps.lon)) {
    gps = [
      { tag: 0x0000, type: T.BYTE, values: [2, 3, 0, 0] },
      { tag: 0x0001, type: T.ASCII, values: m.gps.lat >= 0 ? "N" : "S" },
      { tag: 0x0002, type: T.RATIONAL, values: dms(m.gps.lat) },
      { tag: 0x0003, type: T.ASCII, values: m.gps.lon >= 0 ? "E" : "W" },
      { tag: 0x0004, type: T.RATIONAL, values: dms(m.gps.lon) },
    ];
    if (m.gps.alt !== undefined) {
      gps.push({ tag: 0x0005, type: T.BYTE, values: [m.gps.alt < 0 ? 1 : 0] });
      gps.push({ tag: 0x0006, type: T.RATIONAL, values: rational(Math.abs(m.gps.alt), 100) });
    }
  }
  return { ifd0, exif, gps };
}

/** Complete EXIF APP1 payload ("Exif\0\0" + TIFF). */
export function buildExifApp1(m: PhotoMetadata, colorSpace: "srgb" | "p3"): Uint8Array {
  const { ifd0, exif, gps } = captureEntries(m, colorSpace);
  ifd0.push({ tag: 0x8769, type: T.LONG, values: [0], patch: (o) => [o("exif")] });
  if (gps) ifd0.push({ tag: 0x8825, type: T.LONG, values: [0], patch: (o) => [o("gps")] });
  const ifds = [{ key: "ifd0", entries: ifd0 }, { key: "exif", entries: exif }];
  if (gps) ifds.push({ key: "gps", entries: gps });
  const { bytes } = IfdWriter.layout(ifds, 8);
  return concat([new Uint8Array([0x45, 0x78, 0x69, 0x66, 0, 0]), tiffHeader(8), bytes]);
}

/** Inserts APP1 (EXIF) and APP2 (ICC) segments right after SOI, replacing any the encoder wrote. */
export function injectJpegSegments(jpeg: Uint8Array, app1: Uint8Array | undefined, icc: Uint8Array | undefined): Uint8Array {
  if (jpeg[0] !== 0xff || jpeg[1] !== 0xd8) return jpeg;
  // Skip existing APP0/APP1/APP2 written by the browser encoder.
  let p = 2;
  const keep: Uint8Array[] = [];
  while (p + 4 <= jpeg.length && jpeg[p] === 0xff) {
    const marker = jpeg[p + 1];
    const len = (jpeg[p + 2] << 8) | jpeg[p + 3];
    if (marker === 0xe0 || marker === 0xe1 || marker === 0xe2) { if (marker === 0xe0) keep.push(jpeg.subarray(p, p + 2 + len)); p += 2 + len; continue; }
    break;
  }
  const segs: Uint8Array[] = [jpeg.subarray(0, 2), ...keep];
  if (app1 && app1.length + 2 < 65536) segs.push(new Uint8Array([0xff, 0xe1, (app1.length + 2) >> 8, (app1.length + 2) & 255]), app1);
  if (icc) {
    const hdr = new TextEncoder().encode("ICC_PROFILE\0");
    const len = hdr.length + 2 + icc.length + 2;
    if (len < 65536) segs.push(new Uint8Array([0xff, 0xe2, len >> 8, len & 255]), hdr, new Uint8Array([1, 1]), icc);
  }
  segs.push(jpeg.subarray(p));
  return concat(segs);
}
