/**
 * ICC v2 display profiles (matrix/TRC) generated from primaries, so exports
 * carry an exact description of their encoding: sRGB or Display P3 (both with
 * the sRGB transfer curve).
 */
import { bradford, D50_XY, D65_XY, P3_D65, rgbToXYZ, SRGB, type Primaries } from "../color/spaces.ts";
import { mul } from "../color/mat3.ts";
import { concat } from "./tiffWriter.ts";

function s15(v: number): number { return Math.round(v * 65536) | 0; }

function xyzTag(x: number, y: number, z: number): Uint8Array {
  const b = new Uint8Array(20);
  const dv = new DataView(b.buffer);
  b.set([0x58, 0x59, 0x5a, 0x20]); // 'XYZ '
  dv.setInt32(8, s15(x)); dv.setInt32(12, s15(y)); dv.setInt32(16, s15(z));
  return b;
}

function curvTag(): Uint8Array {
  // Parametric sRGB curve as a 1024-entry table (v2 profiles have no 'para').
  const n = 1024;
  const b = new Uint8Array(12 + n * 2);
  const dv = new DataView(b.buffer);
  b.set([0x63, 0x75, 0x72, 0x76]); // 'curv'
  dv.setUint32(8, n);
  for (let i = 0; i < n; i++) {
    const v = i / (n - 1);
    const lin = v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    dv.setUint16(12 + i * 2, Math.round(lin * 65535));
  }
  return b;
}

function descTag(text: string): Uint8Array {
  const n = text.length + 1;
  const b = new Uint8Array(12 + n + 12 + 67);
  const dv = new DataView(b.buffer);
  b.set([0x64, 0x65, 0x73, 0x63]); // 'desc'
  dv.setUint32(8, n);
  for (let i = 0; i < text.length; i++) b[12 + i] = text.charCodeAt(i);
  return b;
}

function textTag(text: string): Uint8Array {
  const b = new Uint8Array(8 + text.length + 1);
  b.set([0x74, 0x65, 0x78, 0x74]); // 'text'
  for (let i = 0; i < text.length; i++) b[8 + i] = text.charCodeAt(i);
  return b;
}

export function buildIcc(which: "srgb" | "p3"): Uint8Array {
  const prim: Primaries = which === "srgb" ? SRGB : P3_D65;
  // ICC PCS is D50: adapt the colourants with Bradford.
  const M = mul(bradford(D65_XY, D50_XY), rgbToXYZ(prim));
  const col = (c: number) => xyzTag(M[c], M[3 + c], M[6 + c]);
  const curve = curvTag();
  const tags: Array<[string, Uint8Array]> = [
    ["desc", descTag(which === "srgb" ? "sRGB (Shkarno)" : "Display P3 (Shkarno)")],
    ["cprt", textTag("No copyright, use freely")],
    ["wtpt", xyzTag(0.9642, 1.0, 0.8249)],
    ["rXYZ", col(0)], ["gXYZ", col(1)], ["bXYZ", col(2)],
    ["rTRC", curve], ["gTRC", curve], ["bTRC", curve],
  ];
  // Shared data for the three TRCs.
  const unique: Uint8Array[] = [];
  const offsets: number[] = [];
  const sizes: number[] = [];
  let pos = 128 + 4 + tags.length * 12;
  for (const [, data] of tags) {
    const prev = unique.indexOf(data);
    if (prev >= 0) { offsets.push(offsets[tags.findIndex((t) => t[1] === data)]); sizes.push(data.length); continue; }
    unique.push(data);
    offsets.push(pos);
    sizes.push(data.length);
    pos += data.length + ((4 - (data.length % 4)) % 4);
  }
  const total = pos;
  const head = new Uint8Array(128 + 4 + tags.length * 12);
  const dv = new DataView(head.buffer);
  dv.setUint32(0, total);
  dv.setUint32(8, 0x02100000); // v2.1
  head.set([0x6d, 0x6e, 0x74, 0x72], 12); // 'mntr'
  head.set([0x52, 0x47, 0x42, 0x20], 16); // 'RGB '
  head.set([0x58, 0x59, 0x5a, 0x20], 20); // 'XYZ '
  const now = new Date();
  dv.setUint16(24, now.getUTCFullYear()); dv.setUint16(26, now.getUTCMonth() + 1); dv.setUint16(28, now.getUTCDate());
  head.set([0x61, 0x63, 0x73, 0x70], 36); // 'acsp'
  dv.setUint32(64, 0); // perceptual
  dv.setInt32(68, s15(0.9642)); dv.setInt32(72, s15(1.0)); dv.setInt32(76, s15(0.8249));
  dv.setUint32(128, tags.length);
  tags.forEach(([sig, ], i) => {
    const o = 132 + i * 12;
    for (let k = 0; k < 4; k++) head[o + k] = sig.charCodeAt(k);
    dv.setUint32(o + 4, offsets[i]);
    dv.setUint32(o + 8, sizes[i]);
  });
  const body = unique.map((d) => { const pad = (4 - (d.length % 4)) % 4; return pad ? concat([d, new Uint8Array(pad)]) : d; });
  return concat([head, ...body]);
}
