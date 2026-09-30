/**
 * Gain-map JPEG (HDR): an ordinary SDR JPEG followed by a second JPEG — the
 * gain map — that tells HDR screens how much brighter each area may be.
 * SDR viewers show the first image and ignore the rest.
 *
 * Written in all three forms readers look for:
 *   Ultra HDR / Adobe gain map   XMP `hdrgm:` + Google GContainer directory in
 *                                the primary, `hdrgm:` values in the gain map
 *                                (Chrome, Android, Google Photos, Adobe)
 *   MPF                          CIPA DC-007 index locating the second image
 *   ISO 21496-1                  APP2 "urn:iso:std:iso:ts:21496:-1" — version in
 *                                the primary, the binary metadata in the gain
 *                                map (Apple iOS 18 / macOS 15, recent Chrome)
 *
 * Layout: primary = SOI, APP0 (as encoded), APP1 Exif, APP1 XMP, APP2 ISO
 * (version), APP2 ICC, APP2 MPF, …image…, EOI; gain map (right after) = SOI,
 * APP1 XMP, APP2 ISO (metadata), …image…, EOI.
 *
 * Decoder model the values describe (log2 domain, single channel):
 *   G   = 2^(min + (max − min)·code^(1/gamma))
 *   HDR = (SDR + offSdr)·G^w − offHdr,  w = clamp((log2 headroom − capMin)/(capMax − capMin), 0, 1)
 */

export interface GainMapMeta {
  /** log2 of the smallest and largest gain the map encodes. */
  min: number;
  max: number;
  gamma: number;
  offSdr: number;
  offHdr: number;
  /** log2 display headroom where the map starts / is fully applied. */
  capMin: number;
  capMax: number;
}

const XMP_NS = "http://ns.adobe.com/xap/1.0/\0";
const ISO_NS = "urn:iso:std:iso:ts:21496:-1\0";
const enc = new TextEncoder();

/** A marker segment FF xx + length + payload. */
export function segment(marker: number, ...parts: Uint8Array[]): Uint8Array {
  const n = parts.reduce((a, p) => a + p.length, 0) + 2;
  if (n >= 65536) throw new Error(`JPEG segment too large (${n} bytes)`);
  return concat([new Uint8Array([0xff, marker, n >> 8, n & 255]), ...parts]);
}

export function concat(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

export interface JpegSegment { marker: number; start: number; end: number }
/** The marker segments before the image data (SOI excluded), with their byte ranges. */
export function parseJpegSegments(b: Uint8Array, from = 0): { segments: JpegSegment[]; body: number } {
  if (b[from] !== 0xff || b[from + 1] !== 0xd8) throw new Error("not a JPEG");
  let p = from + 2;
  const segments: JpegSegment[] = [];
  while (p + 4 <= b.length && b[p] === 0xff) {
    const marker = b[p + 1];
    if (marker === 0xda) break; // start of scan: the rest is image data
    const len = (b[p + 2] << 8) | b[p + 3];
    segments.push({ marker, start: p, end: p + 2 + len });
    p += 2 + len;
  }
  return { segments, body: p };
}

/**
 * The JPEG with its APP0/APP1/APP2 segments replaced by `segs` (APP0 is kept
 * first when `keepApp0`); every other segment (tables, frame header) and the
 * image data stay as they are.
 */
export function rebuildJpeg(jpeg: Uint8Array, segs: Uint8Array[], keepApp0 = true): Uint8Array {
  const { segments, body } = parseJpegSegments(jpeg);
  // The leading APP0–2 run is replaced; from the first other segment on, everything stays.
  let i = 0;
  while (i < segments.length && segments[i].marker >= 0xe0 && segments[i].marker <= 0xe2) i++;
  const app0 = segments.slice(0, i).filter((s) => s.marker === 0xe0).map((s) => jpeg.subarray(s.start, s.end));
  const rest = jpeg.subarray(i < segments.length ? segments[i].start : body);
  return concat([jpeg.subarray(0, 2), ...(keepApp0 ? app0 : []), ...segs, rest]);
}

const num = (v: number) => String(Math.round(v * 1e6) / 1e6);

export function buildXmpPrimary(gainLength: number): Uint8Array {
  const xml =
    `<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Shkarno">` +
    `<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">` +
    `<rdf:Description rdf:about="" xmlns:Container="http://ns.google.com/photos/1.0/container/" xmlns:Item="http://ns.google.com/photos/1.0/container/item/" xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" hdrgm:Version="1.0">` +
    `<Container:Directory><rdf:Seq>` +
    `<rdf:li rdf:parseType="Resource"><Container:Item Item:Semantic="Primary" Item:Mime="image/jpeg"/></rdf:li>` +
    `<rdf:li rdf:parseType="Resource"><Container:Item Item:Semantic="GainMap" Item:Mime="image/jpeg" Item:Length="${gainLength}"/></rdf:li>` +
    `</rdf:Seq></Container:Directory>` +
    `</rdf:Description></rdf:RDF></x:xmpmeta>`;
  return concat([enc.encode(XMP_NS), enc.encode(xml)]);
}

export function buildXmpGainMap(m: GainMapMeta): Uint8Array {
  const xml =
    `<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Shkarno">` +
    `<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">` +
    `<rdf:Description rdf:about="" xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" hdrgm:Version="1.0"` +
    ` hdrgm:GainMapMin="${num(m.min)}" hdrgm:GainMapMax="${num(m.max)}" hdrgm:Gamma="${num(m.gamma)}"` +
    ` hdrgm:OffsetSDR="${num(m.offSdr)}" hdrgm:OffsetHDR="${num(m.offHdr)}"` +
    ` hdrgm:HDRCapacityMin="${num(m.capMin)}" hdrgm:HDRCapacityMax="${num(m.capMax)}" hdrgm:BaseRenditionIsHDR="False"/>` +
    `</rdf:RDF></x:xmpmeta>`;
  return concat([enc.encode(XMP_NS), enc.encode(xml)]);
}

/** ISO 21496-1 payload: version only (primary), or the full single-channel metadata (gain map). */
export function buildIso(m?: GainMapMeta): Uint8Array {
  if (!m) return concat([enc.encode(ISO_NS), new Uint8Array(4)]);
  const b = new Uint8Array(61);
  const dv = new DataView(b.buffer);
  let o = 0;
  const u16 = (v: number) => { dv.setUint16(o, v); o += 2; };
  const u8 = (v: number) => { dv.setUint8(o, v); o += 1; };
  const u32 = (v: number) => { dv.setUint32(o, v >>> 0); o += 4; };
  const s32 = (v: number) => { dv.setInt32(o, v | 0); o += 4; };
  const frac = (v: number, signed: boolean) => {
    // Exact for integers and 1/64 offsets, else micro-units.
    const d = Number.isInteger(v) ? 1 : Number.isInteger(v * 64) ? 64 : 1000000;
    (signed ? s32 : u32)(Math.round(v * d)); u32(d);
  };
  u16(0); u16(0);                      // minimum version, writer version
  u8(0x40);                            // single channel, use the base colour space, separate denominators
  frac(m.capMin, false);               // base HDR headroom (log2)
  frac(m.capMax, false);               // alternate HDR headroom (log2)
  frac(m.min, true);                   // gain map min (log2)
  frac(m.max, true);                   // gain map max (log2)
  frac(m.gamma, false);
  frac(m.offSdr, true);                // base offset
  frac(m.offHdr, true);                // alternate offset
  return concat([enc.encode(ISO_NS), b]);
}

/** MPF APP2 payload: an index of two images (sizes and the gain map's offset patched in later). */
export function buildMpf(primarySize: number, gainSize: number, gainOffset: number): Uint8Array {
  const b = new Uint8Array(4 + 82);
  b.set(enc.encode("MPF\0"), 0);
  const dv = new DataView(b.buffer, 4);
  dv.setUint32(0, 0x4d4d002a);         // big-endian TIFF header
  dv.setUint32(4, 8);                  // offset of the MP Index IFD
  dv.setUint16(8, 3);                  // three entries
  const entry = (at: number, tag: number, type: number, count: number, value: number | Uint8Array) => {
    dv.setUint16(at, tag); dv.setUint16(at + 2, type); dv.setUint32(at + 4, count);
    if (typeof value === "number") dv.setUint32(at + 8, value);
    else for (let i = 0; i < 4; i++) dv.setUint8(at + 8 + i, value[i]);
  };
  entry(10, 0xb000, 7, 4, enc.encode("0100")); // MPFVersion
  entry(22, 0xb001, 4, 1, 2);                   // NumberOfImages
  entry(34, 0xb002, 7, 32, 50);                 // MPEntry: 2 × 16 bytes at offset 50
  dv.setUint32(46, 0);                          // no next IFD
  // MP entries: attributes, size, offset (from the endian marker; 0 = first image), two dependent-image entries.
  dv.setUint32(50, 0x00030000); dv.setUint32(54, primarySize); dv.setUint32(58, 0); dv.setUint16(62, 0); dv.setUint16(64, 0);
  dv.setUint32(66, 0x00000000); dv.setUint32(70, gainSize); dv.setUint32(74, gainOffset); dv.setUint16(78, 0); dv.setUint16(80, 0);
  return b;
}

/**
 * Joins an SDR JPEG and a gain-map JPEG into one gain-map JPEG.
 * `exif` / `icc` are the APP1 Exif payload and the ICC profile of the primary.
 */
export function muxGainMapJpeg(primaryJpeg: Uint8Array, gainJpeg: Uint8Array, m: GainMapMeta, exif?: Uint8Array, icc?: Uint8Array): Uint8Array {
  // 1. The gain map image: its own XMP and ISO metadata, nothing else.
  const gain = rebuildJpeg(gainJpeg, [segment(0xe1, buildXmpGainMap(m)), segment(0xe2, buildIso(m))], false);
  // 2. The primary, with a placeholder MPF of fixed size.
  const iccSeg = icc ? segment(0xe2, enc.encode("ICC_PROFILE\0"), new Uint8Array([1, 1]), icc) : undefined;
  const mpfSeg = segment(0xe2, buildMpf(0, 0, 0));
  const segs = [
    ...(exif ? [segment(0xe1, exif)] : []),
    segment(0xe1, buildXmpPrimary(gain.length)),
    segment(0xe2, buildIso()),
    ...(iccSeg ? [iccSeg] : []),
    mpfSeg,
  ];
  const primary = rebuildJpeg(primaryJpeg, segs);
  // 3. Patch the MPF with the real sizes and the gain map's offset from the endian marker.
  const at = findSegment(primary, mpfSeg);
  const endian = at + 8; // FF E2 LL LL "MPF\0"
  primary.set(segment(0xe2, buildMpf(primary.length, gain.length, primary.length - endian)), at);
  return concat([primary, gain]);
}

function findSegment(hay: Uint8Array, needle: Uint8Array): number {
  outer: for (let i = 0; i + needle.length <= hay.length; i++) {
    for (let j = 0; j < needle.length; j++) if (hay[i + j] !== needle[j]) continue outer;
    return i;
  }
  throw new Error("segment not found");
}
