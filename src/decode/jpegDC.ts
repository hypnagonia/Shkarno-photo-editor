/**
 * A JPEG at one eighth of its size, from its DC coefficients only.
 *
 * Every 8×8 block of a JPEG stores its average as the DC coefficient; reading
 * only those (the AC coefficients are entropy-decoded and skipped, never
 * transformed) gives the picture at 1/8 scale with memory for that size alone.
 * A 48 MP camera preview becomes 1008×756 using ≈ 1 MB, where a browser decode
 * holds ≈ 200 MB — too much on a phone. 8-bit baseline and extended sequential
 * Huffman JPEGs (SOF0/SOF1), one scan, any chroma subsampling, restart intervals.
 * Anything else — progressive, arithmetic, 12-bit, several scans, or data cut
 * short — returns undefined (never a picture that is partly made up).
 */

/** Canonical Huffman table: a 9-bit lookahead table plus the slow path for longer codes. */
interface Huff { lut: Uint16Array; maxcode: Int32Array; valptr: Int32Array; mincode: Int32Array; symbols: Uint8Array }
const LOOK = 9;

function buildHuff(counts: Uint8Array, symbols: Uint8Array): Huff {
  const lut = new Uint16Array(1 << LOOK); // (length << 8) | symbol, 0 = not a short code
  const maxcode = new Int32Array(18).fill(-1), valptr = new Int32Array(17), mincode = new Int32Array(17);
  let code = 0, k = 0;
  for (let len = 1; len <= 16; len++) {
    valptr[len] = k; mincode[len] = code;
    for (let i = 0; i < counts[len - 1]; i++, k++, code++) {
      if (len <= LOOK) {
        const shift = LOOK - len;
        for (let f = 0; f < 1 << shift; f++) lut[(code << shift) | f] = (len << 8) | symbols[k];
      }
    }
    maxcode[len] = counts[len - 1] ? code - 1 : -1;
    code <<= 1;
  }
  maxcode[17] = 0x7fffffff;
  return { lut, maxcode, valptr, mincode, symbols: symbols.slice() };
}

interface Comp { id: number; h: number; v: number; tq: number; td: number; ta: number; pred: number; bw: number; bh: number; dc: Float32Array }

export function jpegDC(b: Uint8Array): { rgba: Uint8Array; w: number; h: number } | undefined {
  if (b[0] !== 0xff || b[1] !== 0xd8) return undefined;
  const qt: Array<Uint16Array | undefined> = [];
  const dcT: Array<Huff | undefined> = [], acT: Array<Huff | undefined> = [];
  let comps: Comp[] = [];
  let W = 0, H = 0, restart = 0;
  let i = 2;
  while (i < b.length - 4) {
    if (b[i] !== 0xff) return undefined;
    const m = b[i + 1];
    if (m === 0xff) { i++; continue; }
    const len = (b[i + 2] << 8) | b[i + 3];
    const seg = i + 4;
    if (m === 0xdb) { // quantisation tables
      let p = seg;
      while (p < i + 2 + len) {
        const pq = b[p] >> 4, tq = b[p] & 15; p++;
        const t = new Uint16Array(64);
        for (let k = 0; k < 64; k++) { t[k] = pq ? (b[p] << 8) | b[p + 1] : b[p]; p += pq ? 2 : 1; }
        qt[tq] = t;
      }
    } else if (m === 0xc4) { // Huffman tables
      let p = seg;
      while (p < i + 2 + len) {
        const tc = b[p] >> 4, th = b[p] & 15; p++;
        const counts = b.subarray(p, p + 16); p += 16;
        const n = counts.reduce((a, c) => a + c, 0);
        const h = buildHuff(counts, b.subarray(p, p + n)); p += n;
        (tc === 0 ? dcT : acT)[th] = h;
      }
    } else if (m === 0xc0 || m === 0xc1) { // baseline / extended sequential frame
      if (b[seg] !== 8) return undefined; // 12-bit samples: not this decoder
      H = (b[seg + 1] << 8) | b[seg + 2]; W = (b[seg + 3] << 8) | b[seg + 4];
      const nc = b[seg + 5];
      comps = [];
      for (let k = 0; k < nc; k++) {
        const o = seg + 6 + k * 3;
        comps.push({ id: b[o], h: b[o + 1] >> 4, v: b[o + 1] & 15, tq: b[o + 2], td: 0, ta: 0, pred: 0, bw: 0, bh: 0, dc: new Float32Array(0) });
      }
      // One component alone is never interleaved: one block per MCU whatever its factors.
      if (nc === 1) { comps[0].h = 1; comps[0].v = 1; }
    } else if (m === 0xc2 || m === 0xc3 || (m >= 0xc5 && m <= 0xcf && m !== 0xc8 && m !== 0xcc)) {
      return undefined; // progressive, lossless or arithmetic: not handled
    } else if (m === 0xdd) {
      restart = (b[seg] << 8) | b[seg + 1];
    } else if (m === 0xda) { // start of scan: the entropy-coded data follows
      const ns = b[seg];
      if (ns !== comps.length || !W || !H) return undefined; // one interleaved scan only
      // Blocks come in the scan's component order.
      const order: Comp[] = [];
      for (let k = 0; k < ns; k++) {
        const c = comps.find((x) => x.id === b[seg + 1 + k * 2]);
        if (!c) return undefined;
        c.td = b[seg + 2 + k * 2] >> 4; c.ta = b[seg + 2 + k * 2] & 15;
        order.push(c);
      }
      return decodeScan(b, i + 2 + len, comps, order, qt, dcT, acT, W, H, restart);
    } else if (m === 0xd9) break;
    i += 2 + len;
  }
  return undefined;
}

/** Thrown when the entropy data ends early (cut-off file, or a marker in the middle). */
class Truncated extends Error {}

function decodeScan(b: Uint8Array, start: number, comps: Comp[], order: Comp[], qt: Array<Uint16Array | undefined>, dcT: Array<Huff | undefined>, acT: Array<Huff | undefined>, W: number, H: number, restart: number) {
  const hmax = Math.max(...comps.map((c) => c.h)), vmax = Math.max(...comps.map((c) => c.v));
  const mcuX = Math.ceil(W / (8 * hmax)), mcuY = Math.ceil(H / (8 * vmax));
  for (const c of comps) { c.bw = mcuX * c.h; c.bh = mcuY * c.v; c.dc = new Float32Array(c.bw * c.bh); }
  // 32-bit bit buffer over the entropy-coded bytes (0xFF00 is a literal 0xFF; a marker ends the data).
  let p = start, acc = 0, accBits = 0, ended = false;
  const fill = () => {
    while (accBits <= 24) {
      let x = 0;
      if (!ended && p < b.length) {
        x = b[p++];
        if (x === 0xff) {
          const n = b[p];
          if (n === 0) p++;
          else { ended = true; p--; x = 0; } // a marker (RSTn or the end): nothing more in this segment
        }
      } else ended = true;
      acc = ((acc << 8) | x) >>> 0;
      accBits += 8;
    }
  };
  let pad = 0; // bits handed out past the real data
  const need = (n: number) => { if (accBits < n) fill(); };
  const take = (n: number) => {
    need(n);
    const v = (acc >>> (accBits - n)) & ((1 << n) - 1);
    accBits -= n;
    if (ended) { pad += n; if (pad > 64) throw new Truncated(); } // real streams end within one byte of padding
    return v;
  };
  const extend = (v: number, n: number) => (n === 0 ? 0 : v < 1 << (n - 1) ? v - (1 << n) + 1 : v);
  const sym = (t: Huff): number => {
    need(LOOK);
    const e = t.lut[(acc >>> (accBits - LOOK)) & ((1 << LOOK) - 1)];
    if (e) { take(e >> 8); return e & 255; }
    // A longer code: the canonical slow path.
    let code = take(LOOK), len = LOOK;
    while (len < 16 && code > t.maxcode[len]) { code = (code << 1) | take(1); len++; }
    if (code > t.maxcode[len]) throw new Error("bad Huffman code");
    return t.symbols[t.valptr[len] + code - t.mincode[len]];
  };
  try {
    let mcu = 0;
    for (let my = 0; my < mcuY; my++) for (let mx = 0; mx < mcuX; mx++) {
      if (restart && mcu > 0 && mcu % restart === 0) {
        // Restart: drop the rest of the byte, expect RSTn, reset the predictors.
        acc = 0; accBits = 0; pad = 0; ended = false;
        if (!(b[p] === 0xff && b[p + 1] >= 0xd0 && b[p + 1] <= 0xd7)) throw new Truncated();
        p += 2;
        for (const c of comps) c.pred = 0;
      }
      mcu++;
      for (const c of order) {
        const dt = dcT[c.td], at = acT[c.ta], q = qt[c.tq];
        if (!dt || !at || !q) return undefined;
        for (let by = 0; by < c.v; by++) for (let bx = 0; bx < c.h; bx++) {
          const s = sym(dt);
          c.pred += extend(take(s), s);
          // The block's mean: DC × quantiser / 8, level-shifted by 128.
          c.dc[(my * c.v + by) * c.bw + mx * c.h + bx] = (c.pred * q[0]) / 8 + 128;
          for (let k = 1; k < 64;) { // AC coefficients: decoded only to be skipped
            const rs = sym(at), r = rs >> 4, sz = rs & 15;
            if (sz === 0) { if (r === 15) { k += 16; continue; } break; }
            k += r;
            take(sz);
            k++;
          }
        }
      }
    }
  } catch { return undefined; }
  // YCbCr (JFIF) → RGB at the luma grid; chroma sampled at its own (coarser) grid.
  const w = Math.ceil(W / 8), h = Math.ceil(H / 8);
  const rgba = new Uint8Array(w * h * 4);
  const at = (c: Comp, x: number, y: number) => c.dc[Math.min(c.bh - 1, Math.floor((y * c.v) / vmax)) * c.bw + Math.min(c.bw - 1, Math.floor((x * c.h) / hmax))];
  const clamp = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const o = (y * w + x) * 4;
    if (comps.length === 1) { const Y = at(comps[0], x, y); rgba[o] = rgba[o + 1] = rgba[o + 2] = clamp(Y); }
    else {
      const Y = at(comps[0], x, y), Cb = at(comps[1], x, y) - 128, Cr = at(comps[2], x, y) - 128;
      rgba[o] = clamp(Y + 1.402 * Cr); rgba[o + 1] = clamp(Y - 0.344136 * Cb - 0.714136 * Cr); rgba[o + 2] = clamp(Y + 1.772 * Cb);
    }
    rgba[o + 3] = 255;
  }
  return { rgba, w, h };
}
