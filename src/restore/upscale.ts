/**
 * 2× neural upscaling (Swin2SR lightweight ×2), tiled and incremental.
 *
 *   in:  restored working texture W×H (scene-linear Rec.2020, rgba16float)
 *   out: new working texture 2W×2H in the same representation
 *
 * Colour around the model. Swin2SR was trained on display-referred sRGB
 * images, so scene-linear values are never fed to it directly:
 *
 *   linear × gain → extended Reinhard (white 4) → sRGB curve → Swin2SR ×2
 *   → exact inverse (sRGB⁻¹, Reinhard⁻¹) ÷ gain → linear working
 *
 * The transform is neutral (per channel, no look) and exactly invertible, so
 * where the network changes nothing the round trip returns the input. Values
 * near and above the Reinhard white (≥ ~3 × normalised white) keep the
 * original scene-linear data, bilinearly upsampled; alpha is upsampled too.
 *
 * Tiles are 256 px (the graph is exported with a fixed input size) with 24 px
 * overlap, blended with separable feather weights in output space (exact
 * partition of unity → no seams). Output rows are accumulated in a rolling CPU
 * strip of 2 × 256 rows and written to the texture as soon as no later tile
 * touches them, so no full-resolution float buffer ever exists on the CPU.
 * `step()` processes tiles for a time budget and returns, so the engine can
 * interleave preview renders while a phone works through the tiles.
 */
import type { Gpu } from "../gpu/gpu.ts";
import type * as Ort from "onnxruntime-web";
import { floatsToHalves, halvesToFloats } from "../gpu/half.ts";
import { ort } from "../neural/ort.ts";
import { WHITE, fromDisplay, toDisplay } from "./display.ts";

interface TileGrid { xs: number[]; ys: number[] }
/** Tile origins covering w × h with overlapping tiles (the last one flush with the edge). */
function tileGrid(w: number, h: number, tile: number, overlap: number): TileGrid {
  const axis = (n: number) => {
    if (n <= tile) return [0];
    const stride = tile - overlap;
    const out: number[] = [];
    for (let p = 0; p + tile < n; p += stride) out.push(p);
    out.push(n - tile);
    return out;
  };
  return { xs: axis(w), ys: axis(h) };
}

export const SR_TILE = 256;
export const SR_OVERLAP = 24;
const S = 2;
/**
 * Self-test of a freshly created session: a smooth gradient must come back
 * smooth and at the same brightness. Catches backends that run the graph
 * but compute it wrongly (e.g. a checkerboard from reduced precision), which
 * a range check on the output would miss. One tile; returns the failure or "".
 */
export async function probeUpscaler(session: Ort.InferenceSession): Promise<string> {
  const T = SR_TILE, O = 2 * T;
  const x = new Float32Array(3 * T * T);
  let mIn = 0;
  for (let c = 0; c < 3; c++) for (let y = 0; y < T; y++) for (let i = 0; i < T; i++) {
    const v = 0.2 + 0.3 * (i / T) + 0.1 * Math.sin(y / 20) + 0.05 * c;
    x[c * T * T + y * T + i] = v;
    mIn += v;
  }
  mIn /= 3 * T * T;
  const tIn = new ort.Tensor("float32", x, [1, 3, T, T]);
  const res = await session.run({ [session.inputNames[0]]: tIn });
  const tOut = res[session.outputNames[0]];
  const o = (await tOut.getData()) as Float32Array;
  tIn.dispose();
  tOut.dispose();
  let d = 0, n = 0, mOut = 0;
  for (let c = 0; c < 3; c++) for (let y = 8; y < O - 8; y++) for (let i = 8; i < O - 9; i++) {
    const k = c * O * O + y * O + i;
    d += Math.abs(o[k] - 0.5 * (o[k - 1] + o[k + 1]));
    mOut += o[k];
    n++;
  }
  d /= n; mOut /= n;
  if (!Number.isFinite(d) || d > 0.002) return `probe output not smooth (${d.toFixed(4)})`;
  if (Math.abs(mOut - mIn) > 0.02) return `probe brightness off (${mIn.toFixed(3)} → ${mOut.toFixed(3)})`;
  return "";
}

export interface UpscaleProgress { done: number; total: number; msPerTile: number }

export class UpscaleJob {
  readonly out: GPUTexture;
  readonly total: number;
  private grid: TileGrid;
  private r = 0;
  private c = 0;
  private done = 0;
  private netMs = 0;
  private strip: Float32Array; // 2W × 2T × (r, g, b, a, weight)
  private readonly OW: number;
  private readonly SH = S * SR_TILE;

  constructor(
    private gpu: Gpu,
    private session: Ort.InferenceSession,
    readonly src: GPUTexture,
    readonly W: number,
    readonly H: number,
    private gain: number,
  ) {
    this.grid = tileGrid(W, H, SR_TILE, SR_OVERLAP);
    this.total = this.grid.xs.length * this.grid.ys.length;
    this.OW = S * W;
    this.out = gpu.tex("upscaled", S * W, S * H, "rgba16float");
    this.strip = new Float32Array(this.OW * this.SH * 5);
  }

  get progress(): UpscaleProgress { return { done: this.done, total: this.total, msPerTile: this.done ? this.netMs / this.done : 0 }; }

  /** Processes tiles for about `budgetMs`; true when the whole image is done. */
  async step(budgetMs: number): Promise<boolean> {
    const t0 = performance.now();
    const { xs, ys } = this.grid;
    while (this.r < ys.length) {
      await this.tile(xs[this.c], ys[this.r]);
      this.done++;
      if (++this.c === xs.length) {
        this.c = 0;
        this.flushRow();
        this.r++;
      }
      if (performance.now() - t0 > budgetMs) break;
    }
    if (this.r < ys.length) return false;
    await this.gpu.device.queue.onSubmittedWorkDone();
    return true;
  }

  /** Frees the CPU strip; the output texture belongs to the caller (or `release`). */
  finish() { this.strip = new Float32Array(0); }
  release() { this.finish(); this.gpu.release(this.out); }

  private async tile(tx: number, ty: number) {
    const T = SR_TILE, { W, H, gain } = this;
    const tw = Math.min(T, W), th = Math.min(T, H);
    const px = halvesToFloats(new Uint16Array(await this.gpu.readTexture(this.src, tx, ty, tw, th, 8)));
    // NCHW display-referred input; beyond a small image's edge, replicate.
    const plane = T * T;
    const input = new Float32Array(3 * plane);
    for (let y = 0; y < T; y++) {
      const sy = Math.min(th - 1, y);
      for (let x = 0; x < T; x++) {
        const i = (sy * tw + Math.min(tw - 1, x)) * 4;
        for (let ch = 0; ch < 3; ch++) input[ch * plane + y * T + x] = toDisplay(px[i + ch] * gain);
      }
    }
    const tn = performance.now();
    const name = this.session.inputNames[0];
    const tIn = new ort.Tensor("float32", input, [1, 3, T, T]);
    const res = await this.session.run({ [name]: tIn });
    const tOut = res[this.session.outputNames[0]];
    const y2 = (await tOut.getData()) as Float32Array;
    tIn.dispose();
    tOut.dispose();
    // A backend that produces garbage (numerical failure) must not reach the photo.
    for (let i = 0; i < y2.length; i += 97) if (!Number.isFinite(y2[i]) || y2[i] < -0.5 || y2[i] > 1.5) throw new Error("upscaler produced invalid output");
    this.netMs += performance.now() - tn;

    // Accumulate into the rolling strip (row 0 of the strip = output row S·ys[r]).
    const OT = S * T, oplane = OT * OT;
    const ow = S * tw, oh = S * th;
    const fx0 = tx > 0, fx1 = tx + T < W, fy0 = ty > 0, fy1 = ty + T < H;
    const ramp = S * SR_OVERLAP;
    const top = S * this.grid.ys[this.r];
    const hiA = 0.75 * WHITE, hiB = 0.95 * WHITE;
    for (let y = 0; y < oh; y++) {
      const wy = Math.max(1e-3, Math.min(1, fy0 ? (y + 0.5) / ramp : 1, fy1 ? (oh - y - 0.5) / ramp : 1));
      const row = S * ty + y - top;
      // Source position (bilinear) for highlights and alpha.
      const sy = Math.min(th - 1, Math.max(0, (y + 0.5) / S - 0.5));
      const y0 = Math.floor(sy), y1 = Math.min(th - 1, y0 + 1), fy = sy - y0;
      for (let x = 0; x < ow; x++) {
        const wx = Math.max(1e-3, Math.min(1, fx0 ? (x + 0.5) / ramp : 1, fx1 ? (ow - x - 0.5) / ramp : 1));
        const w = wx * wy;
        const sx = Math.min(tw - 1, Math.max(0, (x + 0.5) / S - 0.5));
        const x0 = Math.floor(sx), x1 = Math.min(tw - 1, x0 + 1), fx = sx - x0;
        const i00 = (y0 * tw + x0) * 4, i01 = (y0 * tw + x1) * 4, i10 = (y1 * tw + x0) * 4, i11 = (y1 * tw + x1) * 4;
        const bil = (ch: number) => (px[i00 + ch] * (1 - fx) + px[i01 + ch] * fx) * (1 - fy) + (px[i10 + ch] * (1 - fx) + px[i11 + ch] * fx) * fy;
        const b0 = bil(0), b1 = bil(1), b2 = bil(2);
        const hv = Math.max(b0, b1, b2) * gain;
        const keep = hv <= hiA ? 0 : hv >= hiB ? 1 : ((hv - hiA) / (hiB - hiA)) ** 2 * (3 - 2 * ((hv - hiA) / (hiB - hiA)));
        const o = (row * this.OW + S * tx + x) * 5;
        const k = y * OT + x;
        const n0 = fromDisplay(y2[k]) / gain, n1 = fromDisplay(y2[oplane + k]) / gain, n2 = fromDisplay(y2[2 * oplane + k]) / gain;
        this.strip[o] += w * (n0 + (b0 - n0) * keep);
        this.strip[o + 1] += w * (n1 + (b1 - n1) * keep);
        this.strip[o + 2] += w * (n2 + (b2 - n2) * keep);
        this.strip[o + 3] += w * bil(3);
        this.strip[o + 4] += w;
      }
    }
  }

  /** Writes the output rows no later tile row touches, then shifts the strip up. */
  private flushRow() {
    const { ys } = this.grid;
    const top = S * ys[this.r];
    const next = this.r + 1 < ys.length ? S * ys[this.r + 1] : S * this.H;
    const rows = next - top;
    const OW = this.OW;
    const px = new Float32Array(OW * rows * 4);
    for (let i = 0; i < OW * rows; i++) {
      const o = i * 5, wsum = this.strip[o + 4] || 1;
      px[i * 4] = this.strip[o] / wsum;
      px[i * 4 + 1] = this.strip[o + 1] / wsum;
      px[i * 4 + 2] = this.strip[o + 2] / wsum;
      px[i * 4 + 3] = this.strip[o + 3] / wsum;
    }
    this.gpu.device.queue.writeTexture({ texture: this.out, origin: { x: 0, y: top } }, floatsToHalves(px), { bytesPerRow: OW * 8, rowsPerImage: rows }, { width: OW, height: rows });
    // Keep the unfinished tail (overlap with the next tile row) at the top.
    this.strip.copyWithin(0, rows * OW * 5);
    this.strip.fill(0, (this.SH - rows) * OW * 5);
  }
}
