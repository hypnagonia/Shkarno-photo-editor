/**
 * Contrast equalizer: local contrast by the size of detail. The image (after the tone
 * rendering, in OkLab) is split into detail layers of doubling size with an
 * edge-avoiding à-trous wavelet transform — 1, 2, 4 … 64 px at full resolution, plus
 * the residual — and each layer's contrast is raised or lowered on its own, for
 * lightness and for colour. The idea of darktable's contrast equalizer (atrous); this
 * implementation is our own.
 *
 *   out = s0 + Σ_j gain_j · (s_j − s_{j+1})
 *
 * s_{j+1} = s_j smoothed by a 5×5 B-spline kernel spread 2^j px apart, with weights
 * that fall off across a lightness step larger than `edges` allows (no halos along
 * edges). Bands finer than a pixel of a reduced preview are not shown there (as in any
 * editor) but apply at full size.
 */

export const BANDS = 7;
/** Detail size (full-resolution px) of each band, fine to coarse. */
export const BAND_PX = Array.from({ length: BANDS }, (_, j) => 2 ** j);

export interface ContrastEq {
  enabled: boolean;
  /** Lightness contrast per band, fine (1 px) to coarse (64 px): −1 removes it, +1 doubles it. */
  luma: number[];
  /** Colour contrast per band, the same way. */
  chroma: number[];
  /** Edge awareness 0…1: how strongly the smoothing stops at lightness steps (halos). */
  edges: number;
}

export const MAX_BAND_GAIN = 1;
export const neutralContrastEq = (): ContrastEq => ({ enabled: true, luma: new Array(BANDS).fill(0), chroma: new Array(BANDS).fill(0), edges: 0.6 });
export const contrastEqActive = (c: ContrastEq | undefined): c is ContrastEq =>
  !!c && c.enabled && [...c.luma, ...c.chroma].some((g) => Math.abs(g) > 1e-4);

/** Lightness difference (OkLab L, 0…1) at which a neighbour's weight has halved: large = no edge awareness. */
export const edgeSigma = (edges: number) => 0.5 * Math.pow(0.03 / 0.5, Math.min(1, Math.max(0, edges)));

/**
 * The levels a render at `scale` (its pixels per full-resolution pixel) computes:
 * each band's spacing in render pixels, and whether it is visible there at all.
 * Bands finer than half a render pixel fold into the image as is.
 */
export function levels(scale: number): Array<{ band: number; step: number }> {
  return BAND_PX.map((px, band) => ({ band, step: px * scale })).filter((l) => l.step >= 0.5);
}

/** Rows of margin a strip needs above and below (the kernel's reach over all levels). */
export const stripApron = (scale: number) => Math.ceil(levels(scale).reduce((a, l) => a + 2 * l.step, 0)) + 2;

export type ContrastEqPresetId = "clarity" | "detail" | "sharpen" | "soften" | "bloom" | "deblur" | "colorPop";
/** Starting points: luma and chroma, fine → coarse. */
export const CONTRAST_EQ_PRESETS: Array<{ id: ContrastEqPresetId; luma: number[]; chroma: number[] }> = [
  { id: "clarity", luma: [0, 0.05, 0.12, 0.25, 0.35, 0.3, 0.15], chroma: [0, 0, 0, 0.05, 0.1, 0.1, 0.05] },
  { id: "detail", luma: [0.1, 0.25, 0.3, 0.2, 0.1, 0, 0], chroma: [0, 0, 0, 0, 0, 0, 0] },
  { id: "sharpen", luma: [0.45, 0.3, 0.1, 0, 0, 0, 0], chroma: [0, 0, 0, 0, 0, 0, 0] },
  { id: "soften", luma: [-0.35, -0.3, -0.2, -0.1, 0, 0, 0], chroma: [-0.3, -0.3, -0.2, 0, 0, 0, 0] },
  { id: "bloom", luma: [0, 0, 0, -0.1, -0.25, -0.35, -0.4], chroma: [0, 0, 0, 0, -0.1, -0.15, -0.2] },
  { id: "deblur", luma: [0.3, 0.45, 0.35, 0.15, 0, 0, 0], chroma: [0, 0, 0, 0, 0, 0, 0] },
  { id: "colorPop", luma: [0, 0, 0, 0, 0.05, 0.05, 0], chroma: [0, 0.1, 0.25, 0.35, 0.35, 0.25, 0.15] },
];
