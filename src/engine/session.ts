/** State and helpers shared by the engine's parts (engine.ts and the modules beside it). */
import type { DecodedImage } from "../decode/types.ts";
import type { WorkingImage } from "../raw/develop.ts";
import { type SceneMaps, GROUPS } from "../neural/scene.ts";
import type { RefinedMaps } from "../refine/refine.ts";
import type { AnalysisReport } from "../analysis/types.ts";
import type { DecisionResult } from "../decision/engine.ts";
import type { Params } from "../decision/params.ts";
import type { FromWorker, UpscaleInfo } from "./protocol.ts";
import { SamSelector } from "../neural/sam.ts";
import type { Rect } from "../retouch/geometry.ts";
import type { Inpainter } from "../retouch/inpaint.ts";
import { isPhone } from "../device.ts";
import { selectKey } from "../layers/model.ts";
import { inverse, mul } from "../color/mat3.ts";
import { P3_D65, SRGB, rgbToXYZ } from "../color/spaces.ts";

// The check's code is loaded on the first check (src/analysis/check*.ts): not part of opening a photo.
export const checkCode = () => Promise.all([import("../analysis/check.ts"), import("../analysis/checkFix.ts")]);

/** The scene's light level (EV at ISO 100) from the photo's exposure settings, when they are physically sensible (resized or re-saved files can carry junk). */
export function sceneEV(m: { fNumber?: number; exposureTime?: number; iso?: number }): number | undefined {
  const ok = m.fNumber && m.exposureTime && m.iso && m.fNumber >= 0.9 && m.fNumber <= 32 && m.exposureTime >= 1 / 32000 && m.exposureTime <= 60 && m.iso >= 12 && m.iso <= 409600;
  const ev = ok ? Math.log2((m.fNumber! * m.fNumber!) / m.exposureTime!) - Math.log2(m.iso! / 100) : NaN;
  return ev >= -6 && ev <= 21 ? ev : undefined;
}

/** Linear Display P3 → linear sRGB (as P3_TO_SRGB in common.wgsl). */
export const P3_TO_SRGB = mul(inverse(rgbToXYZ(SRGB)), rgbToXYZ(P3_D65));

export type Post = (m: FromWorker, transfer?: Transferable[]) => void;

export interface Session {
  name: string;
  /** The opened file (a reference, not a copy): Check reads the camera's own rendering from it. */
  file: File;
  /** The camera's own rendering (the JPEG inside a DNG) at 256 px, read on the first check; null = none. */
  cameraRef?: { rgba: Uint8Array; w: number; h: number } | null;
  decoded: DecodedImage;
  work: WorkingImage;
  denoised: GPUTexture; // === work.tex until neural restoration ran
  gain: number;
  scene: SceneMaps;
  maps: RefinedMaps;
  report: AnalysisReport;
  decision: DecisionResult;
  params: Params;
  /**
   * Exposure calibration against the camera's rendering (DNG preview): the first
   * final previews are measured and the automatic exposure corrected (≤ 2 rounds),
   * unless the exposure was changed by then.
   */
  calib?: { ref: number[]; refHi?: number[]; rounds: number; black: boolean; chroma?: number; chroma95?: number; color?: boolean; contrast?: boolean };
  /** Preview proxy. `owned` is false when it aliases the working textures (image ≤ preview size). */
  proxy?: { base: GPUTexture; denoised: GPUTexture; w: number; h: number; owned: boolean };
  /** Quarter-pixel proxy used while a slider is being dragged. */
  draft?: { base: GPUTexture; denoised: GPUTexture; w: number; h: number };
  distCPU?: { w: number; h: number; data: Float32Array };
  lightLinear: [number, number, number];
  /** Working pixels per original working pixel along each axis: 2 after upscaling. */
  scale: 1 | 2;
  /** Apple's skin matte (ProRAW), uploaded once and sampled while rendering. */
  skin?: GPUTexture;
  /** The quality analysis and what the upscale stage did with it. */
  upscale?: UpscaleInfo;
  /** Tap-to-select: the photo's selector, masks by selection, and the texture the renderer samples. */
  sel?: Selections;
  /**
   * Magic brush: the strokes filled in so far, in order — each with the pixels it replaced
   * (rgba16 of the working texture, and of the restored one when separate), so undoing it
   * puts them back exactly — and the network that fills them.
   */
  retouch?: { applied: Array<{ key: string; rect: Rect; before: Uint16Array[] }>; failed?: string; painter?: Inpainter };
  /** A moving object's background plate (objectMotion.ts): for which mask, and the texture the renderer reads. */
  motionPlate?: { key: string; tex?: GPUTexture };
}

export interface Selections {
  sam: SamSelector;
  /** Masks at the guide resolution, by selectKey (most recently used last). */
  cache: Map<string, Uint8Array>;
  /** Selections that failed (not retried on every render). */
  failed: Set<string>;
  /** The selections in the texture, one array layer each, in order. */
  keys: string[];
  tex?: GPUTexture;
  version: number;
  /** The photo's luminance at the guide resolution (edge snapping). */
  guide?: Float32Array;
}

/** Default blur strength whenever depth of field is switched on (scene-independent, by preference). */
export const DEFAULT_DOF_STRENGTH = 0.5;

/** Largest working image (MP) the 2× stage may produce: a 2× texture must fit next to everything else. */
export const UPSCALE_MAX_MP = () => (isMobile() ? 16 : 48);

/** Long edge of the image the analysis networks see. */
export const ANALYSIS_LONG = 1036;
/** Phones: exactly one 512 px segmentation window, no sliding. */
export const ANALYSIS_LONG_LIGHT = 512;

export const isMobile = isPhone;

/** Kinds of subject kept whole in focus (their segmentation group index). */
export function protectGroup(kind: string | undefined): number | undefined {
  return kind === "person" || kind === "animal" || kind === "vehicle" ? GROUPS.indexOf(kind) : undefined;
}

/** Gain that puts the 60th-percentile luminance of the analysis image at 0.18 (clamped). */
export function exposureGain(rgba: Float32Array): number {
  const n = rgba.length / 4;
  const ys = new Float32Array(n);
  for (let i = 0; i < n; i++) ys[i] = Math.max(0, 0.2627 * rgba[i * 4] + 0.678 * rgba[i * 4 + 1] + 0.0593 * rgba[i * 4 + 2]);
  ys.sort();
  const p60 = ys[Math.floor(n * 0.6)] || 1e-4;
  const p99 = ys[Math.floor(n * 0.99)] || 1;
  let k = 0.18 / Math.max(p60, 1e-5);
  k = Math.min(k, 1.6 / Math.max(p99, 1e-5), 32);
  return Math.max(0.25, k);
}

/** The Check's fixes were being worked out for settings (or a photo) that changed since. */
export class StaleCheck extends Error {}

/**
 * Where the photo recedes to (0…1 of width and height): the centre of its farthest 5 %
 * (motion blur "by depth" streaks away from it, as when moving into the scene).
 */
export function vanishingPoint(d: { w: number; h: number; data: Float32Array }): [number, number] {
  const sorted = Float32Array.from(d.data).sort();
  const cut = sorted[Math.floor(sorted.length * 0.95)] ?? 1;
  let sx = 0, sy = 0, n = 0;
  for (let y = 0; y < d.h; y++) for (let x = 0; x < d.w; x++) {
    if (d.data[y * d.w + x] >= cut) { sx += x + 0.5; sy += y + 0.5; n++; }
  }
  return n ? [sx / n / d.w, sy / n / d.h] : [0.5, 0.5];
}
