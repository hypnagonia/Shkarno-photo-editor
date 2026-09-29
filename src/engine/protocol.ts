import type { MaskHist, ToneEqDetail } from "../tone/toneEq.ts";
import type { FixChange } from "../analysis/checkFix.ts";
import type { CheckItem } from "../analysis/check.ts";
import type { Region } from "../decision/params.ts";
import type { AnalysisLevel } from "../neural/scene.ts";
import type { AutoCurveBands } from "../decision/autoCurves.ts";
import type { Decision, Params } from "../decision/params.ts";
import type { LookProfile } from "../looks/profile.ts";
import type { ColorStats } from "../looks/palette.ts";
import type { ImageQualityReport, UpscaleMode, UpscaleReasonCode } from "../analysis/quality.ts";
export type { UpscaleMode };

export type ExportFormat = "jpeg" | "jpeg-hdr" | "heic" | "tiff16" | "dng";

export interface Capabilities {
  webgpu: boolean;
  f16: boolean;
  backend: "webgpu" | "wasm" | "cpu";
  gpu: string;
  heicEncode: boolean;
  crossOriginIsolated: boolean;
  threads: number;
}

export interface StageProfile {
  stage: string;
  ms: number;
  gpuLiveMB: number;
  gpuPeakMB: number;
  heapMB?: number;
  note?: string;
}

export interface Summary {
  file: string;
  format: string;
  source: string;
  width: number;
  height: number;
  working: { width: number; height: number; factor: number };
  meta: Record<string, string | number>;
  coverage: Record<string, number>;
}

/** The upscale stage's decision and outcome (kept on the session, mirrored to the UI). */
export interface UpscaleInfo {
  state: "skipped" | "pending" | "running" | "applied" | "failed" | "cancelled";
  upscaleApplied: boolean;
  upscaleFactor: 1 | 2;
  upscaleReason: string;
  code: UpscaleReasonCode;
  vars: Record<string, string | number>;
  /** Analyzer measurements (qualityMetrics) and scores. */
  report: ImageQualityReport;
  /** Working size after upscaling. */
  width?: number;
  height?: number;
}

export type ToWorker =
  | { type: "init"; base: string; forceCpu?: boolean; /** Behave as on a phone (local autotest only). */ phone?: boolean }
  | { type: "open"; file: File; resolution: "auto" | "full" | "half"; autoExposure: boolean; autoDof: boolean; upscale: UpscaleMode; /** Scene analysis on the CPU (it crashed this device's GPU before). */ safeAnalysis?: boolean; /** Less scene analysis: the tab died during it on this device before. */ analysis?: AnalysisLevel }
  /** Which tone-equalizer zone (0 = −8 EV … 8 = 0 EV) the photo is in at x, y (0…1). */
  | { type: "toneEqZone"; x: number; y: number }
  /** The tone equalizer's mask histograms as the photo renders now. */
  | { type: "toneEqHist" }
  | { type: "upscale-now" }
  | { type: "params"; params: Params; draft?: boolean }
  | { type: "view"; view: 0 | 1 | 2 | 4 | 5 | 6 | 9 | 11; before?: boolean; region?: number; range?: [number, number] }
  | { type: "focus"; action: "toggle"; x: number; y: number }
  | { type: "focus"; action: "move"; index: number; x: number; y: number }
  | { type: "focus"; action: "clear" }
  | { type: "preview-zoom"; long: number }
  | { type: "export"; format: ExportFormat; quality: number; space: "srgb" | "p3"; stripRows?: number }
  | { type: "importLook"; name: string; text: string }
  | { type: "thumbs"; profiles: LookProfile[]; long: number }
  | { type: "palette" }
  /** Check the finished photo for technical mistakes. */
  | { type: "check" }
  /** GPU memory now and the peak since the last ask (the local memory guard). */
  | { type: "mem" }
  /** Where the light probably is (a lens flare's default position). */
  | { type: "brightest" }
  /** What is under a tap on the photo (x, y: 0…1 of the picture), for building a mask from it. */
  | { type: "pick"; x: number; y: number; /** The layer (index among the live layers) whose mask the tap edits. */ layer?: number; /** A tap selects an object (tap-to-select). */ object?: boolean }
  | { type: "reference"; file: File; mode: "create" | "match"; amount: number }
  | { type: "preview-size"; long: number }
  /** The page's canvas, handed over: previews are drawn into it on the GPU (no readback per frame). */
  | { type: "canvas"; canvas: OffscreenCanvas };

export type FromWorker =
  | { type: "ready"; caps: Capabilities; looks: Array<{ id: string; name: string; description: string }> }
  | { type: "progress"; stage: string; detail?: string; frac?: number }
  /** `data` absent: the frame is already on the page's canvas (GPU display). */
  | { type: "preview"; width: number; height: number; data?: ArrayBuffer; space: "p3" | "srgb"; final: boolean; ms: number }
  /** Whether the handed-over canvas could be set up for GPU display (else the page draws previews itself). */
  | { type: "display"; ok: boolean; message?: string }
  | { type: "analysis"; summary: Summary; decisions: Decision[]; auto: Params; params: Params; dof: { justified: boolean; focus: number; strength: number; reason: string; x?: number; y?: number; zoneEdges?: number[]; zones?: Array<{ share: number; label: string; lo: number; hi: number }>; bands?: Array<{ share: number; label: string; lo: number; hi: number }> }; exposureSuggestion: number; autoCurves?: AutoCurveBands; cellCoverage?: Record<string, number>; /** Why this photo has no depth map (it opened without one), if so. */ noDepth?: string; /** The camera's own white balance (the neutral one). */ cameraWB?: { temp: number; tint: number } }
  | { type: "params"; params: Params }
  /** Automatic exposure corrected after measuring the first preview against the camera's rendering. */
  | { type: "exposureCalibrated"; exposure: number; note: string }
  /** Black point matched to the camera's rendering: the master curve of the automatic "Black point" layer. */
  | { type: "blackPointMatched"; points: Array<{ x: number; y: number }>; note: string }
  /** Intensity histograms of the rendered preview (previewHist.ts), for the curve boxes. */
  | { type: "histograms"; data: Float32Array }
  | { type: "log"; text: string }
  | { type: "profile"; stages: StageProfile[] }
  | { type: "exported"; blob: Blob; name: string; ms: number }
  | { type: "looks"; looks: Array<{ id: string; name: string; description: string }> }
  | { type: "thumbs"; items: Array<{ id: string; width: number; height: number; data: ArrayBuffer }> }
  | { type: "palette"; stats: ColorStats }
  | { type: "pick"; info?: PickInfo }
  | { type: "check"; items: CheckItem[]; rgba: Uint8Array; w: number; h: number }
  | { type: "mem"; liveMB: number; peakMB: number }
  | { type: "brightest"; x: number; y: number }
  /** A worked-out fix for a finding of the last check; `done` when there are no more. */
  | { type: "checkFix"; id?: CheckItem["id"]; fix?: FixChange[]; partial?: boolean; done?: boolean }
  | { type: "lookProfile"; profile: LookProfile; reference: ColorStats; message: string }
  | { type: "gpu-lost"; reason: string }
  | { type: "upscale"; info: UpscaleInfo }
  | { type: "toneEqZone"; zone?: number }
  | { type: "toneEqHist"; hist?: Record<ToneEqDetail, MaskHist> }
  | { type: "error"; message: string; stage?: string };

/** What the photo is at a tapped point: the ingredients of "this object / this colour / this far". */
export interface PickInfo {
  x: number; y: number;
  /** The most likely region there (skin where Apple's skin matte says so) and its probability. */
  region: Region; prob: number;
  /** Distance there (0 = nearest … 1 = farthest) and the depth range of the object under it. */
  dist: number; range: [number, number];
  /** OkLab of the colour before the layers (what colour masks compare with). */
  color: [number, number, number];
  /** How much the layer's mask already covers the tap (0…1): a tap on something selected removes it. */
  inMask?: number;
  /** The selection (selectKey) the tapped object already is, when it is one: that piece is deselected. */
  sameAs?: string;
}
