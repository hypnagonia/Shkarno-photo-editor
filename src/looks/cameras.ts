/**
 * Camera looks: an iPhone photo rendered the way a well-known camera and lens
 * would render it. Three parts, each switchable:
 *
 *   colour  the maker's colour science as a look profile (tone curve, highlight
 *           shoulder, hue / saturation shaping, colour balance) — authored from each
 *           style's widely described character, not the maker's own data (which is
 *           proprietary): "styled after", never a certified copy;
 *   lens    background blur from the lens's optics (focal length, aperture, sensor
 *           size, a typical subject distance) and its wide-open vignette;
 *   sensor  a big sensor's finer grain, gentler sharpening and less of the phone's
 *           HDR flattening (local tone compression).
 *
 * What cannot be done is not faked: perspective stays the phone's, and nothing
 * claims the photo came from that camera (the export's metadata stays the iPhone's).
 */
import { makeProfile, type LookProfile } from "./profile.ts";
import type { Params } from "../decision/params.ts";

export type SensorSize = "ff" | "apsc" | "mf";
/** Sensor long side (mm). */
const SENSOR_MM: Record<SensorSize, number> = { ff: 36, apsc: 23.5, mf: 43.8 };

export interface CameraLook {
  id: string;
  maker: string;
  model: string;
  /** The maker's name for the colour style (Standard, Classic Chrome, …). */
  style: string;
  lens: { name: string; focal: number; aperture: number; sensor: SensorSize; vignette: number };
  sensor: { grain: number; grainSize: number; sharpen: number; localCompression: number };
  profile: LookProfile;
}

/**
 * Background blur of the lens as the app's depth-of-field strength (0…1): the
 * blur disc of a background far behind a subject at `distance` metres, relative
 * to the frame (c = f² / (N · (s − f)), radius over the sensor's long side),
 * scaled to the depth-of-field's maximum radius (2.2 % of the long side).
 */
export function lensBlurStrength(focal: number, aperture: number, sensor: SensorSize, distance = typicalDistance(focal, sensor)): number {
  const s = distance * 1000;
  const c = (focal * focal) / (aperture * Math.max(1, s - focal));
  const radius = c / 2 / SENSOR_MM[sensor];
  return Math.min(1, Math.round((radius / 0.022) * 100) / 100);
}
/** Where people usually stand from the subject with this field of view (m): wider lenses, closer. */
export function typicalDistance(focal: number, sensor: SensorSize): number {
  const eq = focal * (36 / SENSOR_MM[sensor]);
  return eq <= 28 ? 1.2 : eq <= 40 ? 1.5 : eq <= 60 ? 2 : 2.6;
}

const look = (id: string, name: string, p: Omit<Parameters<typeof makeProfile>[0], "id" | "name">) => makeProfile({ ...p, id: `camera-${id}`, name } as Parameters<typeof makeProfile>[0]);

export const CAMERA_LOOKS: CameraLook[] = [
  {
    id: "canon-r5-standard", maker: "Canon", model: "EOS R5", style: "Standard",
    lens: { name: "RF 50mm f/1.2L", focal: 50, aperture: 1.2, sensor: "ff", vignette: -0.18 },
    sensor: { grain: 0.05, grainSize: 0.3, sharpen: 0.35, localCompression: 0.2 },
    // Warm and rich: rosy skin, deep slightly magenta reds, yellowish greens, clean blues.
    profile: look("canon-r5-standard", "Canon Standard", {
      category: "warm cinematic",
      tone: { contrast: 0.16, rolloff: 0.62, highlightCompression: 0.08, blackPoint: -0.01 },
      hsl: { red: { hue: -3, sat: 0.1, lum: -0.02 }, orange: { hue: -2, sat: 0.06, lum: 0.02 }, yellow: { hue: -4, sat: 0.06, lum: 0 }, green: { hue: -5, sat: 0.04, lum: 0 }, blue: { hue: -3, sat: 0.08, lum: -0.02 }, magenta: { hue: 0, sat: 0.06, lum: 0 } },
      colorBalance: { shadows: [0, 0, 0], midtones: [0.008, 0.002, -0.008], highlights: [0.006, 0.002, -0.006] },
      saturation: { global: 1.08, knee: 0.3, compression: 0.25 },
    }),
  },
  {
    id: "canon-r5-portrait", maker: "Canon", model: "EOS R5", style: "Portrait",
    lens: { name: "RF 85mm f/1.2L", focal: 85, aperture: 1.2, sensor: "ff", vignette: -0.12 },
    sensor: { grain: 0.04, grainSize: 0.3, sharpen: 0.22, localCompression: 0.15 },
    // Softer contrast, pinker and lighter skin, calmer everything else.
    profile: look("canon-r5-portrait", "Canon Portrait", {
      category: "portrait-neutral",
      tone: { contrast: 0.06, rolloff: 0.72, highlightCompression: 0.12, shadowLift: 0.03 },
      hsl: { red: { hue: -4, sat: 0.04, lum: 0.02 }, orange: { hue: -3, sat: -0.04, lum: 0.05 }, yellow: { hue: -2, sat: -0.04, lum: 0.02 }, green: { hue: -3, sat: -0.06, lum: 0 } },
      colorBalance: { shadows: [0, 0, 0], midtones: [0.01, 0.002, -0.004], highlights: [0.008, 0.004, 0] },
      saturation: { global: 1.0, knee: 0.26, compression: 0.3 },
    }),
  },
  {
    id: "nikon-z8-standard", maker: "Nikon", model: "Z8", style: "Standard",
    lens: { name: "Z 50mm f/1.8 S", focal: 50, aperture: 1.8, sensor: "ff", vignette: -0.1 },
    sensor: { grain: 0.05, grainSize: 0.3, sharpen: 0.4, localCompression: 0.2 },
    // Neutral and crisp: accurate skin, yellow-green foliage, slightly cool whites.
    profile: look("nikon-z8-standard", "Nikon Standard", {
      category: "clean digital",
      tone: { contrast: 0.18, rolloff: 0.55, highlightCompression: 0.05 },
      hsl: { red: { hue: 2, sat: 0.05, lum: 0 }, orange: { hue: 1, sat: 0.02, lum: 0 }, yellow: { hue: 4, sat: 0.08, lum: 0.02 }, green: { hue: -2, sat: 0.1, lum: 0 }, blue: { hue: 2, sat: 0.06, lum: -0.02 } },
      colorBalance: { shadows: [0, 0, 0], midtones: [-0.004, 0.002, 0.004], highlights: [-0.003, 0, 0.004] },
      saturation: { global: 1.1, knee: 0.32, compression: 0.2 },
    }),
  },
  {
    id: "nikon-z8-landscape", maker: "Nikon", model: "Z8", style: "Landscape",
    lens: { name: "Z 24–70mm f/2.8 S at 35mm", focal: 35, aperture: 2.8, sensor: "ff", vignette: -0.08 },
    sensor: { grain: 0.04, grainSize: 0.25, sharpen: 0.5, localCompression: 0.25 },
    // Punchy: deep blues, dense greens, stronger contrast.
    profile: look("nikon-z8-landscape", "Nikon Landscape", {
      category: "landscape",
      tone: { contrast: 0.24, rolloff: 0.5, highlightCompression: 0.06, blackPoint: -0.01 },
      hsl: { yellow: { hue: 3, sat: 0.1, lum: 0 }, green: { hue: -3, sat: 0.18, lum: -0.03 }, cyan: { hue: 0, sat: 0.12, lum: -0.02 }, blue: { hue: 3, sat: 0.18, lum: -0.05 } },
      saturation: { global: 1.16, knee: 0.3, compression: 0.3 },
    }),
  },
  {
    id: "sony-a7iv-standard", maker: "Sony", model: "A7 IV", style: "Standard",
    lens: { name: "FE 35mm f/1.4 GM", focal: 35, aperture: 1.4, sensor: "ff", vignette: -0.15 },
    sensor: { grain: 0.05, grainSize: 0.3, sharpen: 0.38, localCompression: 0.2 },
    // Clean and slightly cool, a touch of green in the whites, yellower skin.
    profile: look("sony-a7iv-standard", "Sony Standard", {
      category: "clean digital",
      tone: { contrast: 0.12, rolloff: 0.55, highlightCompression: 0.05 },
      hsl: { orange: { hue: 3, sat: 0, lum: 0.01 }, yellow: { hue: 2, sat: 0.04, lum: 0 }, green: { hue: 2, sat: 0.04, lum: 0 }, blue: { hue: 2, sat: 0.06, lum: 0 } },
      colorBalance: { shadows: [-0.004, 0.002, 0.004], midtones: [-0.005, 0.004, 0.004], highlights: [-0.004, 0.003, 0.002] },
      saturation: { global: 1.05, knee: 0.3, compression: 0.2 },
    }),
  },
  {
    id: "fuji-classic-chrome", maker: "Fujifilm", model: "X-T5", style: "Classic Chrome",
    lens: { name: "XF 23mm f/2", focal: 23, aperture: 2, sensor: "apsc", vignette: -0.08 },
    sensor: { grain: 0.1, grainSize: 0.35, sharpen: 0.3, localCompression: 0.15 },
    // Documentary: muted colour, firm shadows, teal-leaning blues, olive greens, deep calm reds.
    profile: look("fuji-classic-chrome", "Classic Chrome", {
      category: "documentary",
      tone: { contrast: 0.22, rolloff: 0.6, highlightCompression: 0.1, blackPoint: 0.005 },
      hsl: { red: { hue: 2, sat: -0.14, lum: -0.04 }, orange: { hue: 0, sat: -0.08, lum: 0 }, yellow: { hue: -3, sat: -0.22, lum: 0 }, green: { hue: 5, sat: -0.28, lum: -0.02 }, cyan: { hue: 0, sat: -0.1, lum: 0 }, blue: { hue: -8, sat: -0.22, lum: -0.04 } },
      colorBalance: { shadows: [-0.006, 0, 0.006], midtones: [0.004, 0.002, -0.002], highlights: [0.006, 0.004, 0] },
      saturation: { global: 0.82, knee: 0.22, compression: 0.35 },
    }),
  },
  {
    id: "fuji-classic-neg", maker: "Fujifilm", model: "X100VI", style: "Classic Negative",
    lens: { name: "23mm f/2", focal: 23, aperture: 2, sensor: "apsc", vignette: -0.12 },
    sensor: { grain: 0.14, grainSize: 0.4, sharpen: 0.28, localCompression: 0.12 },
    // Film: hard contrast, cyan-green shadows, warm highlights, teal greens, magenta-leaning reds.
    profile: look("fuji-classic-neg", "Classic Negative", {
      category: "cinema palette",
      tone: { contrast: 0.32, rolloff: 0.5, highlightCompression: 0.16, blackPoint: 0.025 },
      hsl: { red: { hue: -10, sat: -0.12, lum: -0.04 }, orange: { hue: -4, sat: -0.14, lum: 0 }, yellow: { hue: -6, sat: -0.18, lum: 0 }, green: { hue: 18, sat: -0.14, lum: -0.04 }, cyan: { hue: 0, sat: 0.06, lum: 0 }, blue: { hue: -6, sat: -0.1, lum: -0.03 } },
      colorBalance: { shadows: [-0.03, 0.01, 0.02], midtones: [0, 0.003, 0], highlights: [0.025, 0.008, -0.016] },
      saturation: { global: 0.86, knee: 0.24, compression: 0.35 },
    }),
  },
  {
    id: "fuji-acros", maker: "Fujifilm", model: "X-Pro3", style: "Acros",
    lens: { name: "XF 35mm f/1.4", focal: 35, aperture: 1.4, sensor: "apsc", vignette: -0.14 },
    sensor: { grain: 0.26, grainSize: 0.35, sharpen: 0.35, localCompression: 0.15 },
    // Black and white: rich mid-tone separation, deep but open shadows; reds a little lighter.
    profile: look("fuji-acros", "Acros", {
      category: "documentary",
      tone: { contrast: 0.22, rolloff: 0.6, highlightCompression: 0.08, blackPoint: -0.005 },
      hsl: { red: { hue: 0, sat: 0, lum: 0.04 }, orange: { hue: 0, sat: 0, lum: 0.03 }, blue: { hue: 0, sat: 0, lum: -0.05 }, green: { hue: 0, sat: 0, lum: -0.02 } },
      saturation: { global: 0, shadows: 0, highlights: 0 },
    }),
  },
  {
    id: "leica-m11", maker: "Leica", model: "M11", style: "Standard",
    lens: { name: "Summilux-M 35mm f/1.4", focal: 35, aperture: 1.4, sensor: "ff", vignette: -0.28 },
    sensor: { grain: 0.06, grainSize: 0.28, sharpen: 0.32, localCompression: 0.12 },
    // Deep blacks and micro-contrast, rich natural colour with a slight warmth, strong corners.
    profile: look("leica-m11", "Leica Standard", {
      category: "high-contrast cinematic",
      tone: { contrast: 0.22, rolloff: 0.6, highlightCompression: 0.08, blackPoint: -0.02 },
      hsl: { red: { hue: -2, sat: 0.08, lum: -0.03 }, orange: { hue: -1, sat: 0.03, lum: 0 }, yellow: { hue: -2, sat: 0.02, lum: 0 }, green: { hue: 2, sat: -0.04, lum: -0.02 }, blue: { hue: 0, sat: 0.04, lum: -0.04 } },
      colorBalance: { shadows: [0.002, 0, -0.002], midtones: [0.006, 0.002, -0.004], highlights: [0.004, 0.002, -0.002] },
      saturation: { global: 1.05, knee: 0.28, compression: 0.25 },
    }),
  },
  {
    id: "hasselblad-x2d", maker: "Hasselblad", model: "X2D 100C", style: "Natural Colour",
    lens: { name: "XCD 80mm f/1.9", focal: 80, aperture: 1.9, sensor: "mf", vignette: -0.08 },
    sensor: { grain: 0.015, grainSize: 0.2, sharpen: 0.2, localCompression: 0.1 },
    // Smooth and accurate: gentle contrast, a long highlight roll-off, true skin, calm saturation.
    profile: look("hasselblad-x2d", "Hasselblad Natural Colour", {
      category: "portrait-neutral",
      tone: { contrast: 0.06, rolloff: 0.82, highlightCompression: 0.15, shadowLift: 0.02 },
      hsl: { orange: { hue: 0, sat: -0.02, lum: 0.02 }, green: { hue: 0, sat: -0.04, lum: 0 } },
      saturation: { global: 1.0, knee: 0.3, compression: 0.3 },
    }),
  },
];

/** Which parts of a camera look are on. */
export interface CameraParts { colour: boolean; lens: boolean; sensor: boolean }
export const ALL_PARTS: CameraParts = { colour: true, lens: true, sensor: true };

/**
 * Applies a camera look (or none) to `p`. Everything a look sets is first put
 * back to the automatic rendering (`auto`), so switching looks or parts never
 * stacks. `flatDepth`: no depth map on this photo — no lens blur then.
 */
export function applyCameraLook(p: Params, auto: Params, look: CameraLook | undefined, parts: CameraParts, flatDepth: boolean) {
  p.profile = structuredClone(auto.profile);
  p.enable = { ...p.enable, lut: auto.enable.lut, dof: auto.enable.dof };
  p.vignette = { ...auto.vignette };
  p.grain = { ...auto.grain };
  p.sharpen = { ...p.sharpen, amount: auto.sharpen.amount };
  p.local = { ...p.local, compression: auto.local.compression };
  p.dof = { ...p.dof, strength: auto.dof.strength };
  p.camera = look ? { id: look.id, ...parts } : undefined;
  if (!look) return;
  if (parts.colour) {
    p.profile = structuredClone(look.profile);
    p.enable.lut = true; // the look stage (render_tone.wgsl apply_profile)
  }
  if (parts.lens) {
    p.vignette = { ...p.vignette, amount: look.lens.vignette };
    if (!flatDepth) {
      p.enable.dof = true;
      p.dof = { ...p.dof, mode: "focus", strength: lensBlurStrength(look.lens.focal, look.lens.aperture, look.lens.sensor) };
    }
  }
  if (parts.sensor) {
    p.grain = { ...p.grain, amount: look.sensor.grain, size: look.sensor.grainSize, color: 0.15 };
    p.sharpen = { ...p.sharpen, amount: look.sensor.sharpen };
    // A big sensor's rendering is not flattened by phone-style local tone mapping.
    p.local = { ...p.local, compression: Math.min(auto.local.compression, look.sensor.localCompression) };
  }
}
