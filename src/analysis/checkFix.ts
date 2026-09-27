/**
 * Check fixes: for a finding, the controls that could fix it (in the order worth
 * trying) and which way. The engine moves one control at a time, re-rendering
 * small, until the finding is gone, and reports the value that does it — a number
 * to set, not a hint ("Exposure +1.60 → +0.65").
 */
import type { Params } from "../decision/params.ts";
import type { CheckItem } from "./check.ts";

/** A control and the end of its range in the direction that fixes the finding. */
export interface Lever { path: string; bound: number }
/** What to change: a control from → to, or a layer's opacity (0 = hide it). */
export type FixChange = { path: string; from: number; to: number } | { layer: string; name: string; from: number; to: number };

/** Step of each control (values are rounded to what the slider can show). */
export const STEP: Record<string, number> = { exposure: 0.05, "wb.temp": 10, "wb.tint": 0.5, autoCurves: 0.01 };
export const stepOf = (path: string) => STEP[path] ?? 0.01;

export function getPath(p: Params, path: string): number {
  let o: unknown = p;
  for (const k of path.split(".")) o = (o as Record<string, unknown> | undefined)?.[k];
  return typeof o === "number" ? o : NaN;
}
export function setPath(p: Params, path: string, v: number) {
  const ks = path.split(".");
  let o = p as unknown as Record<string, unknown>;
  for (const k of ks.slice(0, -1)) o = (o[k] ??= {}) as Record<string, unknown>;
  o[ks[ks.length - 1]] = v;
}

export function leversFor(it: CheckItem, p: Params): Lever[] {
  const L = (path: string, bound: number): Lever => ({ path, bound });
  const toward0 = (path: string) => L(path, 0);
  const temp = getPath(p, "wb.temp") || 6504, tint = getPath(p, "wb.tint") || 0;
  let out: Lever[];
  switch (it.id) {
    case "exposure": out = [L("exposure", it.v.dir === "bright" ? -3 : 3)]; break;
    case "highlights": out = [L("exposure", -3), L("tone.whites", -1), L("tone.highlights", -1)]; break;
    case "shadows": out = [L("tone.blacks", 1), L("tone.shadows", 1), L("exposure", 3)]; break;
    case "colorClip": out = [L("color.saturation", -1), L("color.vibrance", -1)]; break;
    case "saturation": out = Number(it.v.ratio) < 100 ? [L("color.saturation", 1), L("color.vibrance", 1)] : [L("color.saturation", -1), L("color.vibrance", -1)]; break;
    case "cast": {
      // Temperature for warm / blue casts, tint for green / magenta (Lightroom's sense: higher = warmer / more magenta).
      const t = String(it.v.tint);
      out = t === "warm" ? [L("wb.temp", Math.max(2000, temp - 3000))]
        : t === "blue" || t === "cyan" ? [L("wb.temp", Math.min(12000, temp + 3000)), L("wb.tint", Math.min(60, tint + 20))]
        : t === "green" ? [L("wb.tint", 60)] : [L("wb.tint", -60)];
      break;
    }
    case "contrast": {
      const i = String(it.v.issue);
      out = i === "flat" ? [L("tone.contrast", 1), L("tone.blacks", -1)] : i === "noBlack" ? [L("tone.blacks", -1), L("tone.contrast", 1)]
        : i === "noWhite" ? [L("tone.whites", 1), L("exposure", 3)] : [L("tone.contrast", -1), L("tone.shadows", 1)];
      break;
    }
    case "noise": out = [toward0("local.texture"), toward0("local.clarity"), toward0("sharpen.amount"), L("denoise.luma", 1)]; break;
    case "halos": out = [toward0("local.clarity"), toward0("local.compression"), toward0("dehaze.strength")]; break;
    case "vignette": out = [toward0("vignette.amount")]; break;
    default: out = [];
  }
  // The automatic grade as a whole (Auto strength) is the last resort before blaming a layer.
  if (it.id !== "vignette") out.push(L("autoCurves", 0));
  // A control already at the fixing end cannot help.
  return out.filter((l) => { const v = getPath(p, l.path); return Number.isFinite(v) && Math.abs(v - l.bound) > stepOf(l.path) / 2; });
}
