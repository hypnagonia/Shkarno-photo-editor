/**
 * What the user changes after the automatic development, remembered on this device
 * only (never sent anywhere): per photo, the difference between the automatic
 * settings and the final ones, with what the photo was. Downloaded as a file from
 * More › Debug, it shows the automatic mode's systematic misses (exposure raised
 * every time, saturation always taken down…) to correct its defaults.
 */
import type { Params } from "../decision/params.ts";
import type { Summary } from "../engine/protocol.ts";

const KEY = "autoFeedback";
const MAX = 400;

/** The settings compared (dotted paths into Params). */
const PATHS = [
  "exposure", "wb.temp", "wb.tint",
  "tone.highlights", "tone.shadows", "tone.whites", "tone.blacks", "tone.contrast", "tone.rolloff",
  "local.compression", "local.clarity", "local.texture",
  "color.saturation", "color.vibrance", "dehaze.strength",
  "denoise.luma", "denoise.chroma", "sharpen.amount", "vignette.amount", "grain.amount", "film.strength",
  "autoCurves", "render.purity", "render.strength", "hdr.headroom",
];

export interface FeedbackEntry {
  t: string;
  file: string;
  /** "export": the photo was exported with these settings; "leave": another photo was opened. */
  event: "export" | "leave";
  source?: string;
  size?: string;
  meta?: Record<string, string | number>;
  /** final − automatic, only what changed. */
  delta: Record<string, number>;
  /** Other things done: layers added by type, a look chosen, equalizers, render engine. */
  extras: Record<string, string | number | boolean>;
}

const get = (p: unknown, path: string): number | undefined => {
  let o = p as Record<string, unknown> | undefined;
  for (const k of path.split(".")) o = o?.[k] as Record<string, unknown> | undefined;
  return typeof o === "number" ? o : undefined;
};

function read(): FeedbackEntry[] {
  try { return JSON.parse(localStorage.getItem(KEY) ?? "[]") as FeedbackEntry[]; } catch { return []; }
}

/** Remembers how `final` differs from `auto` for this photo (the latest per photo wins). Returns how many photos are remembered. */
export function recordEdit(event: FeedbackEntry["event"], summary: Summary | undefined, auto: Params | undefined, final: Params | undefined): number {
  if (!summary || !auto || !final) return read().length;
  const delta: Record<string, number> = {};
  for (const p of PATHS) {
    const a = get(auto, p), f = get(final, p);
    if (a === undefined || f === undefined) continue;
    const d = f - a;
    if (Math.abs(d) > (p === "wb.temp" ? 20 : 0.005)) delta[p] = Math.round(d * 1000) / 1000;
  }
  const extras: FeedbackEntry["extras"] = {};
  const own = (final.layers ?? []).filter((l) => !l.auto);
  for (const l of own) extras[`layer.${l.type}`] = ((extras[`layer.${l.type}`] as number) ?? 0) + 1;
  const autoLayers = (auto.layers ?? []).filter((l) => l.auto).length, keptAuto = (final.layers ?? []).filter((l) => l.auto && l.visible).length;
  if (keptAuto < autoLayers) extras.autoLayersRemoved = autoLayers - keptAuto;
  if (final.profile?.id && final.profile.id !== auto.profile?.id) extras.look = final.profile.id;
  if (final.toneEq?.enabled && final.toneEq.gains.some((g) => Math.abs(g) > 1e-3)) extras.toneEq = final.toneEq.gains.map((g) => Math.round(g * 100) / 100).join(" ");
  if (final.contrastEq?.enabled && [...final.contrastEq.luma, ...final.contrastEq.chroma].some((g) => Math.abs(g) > 1e-3)) extras.contrastEq = true;
  if (final.render?.engine !== auto.render?.engine) extras.render = final.render?.engine ?? "classic";
  if (final.enable.dof !== auto.enable.dof) extras.dof = final.enable.dof;
  // Nothing changed at all: the automatic result was kept (worth knowing too, on export).
  if (!Object.keys(delta).length && !Object.keys(extras).length && event === "leave") return read().length;
  const entry: FeedbackEntry = {
    t: new Date().toISOString(), file: summary.file, event, source: summary.source,
    size: `${summary.working.width}×${summary.working.height}`, meta: summary.meta, delta, extras,
  };
  const all = read().filter((e) => e.file !== entry.file);
  all.push(entry);
  try { localStorage.setItem(KEY, JSON.stringify(all.slice(-MAX))); } catch { /* quota: keep what is there */ }
  return Math.min(all.length, MAX);
}

export function feedbackCount(): number { return read().length; }

/** Everything remembered, with a summary per setting (mean change and how often it was changed). */
export function feedbackBlob(): Blob {
  const all = read();
  const stats: Record<string, { n: number; mean: number; up: number; down: number }> = {};
  for (const e of all) for (const [k, v] of Object.entries(e.delta)) {
    const s = (stats[k] ??= { n: 0, mean: 0, up: 0, down: 0 });
    s.n++; s.mean += v; if (v > 0) s.up++; else s.down++;
  }
  for (const s of Object.values(stats)) s.mean = Math.round((s.mean / s.n) * 1000) / 1000;
  return new Blob([JSON.stringify({ photos: all.length, summary: stats, entries: all }, null, 2)], { type: "application/json;charset=utf-8" });
}

export function clearFeedback() { try { localStorage.removeItem(KEY); } catch { /* private mode */ } }
