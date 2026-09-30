/**
 * Ask-an-LLM tab: builds a self-contained prompt (the photo's measurements,
 * every adjustable parameter with its range and current value, the available
 * looks, and a strict answer format), and applies the answer pasted back.
 *
 * The answer is a JSON object of *absolute* values keyed by parameter path;
 * every key is checked against the table below and clamped to its range, and
 * anything unknown is reported and ignored. One step of undo is kept.
 */
import { DEPTH_BANDS, neutralSemantic, type CurvePoint, type Curves, type Decision, type DepthBand, type Params, type Region, type SemanticAdjust } from "../decision/params.ts";
import { isFlat } from "../render/curves.ts";
import { flatCurves } from "./toneCurves.ts";
import type { Summary } from "../engine/protocol.ts";
import { GROUPS } from "../neural/scene.ts";
import { t } from "./i18n.ts";
import { el } from "./dom.ts";

type Ctx = {
  params: () => Params | undefined;
  auto: () => Params | undefined;
  summary: () => Summary | undefined;
  decisions: () => Decision[];
  looks: () => Array<{ id: string; name: string; description: string }>;
  /** Share of the frame (%) of each region at each distance. */
  cellCoverage?: () => Record<string, number> | undefined;
  /** Makes the look with this id the active one; false if there is none. */
  selectLook: (id: string) => boolean;
  changed: () => void;
  /** The preview as shown (for attaching to the conversation). */
  canvas: HTMLCanvasElement;
};

interface Num { path: string; min: number; max: number; what: string }

/** Everything the LLM may set, with the ranges the UI sliders use. */
const NUMS: Num[] = [
  { path: "exposure", min: -3, max: 3, what: "global exposure, EV" },
  { path: "tone.highlights", min: -1, max: 1, what: "negative recovers / compresses highlights" },
  { path: "tone.shadows", min: -1, max: 1, what: "positive lifts shadows (local, edge-aware)" },
  { path: "tone.whites", min: -1, max: 1, what: "white point" },
  { path: "tone.blacks", min: -1, max: 1, what: "black point; negative deepens blacks" },
  { path: "tone.contrast", min: -1, max: 1, what: "global contrast of the display curve" },
  { path: "tone.rolloff", min: 0, max: 1, what: "highlight shoulder softness" },
  { path: "wb.temp", min: 2000, max: 12000, what: "white balance temperature, Kelvin (higher = warmer rendering)" },
  { path: "wb.tint", min: -60, max: 60, what: "white balance tint (positive = magenta)" },
  { path: "local.compression", min: 0, max: 0.8, what: "local range compression (HDR-like; >0.5 looks flat)" },
  { path: "local.clarity", min: -0.5, max: 1, what: "medium-scale local contrast" },
  { path: "local.texture", min: -0.5, max: 1, what: "fine detail gain (also amplifies noise)" },
  { path: "dehaze.strength", min: 0, max: 1, what: "depth-aware haze removal" },
  { path: "color.vibrance", min: -1, max: 1, what: "saturation of weak colours" },
  { path: "color.saturation", min: -1, max: 1, what: "global saturation (0 = unchanged)" },
  { path: "denoise.luma", min: 0, max: 1, what: "luminance noise reduction blend" },
  { path: "denoise.chroma", min: 0, max: 1, what: "colour noise reduction blend" },
  { path: "sharpen.amount", min: 0, max: 1.5, what: "edge-aware sharpening" },
  { path: "sharpen.radius", min: 0.5, max: 2.5, what: "sharpening radius, px" },
  { path: "depth.near", min: 0.5, max: 1.5, what: "detail multiplier for near objects" },
  { path: "depth.far", min: 0.2, max: 1.5, what: "detail multiplier for far objects" },
  { path: "vignette.amount", min: -1, max: 1, what: "vignette in linear light: negative darkens the edges (≈ −2 EV in the corners at −1), positive lightens; 0 = none; −0.2…−0.4 is a natural lens-like vignette" },
  { path: "vignette.midpoint", min: 0, max: 1, what: "where the vignette falloff starts (higher = only the corners)" },
  { path: "vignette.feather", min: 0, max: 1, what: "how gradual the vignette falloff is" },
  { path: "vignette.roundness", min: 0, max: 1, what: "vignette shape: 0 follows the frame's aspect, 1 is a circle" },
  { path: "vignette.highlights", min: 0, max: 1, what: "how much bright light sources are spared by a darkening vignette" },
  { path: "grain.amount", min: 0, max: 1, what: "film grain, added last (not sharpened or blurred); 0 = none, 0.15–0.3 subtle fine-art film, 0.5+ strong high-ISO film" },
  { path: "grain.size", min: 0, max: 1, what: "grain particle size relative to the image (0 = fine 100-ISO, 1 = coarse 3200-ISO)" },
  { path: "grain.roughness", min: 0, max: 1, what: "grain clumping (0 = even, 1 = clumpy)" },
  { path: "grain.color", min: 0, max: 1, what: "0 = monochrome grain, 1 = colour (dye-cloud) grain" },
  { path: "hdr.headroom", min: 0, max: 3, what: "HDR: how far highlights may rise above SDR white on HDR screens and in the JPEG (HDR) export, in stops; 0 = SDR only; the SDR rendering does not change" },
  { path: "profile.intensity", min: 0, max: 1, what: "strength of the active look (0 = technical rendering only)" },
];

const SEM: Array<{ key: keyof SemanticAdjust; min: number; max: number; what: string }> = [
  { key: "exposure", min: -2, max: 2, what: "EV offset" },
  { key: "highlights", min: 0, max: 1, what: "extra highlight compression" },
  { key: "warmth", min: -1, max: 1, what: "cool … warm" },
  { key: "tint", min: -1, max: 1, what: "green … magenta" },
  { key: "saturation", min: -1, max: 1, what: "0 = unchanged" },
  { key: "vibrance", min: -1, max: 1, what: "" },
  { key: "hue", min: -30, max: 30, what: "hue rotation, degrees" },
  { key: "clarity", min: 0, max: 2, what: "multiplier, 1 = unchanged" },
  { key: "texture", min: 0, max: 2, what: "multiplier" },
  { key: "sharpen", min: 0, max: 2, what: "multiplier" },
  { key: "denoise", min: 0, max: 2, what: "multiplier" },
  { key: "dehaze", min: 0, max: 2, what: "multiplier" },
];
const CURVES = ["l", "r", "g", "b"] as const;


/** Path segments; under `cells` / `cellCurves` the region and distance form one key ("building.far"). */
function keys(path: string): string[] {
  const ks = path.split(".");
  return (ks[0] === "cells" || ks[0] === "cellCurves") && ks.length >= 3 ? [ks[0], `${ks[1]}.${ks[2]}`, ...ks.slice(3)] : ks;
}
const get = (o: unknown, path: string): unknown => keys(path).reduce((a: unknown, k) => (a as Record<string, unknown> | undefined)?.[k], o);
function set(o: unknown, path: string, v: unknown) {
  const ks = keys(path);
  const last = ks.pop()!;
  const parent = ks.reduce((a: Record<string, unknown>, k) => (a[k] ??= {}) as Record<string, unknown>, o as Record<string, unknown>);
  if (v === undefined) delete parent[last]; else parent[last] = v;
}
const r3 = (v: number) => Math.round(v * 1000) / 1000;
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const curveText = (c: CurvePoint[]) => JSON.stringify(c.map((q) => [r3(q.x), r3(q.y)]));

/** The prompt: context, parameters, looks, answer format. Always in English (it is for the model). */
export function buildPrompt(p: Params, auto: Params | undefined, s: Summary | undefined, decisions: Decision[], looks: Array<{ id: string; name: string; description: string }>, goal: string, cells?: Record<string, number>): string {
  const L: string[] = [];
  L.push("You are a photo editor adjusting a photograph in a raw development app (\"Shkarno\"). I will show you the current rendering (attached image, if any), the measurements the app made, and every parameter you can change. Propose changes that make this photograph look its best" + (goal.trim() ? " while following my request below." : "."));
  if (goal.trim()) L.push("", "## My request", goal.trim());
  L.push("", "## How the app renders",
    "Pipeline: denoise → white balance → dehaze → exposure → local tone mapping (compression, shadows/highlights, clarity, texture) → display tone curve (contrast, blacks, whites, rolloff) → user curves → saturation/vibrance and per-region colour → look profile (creative grade) → sharpening.",
    "The automatic values were chosen by the app from measurements and are usually sensible; prefer small, targeted changes over rewriting everything.",
    "",
    "The app has already analysed the photo and built two maps. You don't draw masks: name a region or a depth range and the app applies it through these maps.",
    "- Semantic objects map: every pixel has soft probabilities for 11 regions (sky, vegetation, building, ground, terrain, water, person, vehicle, animal, interior, other), from a segmentation network (and Apple's own sky/skin mattes on ProRAW), snapped to the real edges of the photo. Any semantic.<region>.<key> value is applied through this map, blended softly, with no hard edges or halos. So \"darken only the sky by 0.3 EV\", \"warm the people\", \"desaturate the buildings\" or \"more clarity on vegetation\" is one line each. The share of the frame each region covers is listed under Photo.",
    "- Skin: a layer across people (Apple's skin mask on ProRAW, otherwise the person region × skin colour) — faces and hands, not clothes. It takes all the same settings as a region (semantic.skin.<key>), which replace the region's own where there is skin: e.g. warmer or brighter faces, lower texture to smooth skin, less saturation for natural skin. Prefer small skin changes; people notice skin first.",
    "- Depth map: every pixel has a relative distance (near 0 … far 1) from a depth network, also edge-snapped. depth.near / depth.far scale clarity, texture and sharpening by distance (e.g. crisper foreground, softer background for depth). Dehaze uses it too (it clears distant haze more than near objects), and so does the look's depth-aware grading (far = slightly hazier and cooler, near = a touch more contrast). Skin is always protected from depth grading.",
    "- Curves: point curves on the display-encoded image. They are the most precise tone and colour tool here, so use them. curves.l shapes brightness and contrast: an S-curve for punch, a lifted first point for matte blacks, a lowered last point for softer whites, a bump in the mids to open up a dark photo. curves.r / curves.g / curves.b grade colour by tone: e.g. lift blue and lower red in the shadows and do the opposite in the highlights for teal/orange split toning; a small red lift in the mids for warmth. Keep them smooth (3–6 points; moves of about 0.02–0.08 are already clearly visible). They combine with the look, so a look plus a gentle curve is a good way to get a specific mood.");
  if (s) {
    L.push("", "## Photo",
      `File: ${s.file} · ${s.format} · source: ${s.source} · ${s.width}×${s.height}` + (s.working.factor > 1 ? ` (working ${s.working.width}×${s.working.height})` : ""));
    const meta = Object.entries(s.meta).map(([k, v]) => `${k}: ${v}`).join(" · ");
    if (meta) L.push(meta);
    const cov = Object.entries(s.coverage).filter(([, v]) => v >= 1).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${v}%`).join(", ");
    if (cov) L.push(`Regions (share of frame): ${cov}`);
  }
  if (decisions.length) {
    L.push("", "## What the app measured and decided");
    for (const d of decisions) {
      const v = Array.isArray(d.value) ? d.value.join(" / ") : String(d.value);
      L.push(`- ${d.id} = ${v.length > 40 ? v.slice(0, 40) + "…" : v}: ${d.reason}`);
    }
  }
  L.push("", "## Parameters (path — range — current [automatic] — meaning)");
  for (const n of NUMS) {
    const cur = get(p, n.path) as number, a = auto ? get(auto, n.path) as number : undefined;
    L.push(`- ${n.path} — ${n.min}…${n.max} — ${r3(cur)}${a !== undefined && n.path !== "profile.intensity" ? ` [${r3(a)}]` : ""} — ${n.what}`);
  }
  L.push("", "### Per-region adjustments",
    `Path: semantic.<region>.<key>. Regions: ${GROUPS.join(", ")}, and skin (the layer described above). Only regions present in the photo matter.`,
    "Keys: " + SEM.map((q) => `${q.key} (${q.min}…${q.max}${q.what ? ", " + q.what : ""})`).join("; ") + ".",
    "Current values that differ from neutral (exposure/highlights/warmth/tint/saturation/vibrance/hue 0, multipliers 1):");
  let any = false;
  for (const g of [...GROUPS, "skin"] as const) {
    const cur = g === "skin" ? p.skin : p.semantic[g];
    const diff = SEM.filter((q) => Math.abs(cur[q.key] - (["clarity", "texture", "sharpen", "denoise", "dehaze"].includes(q.key) ? 1 : 0)) > 1e-3);
    if (diff.length) { any = true; L.push(`- ${g}: ` + diff.map((q) => `${q.key} ${r3(cur[q.key])}`).join(", ")); }
  }
  if (!any) L.push("- (all neutral)");
  L.push("", "### By distance, and by region at a distance",
    "The semantic and depth maps together split every region by distance, so the near buildings and the far buildings (or near and far trees) are separate and can be treated differently — e.g. a warmer, brighter foreground building and a cooler, softer one behind it.",
    "- distance.<near|middle|far>.<key> — everything at that distance (all regions).",
    "- cells.<region>.<near|middle|far>.<key> — one region at one distance, e.g. cells.building.far.saturation.",
    "- cellCurves.<region>.<near|middle|far>.<l|r|g|b> — that part's own curve (same format as curves.l).",
    "Keys and ranges as for regions, but these are RELATIVE layers on top of the region's own settings: exposure/highlights/warmth/tint/saturation/vibrance/hue add (0 = no change), clarity/texture/sharpen/denoise/dehaze multiply (1 = no change). Order: region → distance → region at a distance → skin.",
    "Present in this photo (share of the frame): " + (Object.entries(cells ?? {}).filter(([, v]) => v >= 1).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${v}%`).join(", ") || "(unknown)"));
  const relDiff = (s: SemanticAdjust | undefined) => s ? SEM.filter((q) => Math.abs(s[q.key] - (["clarity", "texture", "sharpen", "denoise", "dehaze"].includes(q.key) ? 1 : 0)) > 1e-3).map((q) => `${q.key} ${r3(s[q.key])}`) : [];
  const layers = [
    ...DEPTH_BANDS.map((b) => [`distance.${b}`, relDiff(p.distance?.[b])] as const),
    ...Object.entries(p.cells ?? {}).map(([k, s]) => [`cells.${k}`, relDiff(s)] as const),
  ].filter(([, d]) => d.length).map(([k, d]) => `- ${k}: ${d.join(", ")}`);
  const cellCurveText = Object.entries(p.cellCurves ?? {}).flatMap(([r, c]) => CURVES.filter((k) => c?.[k] && !isFlat(c[k])).map((k) => `${r}.${k} ${curveText(c![k])}`));
  L.push("Current: " + (layers.length ? "" : "(none)"), ...layers, "Current region-at-a-distance curves: " + (cellCurveText.join(" · ") || "(none)"));
  L.push("", "### Curves",
    "Paths: curves.l (luminance), curves.r, curves.g, curves.b. Value: list of [x, y] points in display-encoded 0…1 (x = input, y = output), sorted by x, including [0, y0] and [1, y1], at most 8 points. The user edits them as point curves, so a few well-placed points read best. [[0,0],[1,1]] = unchanged. Examples: gentle S-curve [[0,0],[0.25,0.22],[0.75,0.79],[1,1]]; matte blacks [[0,0.04],[0.2,0.2],[1,1]]; cooler shadows via curves.b [[0,0.03],[0.3,0.31],[1,1]].",
    "Current: " + CURVES.map((c) => `${c} ${curveText(p.curves[c])}`).join(" · "),
    "Every region, and skin, also has its own curves, blended through its mask: regionCurves.<region>.l / .r / .g / .b (same format), applied after the photo's curves. E.g. regionCurves.sky.l to deepen only the sky, regionCurves.skin.r to warm only the skin's mid-tones, regionCurves.vegetation.g for greener foliage.",
    "Distance has its own curves too: depthCurves.near / .middle / .far . l / r / g / b — soft thirds of this photo's depth layers (sky is far), applied after the region curves and never on skin. E.g. depthCurves.far.l lifting darks slightly for aerial perspective, depthCurves.near.l a gentle S for a punchier foreground.",
    "Current distance curves: " + (Object.entries(p.depthCurves ?? {}).flatMap(([r, c]) => CURVES.filter((k) => c?.[k] && !isFlat(c[k])).map((k) => `${r}.${k} ${curveText(c![k])}`)).join(" · ") || "(none)"),
    "Current region curves: " + (Object.entries(p.regionCurves ?? {}).flatMap(([r, c]) => CURVES.filter((k) => c?.[k] && !isFlat(c[k])).map((k) => `${r}.${k} ${curveText(c![k])}`)).join(" · ") || "(none)"));
  L.push("", "### Look (creative grade, applied after the technical rendering)",
    `Active: ${p.profile.id} (intensity ${r3(p.profile.intensity)}). Choose one with "look": "<id>" — available:`);
  for (const l of looks) L.push(`- ${l.id}: ${l.name}${l.description ? " — " + l.description : ""}`);
  L.push("", "## Answer format (strict)",
    "Reply with a short explanation (at most 5 bullet points), then exactly ONE fenced ```json block containing one object:",
    "```json",
    "{",
    "  \"look\": \"<look id, optional>\",",
    "  \"set\": {",
    "    \"<parameter path>\": <absolute new value>,",
    "    \"semantic.sky.saturation\": -0.1,",
    "    \"curves.l\": [[0,0],[0.25,0.23],[0.75,0.78],[1,1]]",
    "  }",
    "}",
    "```",
    "Rules: values are ABSOLUTE (not deltas) and must lie within the ranges above; include only parameters you change; use only the paths listed above; plain JSON — no comments, no trailing commas. If the look is changed, set profile.intensity too if it should differ from 1.");
  return L.join("\n");
}

/** Pulls the JSON object out of a reply (a ```json block, else the outermost braces). */
export function extractJson(text: string): unknown {
  const fence = /```(?:json)?\s*([\s\S]*?)```/gi;
  let m: RegExpExecArray | null;
  const blocks: string[] = [];
  while ((m = fence.exec(text))) if (m[1].includes("{")) blocks.push(m[1]);
  let src = blocks.length ? blocks[blocks.length - 1] : text;
  const a = src.indexOf("{"), b = src.lastIndexOf("}");
  if (a < 0 || b <= a) throw new Error("no JSON object found");
  src = src.slice(a, b + 1);
  // Strict JSON first; then with comments and trailing commas removed (outside
  // strings only); last, with typographic quotes taken as quotes (a reply that
  // went through a "smart quotes" editor).
  const tries = [src, relax(src), relax(src.replace(/[“”]/g, '"'))];
  let err: unknown;
  for (const t of tries) { try { return JSON.parse(t); } catch (e) { err = e; } }
  throw err;
}

/** Drops // and /* *\/ comments and trailing commas, leaving string contents alone. */
function relax(src: string): string {
  let out = "", i = 0, inStr = false;
  while (i < src.length) {
    const c = src[i];
    if (inStr) {
      out += c;
      if (c === "\\") { out += src[i + 1] ?? ""; i += 2; continue; }
      if (c === '"') inStr = false;
      i++;
      continue;
    }
    if (c === '"') { inStr = true; out += c; i++; continue; }
    if (c === "/" && src[i + 1] === "/") { while (i < src.length && src[i] !== "\n") i++; continue; }
    if (c === "/" && src[i + 1] === "*") { const e = src.indexOf("*/", i + 2); i = e < 0 ? src.length : e + 2; continue; }
    if (c === ",") {
      let j = i + 1;
      while (j < src.length && /\s/.test(src[j])) j++;
      if (src[j] === "}" || src[j] === "]") { i++; continue; }
    }
    out += c;
    i++;
  }
  return out;
}

/** A number, or a string that is one; anything else (null, true, "", [5]) is not. */
function num(raw: unknown): number {
  if (typeof raw === "number") return raw;
  if (typeof raw === "string" && raw.trim() !== "") return Number(raw);
  return NaN;
}

/** A curve from a reply: [[x, y], …] or [{x, y}, …], sorted, one point per x, with identity ends added. */
function parseCurve(raw: unknown): CurvePoint[] | undefined {
  const pts = Array.isArray(raw) ? raw.map((q) => Array.isArray(q) ? { x: num(q[0]), y: num(q[1]) } : { x: num((q as CurvePoint)?.x), y: num((q as CurvePoint)?.y) }) : [];
  const ok = pts.filter((q) => Number.isFinite(q.x) && Number.isFinite(q.y)).map((q) => ({ x: clamp(q.x, 0, 1), y: clamp(q.y, 0, 1) })).sort((a, b) => a.x - b.x)
    .filter((q, i, arr) => i === 0 || q.x - arr[i - 1].x > 1e-4); // one point per x
  if (!ok.length) return undefined;
  // A curve without its ends would hold the first/last value flat to 0 and 1
  // (crushed shadows, clipped highlights): missing ends are the identity — so a
  // single mid point ([[0.5, 0.53]]) is a valid gentle lift.
  if (ok[0].x > 1e-4) ok.unshift({ x: 0, y: 0 });
  if (ok[ok.length - 1].x < 1 - 1e-4) ok.push({ x: 1, y: 1 });
  return ok.length <= 16 ? ok : undefined;
}

/** Applies an answer to `p` (mutating). Returns what happened, one line each. */
export function applyAnswer(p: Params, answer: unknown, selectLook: (id: string) => boolean): { applied: string[]; ignored: string[]; previous: Array<[string, unknown]> } {
  const applied: string[] = [], ignored: string[] = [];
  // What each touched path held before, so undo restores only those.
  const previous: Array<[string, unknown]> = [];
  const remember = (path: string) => { if (!previous.some(([q]) => q === path)) previous.push([path, structuredClone(get(p, path))]); };
  if (!answer || typeof answer !== "object") throw new Error("the answer is not a JSON object");
  const o = answer as Record<string, unknown>;
  // A flat object without "set" is accepted too.
  const changes = (o.set && typeof o.set === "object" ? o.set : Object.fromEntries(Object.entries(o).filter(([k]) => k !== "look" && k !== "why"))) as Record<string, unknown>;
  if (typeof o.look === "string" && o.look) {
    if (o.look === p.profile.id) applied.push(`look ${o.look} (already active)`);
    else {
      const before = structuredClone(p.profile);
      if (selectLook(o.look)) {
        previous.push(["profile", before]);
        // As the prompt says: a new look is at full strength unless the answer sets it.
        p.profile.intensity = 1;
        applied.push(`look → ${o.look}`);
      } else ignored.push(`look ${o.look}: no such look`);
    }
  }
  for (const [path, raw] of Object.entries(changes)) {
    const n = NUMS.find((q) => q.path === path);
    if (n) {
      const v = num(raw);
      if (!Number.isFinite(v)) { ignored.push(`${path}: not a number`); continue; }
      const c = clamp(v, n.min, n.max);
      if (!previous.some(([q]) => q === "profile") || !path.startsWith("profile.")) remember(path);
      set(p, path, c);
      applied.push(`${path} = ${r3(c)}${c !== v ? ` (clamped from ${v})` : ""}`);
      continue;
    }
    const sm = /^semantic\.(\w+)\.(\w+)$/.exec(path);
    if (sm) {
      const g = sm[1], def = SEM.find((q) => q.key === sm[2]);
      const v = num(raw);
      const isSkin = g === "skin";
      if (!(isSkin || GROUPS.includes(g as (typeof GROUPS)[number])) || !def) { ignored.push(`${path}: unknown region or key`); continue; }
      if (!Number.isFinite(v)) { ignored.push(`${path}: not a number`); continue; }
      const c = clamp(v, def.min, def.max);
      const target = isSkin ? `skin.${def.key}` : path; // where it lives in the parameters
      remember(target);
      set(p, target, c);
      applied.push(`${path} = ${r3(c)}${c !== v ? ` (clamped from ${v})` : ""}`);
      continue;
    }
    // By distance (relative): distance.<near|middle|far>.<key>
    const ds = /^distance\.(\w+)\.(\w+)$/.exec(path);
    if (ds) {
      const def = SEM.find((q) => q.key === ds[2]);
      const v = num(raw);
      if (!DEPTH_BANDS.includes(ds[1] as DepthBand) || !def) { ignored.push(`${path}: use near, middle or far and a region key`); continue; }
      if (!Number.isFinite(v)) { ignored.push(`${path}: not a number`); continue; }
      const c = clamp(v, def.min, def.max);
      p.distance ??= { near: neutralSemantic(), middle: neutralSemantic(), far: neutralSemantic() };
      remember(path);
      set(p, path, c);
      applied.push(`${path} = ${r3(c)}${c !== v ? ` (clamped from ${v})` : ""}`);
      continue;
    }
    // A region at a distance (relative): cells.<region>.<band>.<key>, cellCurves.<region>.<band>.<c>
    const cs = /^(cells|cellCurves)\.(\w+)\.(\w+)\.(\w+)$/.exec(path);
    if (cs) {
      const [, what, g, b, k] = cs;
      if (!GROUPS.includes(g as (typeof GROUPS)[number]) || !DEPTH_BANDS.includes(b as DepthBand)) { ignored.push(`${path}: unknown region or distance`); continue; }
      const cell = `${g}.${b}`;
      if (what === "cells") {
        const def = SEM.find((q) => q.key === k);
        const v = num(raw);
        if (!def) { ignored.push(`${path}: unknown key`); continue; }
        if (!Number.isFinite(v)) { ignored.push(`${path}: not a number`); continue; }
        const c = clamp(v, def.min, def.max);
        p.cells ??= {};
        remember(`cells.${cell}`);
        (p.cells as Record<string, SemanticAdjust>)[cell] = { ...((p.cells as Record<string, SemanticAdjust>)[cell] ?? neutralSemantic()), [k]: c };
        applied.push(`${path} = ${r3(c)}${c !== v ? ` (clamped from ${v})` : ""}`);
      } else {
        if (!/^[lrgb]$/.test(k)) { ignored.push(`${path}: channel must be l, r, g or b`); continue; }
        const ok = parseCurve(raw);
        if (!ok) { ignored.push(`${path}: needs 1…16 points`); continue; }
        p.cellCurves ??= {};
        remember(`cellCurves.${cell}`);
        const cc = p.cellCurves as Record<string, Curves>;
        cc[cell] = { ...(cc[cell] ?? flatCurves()), [k]: ok };
        applied.push(`${path} = ${curveText(ok)}`);
      }
      continue;
    }
    const rc = /^regionCurves\.(\w+)\.([lrgb])$/.exec(path);
    if (rc) {
      const r = rc[1] as Region;
      if (!(r === "skin" || GROUPS.includes(r as (typeof GROUPS)[number]))) { ignored.push(`${path}: unknown region`); continue; }
      const ok = parseCurve(raw);
      if (!ok) { ignored.push(`${path}: needs 1…16 points`); continue; }
      p.regionCurves ??= {};
      remember(`regionCurves.${r}`);
      p.regionCurves[r] = { ...(p.regionCurves[r] ?? flatCurves()), [rc[2]]: ok };
      applied.push(`${path} = ${curveText(ok)}`);
      continue;
    }
    const dc = /^depthCurves\.(\w+)\.([lrgb])$/.exec(path);
    if (dc) {
      const b = dc[1] as DepthBand;
      if (!DEPTH_BANDS.includes(b)) { ignored.push(`${path}: use near, middle or far`); continue; }
      const ok = parseCurve(raw);
      if (!ok) { ignored.push(`${path}: needs 1…16 points`); continue; }
      p.depthCurves ??= {};
      remember(`depthCurves.${b}`);
      p.depthCurves[b] = { ...(p.depthCurves[b] ?? flatCurves()), [dc[2]]: ok };
      applied.push(`${path} = ${curveText(ok)}`);
      continue;
    }
    const cm = /^curves\.([lrgb])$/.exec(path);
    if (cm) {
      const ok = parseCurve(raw);
      if (!ok) { ignored.push(`${path}: needs 1…16 points`); continue; }
      remember(path);
      p.curves[cm[1] as (typeof CURVES)[number]] = ok;
      applied.push(`${path} = ${curveText(ok)}`);
      continue;
    }
    ignored.push(`${path}: not an adjustable parameter`);
  }
  return { applied, ignored, previous };
}

export function createLlmPanel(root: HTMLElement, ctx: Ctx) {
  let undo: Array<[string, unknown]> | undefined;

  const goal = el("textarea", { class: "llm-text", rows: "3", placeholder: t("llm.goalPlaceholder") });
  const promptBox = el("textarea", { class: "llm-text mono", rows: "8", readonly: "" });
  const copyBtn = el("button", { class: "btn small primary", text: t("llm.copyPrompt") });
  const imgBtn = el("button", { class: "btn small", text: t("llm.copyImage") });
  const answer = el("textarea", { class: "llm-text mono", rows: "8", placeholder: t("llm.answerPlaceholder") });
  const applyBtn = el("button", { class: "btn small primary", text: t("llm.apply") });
  const undoBtn = el("button", { class: "btn small", text: t("llm.undo") });
  const status = el("pre", { class: "log llm-status" });
  undoBtn.disabled = true;

  const note = (btn: HTMLButtonElement, text: string) => {
    const was = btn.textContent;
    btn.textContent = text;
    setTimeout(() => { btn.textContent = was; }, 1500);
  };

  function refresh() {
    const p = ctx.params();
    promptBox.value = p ? buildPrompt(p, ctx.auto(), ctx.summary(), ctx.decisions(), ctx.looks(), goal.value, ctx.cellCoverage?.()) : t("llm.noPhoto");
  }
  goal.oninput = refresh;

  copyBtn.onclick = async () => {
    refresh();
    if (!ctx.params()) return;
    try { await navigator.clipboard.writeText(promptBox.value); note(copyBtn, t("llm.copied")); }
    catch { promptBox.select(); document.execCommand("copy"); note(copyBtn, t("llm.copied")); }
  };
  imgBtn.onclick = () => {
    if (!ctx.params() || !ctx.canvas.width) return;
    // The clipboard item is created right in the tap (Safari refuses one created
    // later, in the toBlob callback); the PNG arrives as a promise.
    const png = new Promise<Blob>((res, rej) => ctx.canvas.toBlob((b) => (b ? res(b) : rej(new Error("no image"))), "image/png"));
    const save = (blob: Blob) => {
      // No image clipboard (older browsers, Firefox): save the preview instead.
      const a = el("a", { href: URL.createObjectURL(blob), download: (ctx.summary()?.file ?? "photo").replace(/\.[^.]+$/, "") + "-preview.png" });
      document.body.append(a); a.click();
      setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 10000);
    };
    try {
      navigator.clipboard.write([new ClipboardItem({ "image/png": png })])
        .then(() => note(imgBtn, t("llm.copied")), () => png.then(save, () => {}));
    } catch { png.then(save, () => {}); }
  };
  applyBtn.onclick = () => {
    const p = ctx.params();
    if (!p) return;
    let parsed: unknown;
    try { parsed = extractJson(answer.value); }
    catch (e) { status.textContent = t("llm.parseError", { msg: (e as Error).message }); return; }
    let res: { applied: string[]; ignored: string[]; previous: Array<[string, unknown]> };
    try { res = applyAnswer(p, parsed, ctx.selectLook); }
    catch (e) { status.textContent = t("llm.parseError", { msg: (e as Error).message }); return; }
    if (res.applied.length) { undo = res.previous; undoBtn.disabled = false; ctx.changed(); }
    status.textContent = [
      t("llm.applied", { n: res.applied.length }), ...res.applied.map((l) => "  ✓ " + l),
      ...(res.ignored.length ? [t("llm.ignored", { n: res.ignored.length }), ...res.ignored.map((l) => "  ✗ " + l)] : []),
    ].join("\n");
    refresh();
  };
  undoBtn.onclick = () => {
    const p = ctx.params();
    if (!p || !undo) return;
    // Only what the answer changed goes back; later edits elsewhere stay.
    for (const [path, v] of [...undo].reverse()) set(p, path, structuredClone(v));
    undo = undefined;
    undoBtn.disabled = true;
    status.textContent = t("llm.undone");
    ctx.changed();
    refresh();
  };

  root.append(
    el("p", { class: "muted", text: t("llm.hint") }),
    el("div", { class: "group-title", text: t("llm.goal") }), goal,
    el("div", { class: "group-title", text: t("llm.prompt") }), promptBox,
    el("div", { class: "actions" }, copyBtn, imgBtn),
    el("div", { class: "group-title", text: t("llm.answer") }), answer,
    el("div", { class: "actions" }, applyBtn, undoBtn),
    status,
  );
  refresh();

  return {
    /** New photo, new analysis or edits elsewhere: the prompt shows the current values. */
    refresh,
    /** A different photo: the undo step belongs to the previous one. */
    reset() { undo = undefined; undoBtn.disabled = true; status.textContent = ""; refresh(); },
  };
}
