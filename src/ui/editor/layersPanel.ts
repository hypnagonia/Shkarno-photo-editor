/**
 * The layers editor (Photopea-style, phone first).
 *
 *   dock        the stack: top layer first, Develop (the RAW development) last and
 *               fixed; each card: eye, name, "Auto" badge; tap selects, long-press
 *               and drag reorders; ＋ adds a layer above the selected one
 *   properties  the selected layer: name, show/hide, duplicate, delete, reset to
 *               automatic; segments Adjust · Mask · Blend
 *
 * Every change calls ctx.changed(label): the app pushes the parameters (drafts
 * while a control is held) and commits one history step when the gesture ends.
 */
import type { Params, Region, DepthBand } from "../../decision/params.ts";
import { DEPTH_BANDS } from "../../decision/params.ts";
import { GROUPS } from "../../neural/scene.ts";
import type { HistTarget } from "../../analysis/previewHist.ts";
import { BLEND_GROUPS, fullRange, HUE_RANGES, MASK_OPS, MAX_MASK_PARTS, RANGE_CENTRE, makeLayer, newLayerDefaults, newId, type HueRange, type Layer, type LayerParams, type LayerType, type MaskKind, type MaskOp, type MaskPart, type MaskShape, type SmartMask, selectKey } from "../../layers/model.ts";
import type { PickInfo } from "../../engine/protocol.ts";
import { oklabToLinSrgb } from "../../color/oklab.ts";
import { createToneCurves } from "../toneCurves.ts";
import { t, tOr } from "../i18n.ts";
import { icon } from "./icons.ts";
import { createGradientEditor } from "./gradientEditor.ts";
import { defaultShape, liveLayers } from "../../layers/gpu.ts";
import { flareLayers, flareLight, moveFlare } from "../../layers/flare.ts";
import { el } from "../dom.ts";

type Ctx = {
  params: () => Params | undefined;
  auto: () => Params | undefined;
  coverage: () => Record<string, number> | undefined;
  cellCoverage: () => Record<string, number> | undefined;
  histogram: (target: HistTarget, chan: "l" | "r" | "g" | "b") => ArrayLike<number> | undefined;
  /** Parameters changed (label for the history step). */
  changed: (label: string) => void;
  /** Shows a layer's mask on the photo (its index among the visible layers), or stops. */
  showMask: (liveIndex: number | undefined) => void;
  /** The Develop properties (exposure, tone, colour, detail). */
  develop: HTMLElement;
  /** The Blur properties (depth of field: focus, strength, zones, depth views). */
  blur: HTMLElement;
  /** Something other than Blur became selected: its photo tools (focus picking, zone views) end. */
  leftBlur?: () => void;
  /** Panels of their own in the dock (tone / contrast equalizers): element, shown, left. */
  toneEq?: { el: HTMLElement; render: () => void; leave: () => void };
  contrastEq?: { el: HTMLElement; render: () => void; leave: () => void };
  /** The Film card: grain, halation, glow (src/ui/filmPanel.ts). */
  film?: { el: HTMLElement; render: () => void; leave: () => void };
  /** Taps on the photo pick what to mask (on, with a hint for the photo) or do what they normally do (off). */
  pickMode: (on: boolean, hint?: string) => void;
  /** A short message on the photo. */
  notice?: (text: string) => void;
  /** Where the light probably is in the photo (a lens flare's default position). */
  brightest?: () => Promise<{ x: number; y: number }>;
  /** The photo's main colours (hex), for "From photo" in gradients. */
  photoColors?: () => Promise<string[]>;
};

/** Layer types in the ＋ sheet (each type's icon has the type's name). */
/** The cards that are not layers: always there, at the bottom of the stack. */
type Fixed = "develop" | "toneEq" | "contrastEq" | "blur" | "film";

const ADD: LayerType[] = ["curves", "hueSat", "basic", "blur", "gradientMap", "gradientFill", "brightContrast", "exposure"];


/** A layer's name as shown: automatic layers in the interface language (unless renamed). */
export function layerName(l: Layer): string {
  if (!l.auto || (l as Layer & { renamed?: boolean }).renamed) return l.name;
  const [kind, what] = l.auto.split(".");
  const regionOrBand = (w: string) => (DEPTH_BANDS as string[]).includes(w) ? t(`band.${w as DepthBand}`) : tOr(`group.${w}`, w);
  if (l.auto === "curves.photo") return t("lay.photoTone");
  if (l.auto === "curves.black") return t("lay.blackPoint");
  if (l.auto === "subject") return t("lay.subject");
  if (kind === "curves" && what) return t("lay.tone", { name: regionOrBand(what) });
  if (kind === "colour" && what) return t("lay.colour", { name: regionOrBand(what) });
  return l.name;
}

export function createLayersPanel(dock: HTMLElement, props: HTMLElement, ctx: Ctx) {
  let selected = "develop";
  let tab: "adjust" | "mask" | "blend" = "adjust";
  let showMask = true;
  let visible = true;

  const layers = () => ctx.params()?.layers ?? [];
  const sel = () => layers().find((l) => l.id === selected);
  const liveIndex = (id: string) => { const p = ctx.params(); return liveLayers(layers(), p?.autoCurves ?? 1, p?.enable).findIndex((l) => l.id === id); };
  const typeName = (ty: LayerType) => t(`lay.type.${ty}`);
  const edit = (label?: string) => { const l = sel(); ctx.changed(label ?? (l ? t("hist.editLayer", { name: layerName(l) }) : t("hist.edit"))); };

  // ------------------------------------------------------------------ dock
  const list = el("div", { class: "lay-list" });
  const addBtn = el("button", { class: "lay-add", title: t("lay.add"), "aria-label": t("lay.add") }, icon("plus", 22));
  dock.replaceChildren(list, addBtn);

  /** A layer's card, or (no layer) one of the fixed cards: Develop, Blur. */
  function card(l: Layer | undefined, fixed: Fixed = "develop"): HTMLElement {
    const id = l?.id ?? fixed;
    const on = id === selected;
    const name = el("span", { class: "lay-name", text: l ? layerName(l) : t(`lay.${fixed}`) });
    const glyph = el("span", { class: "lay-glyph" }, icon(l ? l.type : fixed, 18));
    const hidden = l && !l.visible;
    const c = el("div", { class: "lay-card" + (on ? " on" : "") + (l?.auto ? " auto" : "") + (l ? "" : " develop") + (hidden ? " hidden" : ""), "data-id": id }, glyph, name);
    // Automatic layers: a small dot (the full word is in the properties); hidden layers: dimmed, eye crossed.
    if (l?.auto) c.append(el("span", { class: "lay-badge", title: t("lay.auto") }));
    const eye = el("span", { class: "lay-eye", role: "button", "aria-label": t("lay.visible") }, icon(hidden ? "eyeOff" : "eye", 16));
    if (l) c.append(eye);
    eye.onclick = (e) => { e.stopPropagation(); if (!l) return; l.visible = !l.visible; edit(t(l.visible ? "hist.show" : "hist.hide", { name: layerName(l) })); render(); };
    c.onclick = () => { if (dragMoved) return; selected = id; tab = "adjust"; render(); };
    if (l) enableReorder(c, l);
    return c;
  }

  // Long-press, then drag along the list: reorder (horizontal strip on phones, vertical list on desktop).
  // The card moves in the list while dragging; the stack changes once, on release (one history step).
  let dragMoved = false;
  function enableReorder(c: HTMLElement, l: Layer) {
    let timer = 0, dragging = false, start = { x: 0, y: 0 };
    c.addEventListener("pointerdown", (e) => {
      dragMoved = false;
      start = { x: e.clientX, y: e.clientY };
      timer = window.setTimeout(() => { dragging = true; c.classList.add("dragging"); try { c.setPointerCapture(e.pointerId); } catch { /* pointer gone */ } }, 350);
    });
    // Once dragging, the finger moves the card, not the list.
    c.addEventListener("touchmove", (e) => { if (dragging) e.preventDefault(); }, { passive: false });
    c.addEventListener("pointermove", (e) => {
      if (!dragging) { if (Math.hypot(e.clientX - start.x, e.clientY - start.y) > 8) clearTimeout(timer); return; }
      dragMoved = true;
      const horizontal = getComputedStyle(list).flexDirection === "row";
      const pos = horizontal ? e.clientX : e.clientY;
      const others = [...list.querySelectorAll<HTMLElement>(".lay-card:not(.develop)")].filter((k) => k !== c);
      const before = others.find((k) => { const r = k.getBoundingClientRect(); return pos < (horizontal ? r.left + r.width / 2 : r.top + r.height / 2); });
      const target = before ?? list.querySelector<HTMLElement>(".lay-card.develop");
      if (target && c.nextElementSibling !== target) list.insertBefore(c, target);
    });
    const end = () => {
      clearTimeout(timer);
      if (dragging) {
        dragging = false;
        c.classList.remove("dragging");
        const p = ctx.params();
        if (p && dragMoved) {
          // Display order is top-first; the array is bottom-first.
          const ids = [...list.querySelectorAll<HTMLElement>(".lay-card:not(.develop)")].map((k) => k.dataset.id).reverse();
          const next = ids.map((id) => p.layers.find((x) => x.id === id)).filter((x): x is Layer => !!x);
          if (next.length === p.layers.length && next.some((x, i) => x !== p.layers[i])) { p.layers.splice(0, p.layers.length, ...next); ctx.changed(t("hist.order")); }
          renderDock();
        }
      }
      setTimeout(() => (dragMoved = false), 0);
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
  }

  function renderDock() {
    list.replaceChildren(...[...layers()].reverse().map((l) => card(l)), card(undefined, "develop"), ...(ctx.toneEq ? [card(undefined, "toneEq")] : []), ...(ctx.contrastEq ? [card(undefined, "contrastEq")] : []), card(undefined, "blur"), ...(ctx.film ? [card(undefined, "film")] : []));
    list.querySelector(".lay-card.on")?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  // ------------------------------------------------------------------ add sheet
  const addSheet = el("div", { class: "lay-addsheet", hidden: "" });
  addBtn.onclick = () => { addSheet.hidden = !addSheet.hidden; };
  addSheet.append(el("div", { class: "group-title", text: t("lay.add") }), el("div", { class: "lay-addgrid" }, ...ADD.map((type) => {
    const b = el("button", { class: "lay-addbtn" }, el("span", { class: "g" }, icon(type, 19)), el("span", { text: typeName(type).replace("/", "/\u200b") }));
    b.onclick = () => {
      const p = ctx.params(); if (!p) return;
      const n = p.layers.filter((l) => l.type === type).length + 1;
      const l = makeLayer(type, `${typeName(type)} ${n}`, newLayerDefaults(type));
      const at = p.layers.findIndex((x) => x.id === selected);
      p.layers.splice(at + 1, 0, l); // above the selected one (Develop: at the bottom of the stack)
      selected = l.id; tab = "adjust"; addSheet.hidden = true;
      ctx.changed(t("hist.new", { name: typeName(type) }));
      render();
    };
    return b;
  }), (() => {
    // Lens flare: six ordinary layers (src/layers/flare.ts) at the photo's brightest spot,
    // then a tap on the light places it exactly.
    const b = el("button", { class: "lay-addbtn" }, el("span", { class: "g" }, icon("flare", 19)), el("span", { text: t("flare.add") }));
    b.onclick = async () => {
      const p0 = ctx.params(); if (!p0) return;
      addSheet.hidden = true;
      const L = (await ctx.brightest?.().catch(() => undefined)) ?? { x: 0.3, y: 0.2 };
      // Undo or another photo meanwhile: the edit this was for is gone.
      const p = ctx.params();
      if (p !== p0 || !p) return;
      const ls = flareLayers(L, { veil: t("flare.veil"), glow: t("flare.glow"), streak: t("flare.streak"), ghost: t("flare.ghost") });
      const at = p.layers.findIndex((x) => x.id === selected);
      p.layers.splice(at + 1, 0, ...ls);
      selected = ls[1].id; tab = "adjust";
      ctx.changed(t("flare.add"));
      setPlacing(ls[0].flare!.set); // before render: the Move light button shows "Tap the light"
      render();
    };
    return b;
  })()));
  dock.append(addSheet);
  // A tap anywhere else closes the ＋ sheet.
  document.addEventListener("pointerdown", (e) => {
    if (!addSheet.hidden && !addSheet.contains(e.target as Node) && !addBtn.contains(e.target as Node)) addSheet.hidden = true;
  }, true);

  // ------------------------------------------------------------------ controls
  function slider(label: string, min: number, max: number, step: number, get: () => number, set: (v: number) => void, fmt: (v: number) => string, def: number): HTMLElement {
    const input = el("input", { type: "range", min: String(min), max: String(max), step: String(step) });
    const out = el("output");
    const show = () => { const v = get(); input.value = String(v); out.textContent = fmt(v); out.classList.toggle("auto", Math.abs(v - def) < 1e-6); };
    input.oninput = () => { set(parseFloat(input.value)); show(); edit(); };
    const lab = el("label", { text: label });
    lab.addEventListener("dblclick", () => { set(def); show(); edit(); });
    show();
    return el("div", { class: "row" }, lab, input, out);
  }
  /**
   * A direction −90…90° (0 = horizontal, positive = up to the right): a slider, and a dial
   * whose two-headed arrow shows it (a streak runs both ways) — drag the dial to set it.
   */
  function angleRow(label: string, get: () => number, set: (v: number) => void): HTMLElement {
    const input = el("input", { type: "range", min: "-90", max: "90", step: "1" });
    const lab = el("label", { text: label });
    const dial = el("span", { class: "angle-dial", role: "slider", "aria-label": label, "aria-valuemin": "-90", "aria-valuemax": "90" });
    dial.innerHTML = `<svg viewBox="-20 -20 40 40" aria-hidden="true"><circle r="18.5" fill="none" stroke="currentColor" stroke-opacity=".35"/><g class="arrow"><path d="M-13 0H13M-13 0l5-4.5M-13 0l5 4.5M13 0l-5-4.5M13 0l-5 4.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></g></svg>`;
    const arrow = dial.querySelector<SVGGElement>(".arrow")!;
    const show = () => {
      const v = get();
      input.value = String(v);
      lab.textContent = `${label} ${Math.round(v)}°`;
      arrow.setAttribute("transform", `rotate(${-v})`); // screen y points down
      dial.setAttribute("aria-valuenow", String(Math.round(v)));
    };
    input.oninput = () => { set(parseFloat(input.value)); show(); edit(); };
    lab.addEventListener("dblclick", () => { set(0); show(); edit(); });
    // Drag on the dial: the angle of the finger from its centre, folded to −90…90.
    const fromPointer = (e: PointerEvent) => {
      const r = dial.getBoundingClientRect();
      let a = (Math.atan2(-(e.clientY - (r.top + r.height / 2)), e.clientX - (r.left + r.width / 2)) * 180) / Math.PI;
      if (a > 90) a -= 180; else if (a < -90) a += 180;
      set(Math.round(a)); show(); edit();
    };
    dial.addEventListener("pointerdown", (e) => { dial.setPointerCapture(e.pointerId); fromPointer(e); });
    dial.addEventListener("pointermove", (e) => { if (dial.hasPointerCapture(e.pointerId)) fromPointer(e); });
    show();
    return el("div", { class: "row" }, lab, input, dial);
  }
  const pct = (v: number) => `${v > 0 ? "+" : ""}${Math.round(v * 100)}`;
  const deg = (v: number) => `${v > 0 ? "+" : ""}${Math.round(v)}°`;
  const ev = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(2)}`;
  function chips<T extends string>(items: Array<{ id: T; label: string; extra?: string }>, current: T | undefined, pick: (id: T) => void): HTMLElement {
    return el("div", { class: "chips" }, ...items.map((it) => {
      const b = el("button", { class: "chip" + (it.id === current ? " on" : ""), text: it.label + (it.extra ? ` ${it.extra}` : "") });
      b.onclick = () => pick(it.id);
      return b;
    }));
  }
  function toggle(label: string, value: boolean, set: (v: boolean) => void): HTMLElement {
    const cb = el("input", { type: "checkbox" });
    cb.checked = value;
    cb.onchange = () => set(cb.checked);
    return el("label", { class: "toggle" }, label, cb);
  }

  // ------------------------------------------------------------------ adjust: per type
  let hsRange: HueRange = "master";
  function adjustBody(l: Layer): HTMLElement[] {
    switch (l.type) {
      case "curves": {
        const target = maskTarget(l.mask);
        const c = createToneCurves({
          histogram: (ch) => ctx.histogram(target, ch),
          get: () => l.params as LayerParams["curves"],
          set: (v) => { l.params = v; },
          changed: () => edit(),
          enabled: () => true,
        });
        curvesUi = c;
        return [c.el];
      }
      case "hueSat": {
        const h = l.params as LayerParams["hueSat"];
        const r = () => (h.ranges[hsRange] ??= { hue: 0, sat: 0, light: 0, inner: 15, outer: 45 });
        const out: HTMLElement[] = [];
        if (!h.colorize) {
          out.push(chips(HUE_RANGES.map((k) => ({ id: k, label: t(`hs.${k}`), extra: h.ranges[k] && (h.ranges[k]!.hue || h.ranges[k]!.sat || h.ranges[k]!.light) ? "•" : "" })), hsRange, (k) => { hsRange = k; renderProps(); }));
          out.push(slider(t("hs.hue"), -180, 180, 1, () => r().hue, (v) => (r().hue = v), deg, 0));
          out.push(slider(t("hs.sat"), -1, 1, 0.01, () => r().sat, (v) => (r().sat = v), pct, 0));
          out.push(slider(t("hs.light"), -1, 1, 0.01, () => r().light, (v) => (r().light = v), pct, 0));
          if (hsRange !== "master") out.push(rangeBar(h, hsRange));
        } else {
          out.push(slider(t("hs.hue"), 0, 360, 1, () => h.cHue, (v) => (h.cHue = v), (v) => `${Math.round(v)}°`, 30));
          out.push(slider(t("hs.sat"), 0, 1, 0.01, () => h.cSat, (v) => (h.cSat = v), (v) => String(Math.round(v * 100)), 0.25));
          out.push(slider(t("hs.light"), -1, 1, 0.01, () => h.cLight, (v) => (h.cLight = v), pct, 0));
        }
        out.push(toggle(t("hs.colorize"), h.colorize, (v) => { h.colorize = v; edit(); renderProps(); }));
        return out;
      }
      case "gradientMap": {
        const g = l.params as LayerParams["gradientMap"];
        return [createGradientEditor(() => g, (label) => edit(label), slider, { photoColors: ctx.photoColors }),
          el("div", { class: "muted grad-tip", text: t("grad.mapTip") })];
      }
      case "gradientFill": {
        const g = l.params as LayerParams["gradientFill"];
        const pctv = (v: number) => `${Math.round(v * 100)}%`;
        return [createGradientEditor(() => g, (label) => edit(label), slider, { photoColors: ctx.photoColors }),
          el("div", { class: "group-title", text: t("grad.shape") }),
          chips([{ id: "linear", label: t("grad.linear") }, { id: "radial", label: t("grad.radial") }] as const, g.style, (v) => { g.style = v; edit(); renderProps(); }),
          ...(g.style === "linear" ? [slider(t("grad.angle"), -180, 180, 1, () => g.angle, (v) => (g.angle = v), (v) => `${Math.round(v)}°`, 90)] : []),
          slider(t("grad.scale"), 0.1, 2, 0.01, () => g.scale, (v) => (g.scale = v), pctv, 1),
          slider(t("grad.x"), 0, 1, 0.01, () => g.x, (v) => (g.x = v), pctv, 0.5),
          slider(t("grad.y"), 0, 1, 0.01, () => g.y, (v) => (g.y = v), pctv, 0.5)];
      }
      case "blur": {
        const b = l.params as LayerParams["blur"];
        return [chips([{ id: "lens", label: t("blurl.lens") }, { id: "motion", label: t("blurl.motion") }] as const, b.motion ? "motion" : "lens", (v) => { b.motion = v === "motion"; edit(); renderProps(); }),
          slider(t("blurl.amount"), 0, 1, 0.01, () => b.amount, (v) => (b.amount = v), (v) => `${Math.round(v * 100)}%`, 0.4),
          ...(b.motion ? [angleRow(t("blurl.angle"), () => b.angle ?? 0, (v) => (b.angle = v))] : []),
          el("p", { class: "muted", text: t(b.motion ? "blurl.motionHint" : "blurl.hint") })];
      }
      case "brightContrast": {
        const b = l.params as LayerParams["brightContrast"];
        return [slider(t("bc.brightness"), -1, 1, 0.01, () => b.brightness, (v) => (b.brightness = v), pct, 0),
          slider(t("bc.contrast"), -1, 1, 0.01, () => b.contrast, (v) => (b.contrast = v), pct, 0)];
      }
      case "exposure": {
        const e = l.params as LayerParams["exposure"];
        return [slider(t("ex.exposure"), -3, 3, 0.01, () => e.exposure, (v) => (e.exposure = v), ev, 0),
          slider(t("ex.offset"), -0.1, 0.1, 0.001, () => e.offset, (v) => (e.offset = v), (v) => v.toFixed(3), 0),
          slider(t("ex.gamma"), 0.3, 3, 0.01, () => e.gamma, (v) => (e.gamma = v), (v) => v.toFixed(2), 1)];
      }
      case "basic": {
        const b = l.params as LayerParams["basic"];
        return [slider(t("basic.exposure"), -2, 2, 0.01, () => b.exposure, (v) => (b.exposure = v), ev, 0),
          slider(t("basic.temp"), -1, 1, 0.01, () => b.temp, (v) => (b.temp = v), pct, 0),
          slider(t("basic.tint"), -1, 1, 0.01, () => b.tint, (v) => (b.tint = v), pct, 0),
          slider(t("basic.saturation"), -1, 1, 0.01, () => b.saturation, (v) => (b.saturation = v), pct, 0),
          slider(t("basic.vibrance"), -1, 1, 0.01, () => b.vibrance, (v) => (b.vibrance = v), pct, 0),
          slider(t("basic.hue"), -30, 30, 0.5, () => b.hue, (v) => (b.hue = v), deg, 0)];
      }
    }
    return [];
  }

  /** Hue/Saturation colour range: a rainbow with draggable limits (full effect inside, fading to the outer marks). */
  function rangeBar(h: LayerParams["hueSat"], k: HueRange): HTMLElement {
    const r = h.ranges[k]!;
    const centre = RANGE_CENTRE[k as Exclude<HueRange, "master">];
    const cv = el("canvas", { class: "hs-range" });
    const W = 600, H = 56;
    cv.width = W; cv.height = H;
    const x = (d: number) => ((((centre + d) % 360) + 360) % 360) / 360 * W;
    const draw = () => {
      const c = cv.getContext("2d")!;
      c.clearRect(0, 0, W, H);
      for (let i = 0; i < W; i++) { c.fillStyle = `hsl(${(i / W) * 360}, 90%, 55%)`; c.fillRect(i, 6, 1, 20); }
      c.fillStyle = "rgba(255,255,255,.18)";
      for (let d = -r.outer!; d <= r.outer!; d += 0.5) c.fillRect(x(d), 30, 1.2, 8);
      c.fillStyle = "rgba(255,255,255,.55)";
      for (let d = -r.inner!; d <= r.inner!; d += 0.5) c.fillRect(x(d), 30, 1.2, 8);
      c.fillStyle = "#fff";
      for (const d of [-r.outer!, -r.inner!, r.inner!, r.outer!]) { const px = x(d); c.beginPath(); c.moveTo(px, 28); c.lineTo(px - 7, 48); c.lineTo(px + 7, 48); c.closePath(); c.fill(); }
      lab.textContent = `${Math.round(centre - r.outer!)}° / ${Math.round(centre - r.inner!)}°    ${Math.round(centre + r.inner!)}° \\ ${Math.round(centre + r.outer!)}°`;
    };
    const lab = el("div", { class: "muted hs-range-label" });
    let which: "inner" | "outer" | undefined;
    const pos = (e: PointerEvent) => { const b = cv.getBoundingClientRect(); return ((e.clientX - b.left) / b.width) * 360; };
    const dist = (a: number) => { let d = Math.abs(a - centre) % 360; if (d > 180) d = 360 - d; return d; };
    cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture(e.pointerId); const d = dist(pos(e)); which = Math.abs(d - r.inner!) < Math.abs(d - r.outer!) ? "inner" : "outer"; });
    cv.addEventListener("pointermove", (e) => {
      if (!which) return;
      const d = Math.min(170, dist(pos(e)));
      if (which === "inner") r.inner = Math.min(d, r.outer! - 1); else r.outer = Math.max(d, r.inner! + 1);
      draw(); edit();
    });
    const up = () => { which = undefined; };
    cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
    draw();
    return el("div", { class: "hs-range-wrap" }, el("div", { class: "group-title", text: t("hs.range") }), cv, lab);
  }

  // ------------------------------------------------------------------ mask
  function maskTarget(m: SmartMask): HistTarget {
    if (m.kind === "region" && m.region) return m.region;
    if (m.kind === "distance" && m.band) return m.band;
    if (m.kind === "cell" && m.region && m.band && m.region !== "skin") return `${m.region}.${m.band}` as HistTarget;
    return "photo";
  }
  // Picking on the photo: while a layer's Mask tab is open, a tap selects. A tap on
  // something not in the mask adds it; a tap on something already in it removes it
  // (the same object tapped again is deselected; something inside a selection is cut
  // out of it). What a tap selects (object / colour / distance) is the one choice.
  // Each tap becomes a piece of the mask, listed below with its few settings.
  let picking = false;
  let pickAs: "object" | "color" | "depth" = "object";
  /** The piece whose settings are open ("main" or a part's index). */
  let open: "main" | number | undefined;
  function setPicking(on: boolean) {
    if (placing || on === picking) return;
    picking = on;
    ctx.pickMode(on, t("mask.tapHint"));
  }
  /** A lens flare waiting for a tap on its light (its set), or none. */
  let placing: string | undefined;
  function setPlacing(set: string | undefined) {
    placing = set;
    picking = !!set;
    ctx.pickMode(!!set, t("flare.tapLight"));
    if (!set) applyMaskView();
  }
  function placeAt(x: number, y: number): boolean {
    const p = ctx.params();
    if (!placing || !p) return false;
    // The flare is gone (undo, another photo): stop waiting, the tap is an ordinary one.
    if (!flareLight(p.layers, placing)) { setPlacing(undefined); return false; }
    moveFlare(p.layers, placing, { x, y });
    setPlacing(undefined);
    ctx.changed(t("flare.moved"));
    render();
    return true;
  }
  function shapeFromPick(info: PickInfo, as: typeof pickAs): Partial<MaskShape> {
    if (as === "color") return { kind: "color", color: info.color, tol: 0.08 };
    if (as === "depth") return { kind: "depth", depth: [Math.max(0, info.dist - 0.06), Math.min(1, info.dist + 0.06), 0.04] };
    // Object: a selection (tap-to-select) — one person of four, not every person.
    return { kind: "select", points: [[info.x, info.y, 1]], level: undefined };
  }
  /** Adds a piece: the main shape while the mask still covers everything (adding), otherwise a part. */
  function addPiece(l: Layer, shape: Partial<MaskShape>, op: MaskOp): boolean {
    const m = l.mask;
    const parts = (m.parts ??= []);
    const clean = { invert: false, feather: 1, points: undefined, color: undefined, depth: undefined, lum: undefined, region: undefined, band: undefined, level: undefined, tol: undefined, shape: undefined };
    if (m.kind === "all" && op === "add") { l.mask = { ...m, ...clean, ...shape } as SmartMask; open = "main"; return true; }
    if (parts.length >= MAX_MASK_PARTS) return false;
    parts.push({ ...clean, ...shape, op } as MaskPart);
    open = parts.length - 1;
    return true;
  }
  function onPick(info: PickInfo, tapped?: { layer: string; as: typeof pickAs }): boolean {
    const l = sel();
    if (!l || !picking) return false;
    // The answer belongs to the layer (and reading) the tap was for: another one selected meanwhile gets nothing.
    if (tapped && tapped.layer !== l.id) return false;
    const as = tapped?.as ?? pickAs;
    const m = l.mask;
    // Nothing chosen yet (the mask is the whole photo): a tap always adds.
    const empty = m.kind === "all" && !(m.parts ?? []).length;
    const inside = !empty && (info.inMask ?? 0) > 0.5;
    if (inside && info.sameAs) {
      // The same object tapped again: deselect it.
      if (m.kind === "select" && selectKey(m) === info.sameAs) removePiece(l, "main");
      else { const i = (m.parts ?? []).findIndex((q) => q.op === "add" && q.kind === "select" && selectKey(q) === info.sameAs); if (i >= 0) removePiece(l, i); }
      edit(t("hist.maskRemove", { name: layerName(l) }));
      renderProps();
      return true;
    }
    if (!addPiece(l, shapeFromPick(info, as), inside ? "subtract" : "add")) { ctx.notice?.(t("mask.full")); return false; }
    edit(t(inside ? "hist.maskRemove" : "hist.maskAdd", { name: layerName(l) }));
    renderProps();
    return true;
  }
  /** Removes one piece; the next added part takes the main place (a mask of only removals keeps "everything" under them). */
  function removePiece(l: Layer, key: "main" | number) {
    const m = l.mask;
    const parts = m.parts ?? [];
    if (key === "main") {
      // The first added part takes the main place (removals keep working on it); with none,
      // "everything" is under the removals.
      const i = parts.findIndex((q) => q.op === "add");
      const next = i >= 0 ? parts.splice(i, 1)[0] : undefined;
      const clean = { points: undefined, color: undefined, depth: undefined, lum: undefined, region: undefined, band: undefined, level: undefined, tol: undefined, shape: undefined };
      l.mask = next ? ({ ...m, ...clean, ...next, op: undefined, parts } as SmartMask) : { ...m, ...clean, kind: "all", invert: false, feather: 1, parts };
    } else parts.splice(key, 1);
    open = undefined;
  }

  const levelName = (lv: MaskShape["level"]) => t(lv === undefined ? "mask.level.auto" : `mask.level.${lv}`);
  const pct100 = (v: number) => String(Math.round(v * 100));
  /** A piece in one line: what it selects. */
  function pieceSummary(sh: MaskShape): Node[] {
    const cov = ctx.coverage() ?? {};
    const reg = (r?: Region) => (r ? tOr(`group.${r}`, r) : "");
    switch (sh.kind) {
      case "select": return [document.createTextNode(`${t("mask.pick.object")} · ${levelName(sh.level)}`)];
      case "color": {
        const lin = oklabToLinSrgb(sh.color ?? [0.6, 0, 0]).map((v) => Math.round(255 * Math.min(1, Math.max(0, v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055))));
        return [el("span", { class: "mask-dot", style: `background: rgb(${lin.join(",")})` }), document.createTextNode(t("mask.pick.color"))];
      }
      case "depth": return [document.createTextNode(`${t("mask.pick.depth")} ${pct100(sh.depth?.[0] ?? 0)}–${pct100(sh.depth?.[1] ?? 1)}`)];
      case "region": return [document.createTextNode(`${reg(sh.region)}${cov[sh.region ?? ""] !== undefined ? ` ${Math.round(cov[sh.region!])}%` : ""}`)];
      case "object": return [document.createTextNode(`${reg(sh.region)} · ${pct100(sh.depth?.[0] ?? 0)}–${pct100(sh.depth?.[1] ?? 1)}`)];
      case "distance": return [document.createTextNode(t(`band.${sh.band ?? "near"}`))];
      case "cell": return [document.createTextNode(`${reg(sh.region)} · ${t(`band.${sh.band ?? "near"}`).toLowerCase()}`)];
      case "luminance": return [document.createTextNode(`${t("mask.luminance")} ${pct100(sh.lum?.[0] ?? 0)}–${pct100(sh.lum?.[1] ?? 1)}`)];
      case "shape": return [document.createTextNode(sh.shape?.style === "radial" ? t("grad.radial") : `${t("grad.linear")} ${Math.round(sh.shape?.angle ?? 90)}°`)];
      default: return [document.createTextNode(t("mask.all"))];
    }
  }

  /** A piece's settings: only what its kind has, then invert and soft edge. */
  function shapeFields(sh: MaskShape, set: (patch: Partial<MaskShape>) => void): HTMLElement[] {
    const cov = ctx.coverage() ?? {};
    const cc = ctx.cellCoverage() ?? {};
    const regions: Region[] = GROUPS.filter((g) => (cov[g] ?? 0) >= 0.5);
    if ((cov.person ?? 0) >= 0.5) regions.splice(regions.indexOf("person") + 1, 0, "skin");
    const out: HTMLElement[] = [];
    if (sh.kind === "region" || sh.kind === "cell" || sh.kind === "object") {
      const rs = sh.kind === "cell" ? regions.filter((r) => r !== "skin") : regions;
      if (sh.region && !rs.includes(sh.region)) rs.push(sh.region);
      out.push(chips(rs.map((r) => ({ id: r, label: tOr(`group.${r}`, r), extra: cov[r] !== undefined ? `${Math.round(cov[r])}%` : "" })), sh.region, (r) => set({ region: r })));
    }
    if (sh.kind === "distance" || sh.kind === "cell") {
      out.push(chips(DEPTH_BANDS.map((b) => ({ id: b, label: t(`band.${b}`), extra: sh.kind === "cell" && sh.region && sh.region !== "skin" ? `${Math.round(cc[`${sh.region}.${b}`] ?? 0)}%` : "" })), sh.band, (b) => set({ band: b })));
    }
    if (sh.kind === "luminance") {
      const lum = (sh.lum ??= [0, 0.3, 0.08]);
      out.push(slider(t("mask.low"), 0, 1, 0.01, () => lum[0], (v) => (lum[0] = v), pct100, 0));
      out.push(slider(t("mask.high"), 0, 1, 0.01, () => lum[1], (v) => (lum[1] = v), pct100, 0.3));
      out.push(slider(t("mask.soft"), 0.01, 0.3, 0.01, () => lum[2], (v) => (lum[2] = v), pct100, 0.08));
    }
    if (sh.kind === "depth" || sh.kind === "object") {
      const d = (sh.depth ??= [0, 0.3, 0.05]);
      out.push(slider(t("mask.low"), 0, 1, 0.01, () => d[0], (v) => (d[0] = v), pct100, 0));
      out.push(slider(t("mask.high"), 0, 1, 0.01, () => d[1], (v) => (d[1] = v), pct100, 0.3));
      out.push(slider(t("mask.soft"), 0.005, 0.2, 0.005, () => d[2], (v) => (d[2] = v), pct100, 0.05));
      out.push(el("p", { class: "muted", text: t("mask.depthHint") }));
    }
    if (sh.kind === "select") {
      // How much of what was tapped: SAM's own best reading, or whole / part / detail (a person / their clothes / a piece).
      out.push(chips<"auto" | "0" | "1" | "2">(([undefined, 0, 1, 2] as const).map((lv) => ({ id: lv === undefined ? "auto" : (String(lv) as "0"), label: levelName(lv) })),
        sh.level === undefined ? "auto" : (String(sh.level) as "0"), (v) => set({ level: v === "auto" ? undefined : (+v as 0 | 1 | 2) })));
    }
    if (sh.kind === "shape") {
      // Laid out as the Gradient Fill layer: the same controls, the same meaning.
      const g = (sh.shape ??= defaultShape("linear"));
      const pctv = (v: number) => `${Math.round(v * 100)}%`;
      out.push(chips<"linear" | "radial">([{ id: "linear", label: t("grad.linear") }, { id: "radial", label: t("grad.radial") }], g.style,
        (st) => set({ shape: st === g.style ? g : { ...defaultShape(st), x: g.x, y: g.y } })));
      if (g.style === "linear") out.push(slider(t("grad.angle"), -180, 180, 1, () => g.angle, (v) => (g.angle = v), (v) => `${Math.round(v)}°`, 90));
      out.push(slider(t("grad.scale"), 0.05, 2, 0.01, () => g.scale, (v) => (g.scale = v), pctv, g.style === "radial" ? 0.6 : 1));
      out.push(slider(t("grad.x"), 0, 1, 0.01, () => g.x, (v) => (g.x = v), pctv, 0.5));
      out.push(slider(t("grad.y"), 0, 1, 0.01, () => g.y, (v) => (g.y = v), pctv, 0.5));
      out.push(slider(t("mask.soft"), 0, 1, 0.01, () => g.soft, (v) => (g.soft = v), pctv, g.style === "radial" ? 0.5 : 0.6));
    }
    if (sh.kind === "color") out.push(slider(t("mask.tol"), 0.01, 0.3, 0.005, () => sh.tol ?? 0.08, (v) => (sh.tol = v), pct100, 0.08));
    if (sh.kind !== "all") {
      out.push(slider(t("mask.feather"), 0, 1, 0.01, () => sh.feather, (v) => (sh.feather = v), (v) => `${Math.round(v * 100)}%`, 1));
      out.push(toggle(t("mask.invert"), sh.invert, (v) => set({ invert: v })));
    }
    return out;
  }

  function maskBody(l: Layer): HTMLElement[] {
    const m = l.mask;
    const parts = m.parts ?? [];
    const changed = () => { edit(t("hist.mask", { name: layerName(l) })); renderProps(); };

    // What a tap selects (the only choice: adding or removing follows from where you tap).
    const out: HTMLElement[] = [
      el("div", { class: "mask-mode" },
        el("span", { class: "muted", text: t("mask.tapSelects") }),
        chips<typeof pickAs>([{ id: "object", label: t("mask.pick.object") }, { id: "color", label: t("mask.pick.color") }, { id: "depth", label: t("mask.pick.depth") }], pickAs, (a) => { pickAs = a; renderProps(); })),
    ];

    // The photo's regions (the scene analysis), one tap each: on adds the region, off removes it.
    {
      const cov = ctx.coverage() ?? {};
      const regions: Region[] = GROUPS.filter((g) => (cov[g] ?? 0) >= 0.5);
      if ((cov.person ?? 0) >= 0.5) regions.splice(regions.indexOf("person") + 1, 0, "skin");
      const findRegion = (r: Region): "main" | number | undefined => {
        if (m.kind === "region" && m.region === r && !m.invert) return "main";
        const i = parts.findIndex((q) => q.kind === "region" && q.region === r && q.op === "add" && !q.invert);
        return i >= 0 ? i : undefined;
      };
      if (regions.length) out.push(el("div", { class: "mask-regions" },
        el("span", { class: "muted", text: t("mask.regions") }),
        chips<Region>(regions.map((r) => ({ id: r, label: tOr(`group.${r}`, r), extra: cov[r] !== undefined ? `${Math.round(cov[r])}%` : "" })), undefined, (r) => {
          const k = findRegion(r);
          if (k !== undefined) removePiece(l, k);
          else if (!addPiece(l, { kind: "region", region: r }, "add")) { ctx.notice?.(t("mask.full")); return; }
          changed();
        })));
      // Chips that are on show as on.
      const last = out[out.length - 1];
      if (regions.length) for (const b of last.querySelectorAll<HTMLElement>(".chip")) {
        const r = regions[[...last.querySelectorAll(".chip")].indexOf(b)];
        b.classList.toggle("on", findRegion(r) !== undefined);
      }
    }

    // Shapes: a graduated (linear) or radial filter as a piece of the mask, one tap each.
    out.push(el("div", { class: "mask-regions" },
      el("span", { class: "muted", text: t("mask.shapes") }),
      chips<"linear" | "radial">([{ id: "linear", label: t("grad.linear") }, { id: "radial", label: t("grad.radial") }], undefined, (st) => {
        if (addPiece(l, { kind: "shape", shape: defaultShape(st) }, "add")) changed(); else ctx.notice?.(t("mask.full"));
      })));

    // The pieces.
    type Key = "main" | number;
    const pieces: Array<{ key: Key; sh: MaskShape; op?: MaskOp }> = [
      ...(m.kind !== "all" ? [{ key: "main" as Key, sh: m as MaskShape }] : []),
      ...parts.map((p, i) => ({ key: i as Key, sh: p as MaskShape, op: p.op })),
    ];
    if (!pieces.length) out.push(el("p", { class: "muted mask-empty", text: t("mask.empty") }));
    for (const { key, sh, op } of pieces) {
      const setPiece = (patch: Partial<MaskShape>) => {
        if (key === "main") l.mask = { ...m, ...patch };
        else parts[key] = { ...parts[key], ...patch } as MaskPart;
        changed();
      };
      const remove = el("button", { class: "mask-x", title: t("mask.remove"), "aria-label": t("mask.remove") }, icon("close", 15));
      remove.onclick = (e) => { e.stopPropagation(); removePiece(l, key); changed(); };
      const sign = el("span", { class: "mask-sign", text: key === "main" ? "+" : op === "subtract" ? "−" : op === "intersect" ? "∩" : "+" });
      const row = el("div", { class: "mask-piece" + (open === key ? " open" : "") + (sh.invert ? " inv" : "") }, sign, el("span", { class: "mask-what" }, ...pieceSummary(sh)), remove);
      row.onclick = () => { open = open === key ? undefined : key; renderProps(); };
      out.push(row);
      if (open === key) {
        const body = el("div", { class: "mask-piece-body" });
        if (key !== "main") body.append(chips<MaskOp>(MASK_OPS.map((o) => ({ id: o, label: t(`mask.op.${o}`) })), op, (o) => setPiece({ op: o } as Partial<MaskShape>)));
        body.append(...shapeFields(sh, setPiece));
        out.push(body);
      }
    }
    if (parts.length >= MAX_MASK_PARTS) out.push(el("p", { class: "muted", text: t("mask.full") }));

    // More: masks by type (no tap needed), and the whole mask's strength.
    const more = el("details", { class: "mask-more" }, el("summary", { text: t("mask.more") }));
    const addBy = chips<MaskKind>((["luminance", "distance", "cell"] as MaskKind[]).map((k) => ({ id: k, label: t(`mask.${k}`) })), undefined, (k) => {
      const cov = ctx.coverage() ?? {};
      const region = (GROUPS.find((g) => (cov[g] ?? 0) >= 5) ?? "sky") as Region;
      if (addPiece(l, { kind: k, region, band: "near", lum: [0, 0.3, 0.08] }, "add")) changed();
    });
    const clear = el("button", { class: "btn small", text: t("mask.clear") });
    clear.onclick = () => { l.mask = { kind: "all", invert: false, feather: 1, density: m.density, exceptSkin: m.exceptSkin }; open = undefined; changed(); };
    more.append(el("div", { class: "muted", text: t("mask.addBy") }), addBy,
      slider(t("mask.density"), 0, 1, 0.01, () => m.density, (v) => (m.density = v), (v) => `${Math.round(v * 100)}%`, 1),
      el("div", { class: "actions" }, clear));
    more.open = moreOpen;
    more.ontoggle = () => { moreOpen = more.open; };
    out.push(more);
    out.push(toggle(t("mask.show"), showMask, (v) => { showMask = v; applyMaskView(); }));
    return out;
  }
  let moreOpen = false;
  // Only changes reach the app (each one re-renders the photo and resets other views).
  let maskSent: number | undefined | null = null;
  function applyMaskView() {
    const l = sel();
    const i = l ? liveIndex(l.id) : -1;
    const want = visible && tab === "mask" && showMask && i >= 0 ? i : undefined;
    // Taps pick while the selected layer's Mask tab is open, and only then.
    // (A hidden layer has no mask to read under a tap: no picking there.)
    setPicking(visible && tab === "mask" && !!l && i >= 0);
    if (want === maskSent) return;
    maskSent = want;
    ctx.showMask(want);
  }

  // ------------------------------------------------------------------ blend
  function blendBody(l: Layer): HTMLElement[] {
    const pick = (b: Layer["blend"]) => { l.blend = b; edit(t("hist.blend", { name: layerName(l) })); renderProps(); };
    const modes = el("div", { class: "blend-groups" }, ...BLEND_GROUPS.map((g) =>
      chips(g.modes.map((b) => ({ id: b, label: t(`blend.${b}`) })), l.blend, pick)));
    const bi = l.blendIf;
    const range = (which: "this" | "under") => {
      const get = () => l.blendIf?.[which] ?? fullRange();
      const set = (r: Partial<ReturnType<typeof fullRange>>) => {
        l.blendIf ??= { this: fullRange(), under: fullRange() };
        Object.assign(l.blendIf[which], r);
        const { this: a, under: b } = l.blendIf;
        if (a.low <= 0 && a.high >= 1 && b.low <= 0 && b.high >= 1) l.blendIf = undefined; // nothing limited
      };
      return [
        el("div", { class: "blendif-label", text: t(`blend.if.${which}`) }),
        lumaBar(() => get(), (low, high) => set({ low, high })),
        slider(t("blend.if.soft"), 0, 0.5, 0.01, () => get().soft, (v) => set({ soft: v }), (v) => `${Math.round(v * 100)}`, 0.2),
      ];
    };
    return [
      el("div", { class: "group-title", text: t("blend.mode") }),
      modes,
      slider(t("blend.opacity"), 0, 1, 0.01, () => l.opacity, (v) => (l.opacity = v), (v) => `${Math.round(v * 100)}%`, 1),
      el("div", { class: "group-title", text: t("blend.if") }),
      el("p", { class: "muted", text: t("blend.ifTip") }),
      ...range("this"),
      ...range("under"),
      ...(bi ? [(() => { const b = el("button", { class: "btn small ghost", text: t("blend.if.reset") }); b.onclick = () => { l.blendIf = undefined; edit(); renderProps(); }; return b; })()] : []),
    ];
  }

  /**
   * A brightness range with two handles (black → white bar): drag a handle, or tap
   * the bar to move the nearer one there. Vertical moves scroll the panel.
   */
  function lumaBar(get: () => { low: number; high: number }, set: (low: number, high: number) => void): HTMLElement {
    const lo = el("span", { class: "rb-knob" }), hi = el("span", { class: "rb-knob" });
    const shade = [el("span", { class: "rb-off" }), el("span", { class: "rb-off" })];
    const bar = el("div", { class: "rangebar", role: "group" }, shade[0], shade[1], lo, hi);
    const show = () => {
      const r = get();
      lo.style.left = `${r.low * 100}%`; hi.style.left = `${r.high * 100}%`;
      shade[0].style.cssText = `left:0;width:${r.low * 100}%`; shade[1].style.cssText = `left:${r.high * 100}%;right:0`;
      bar.title = `${Math.round(r.low * 255)} – ${Math.round(r.high * 255)}`;
    };
    let which: "low" | "high" | undefined, x0 = 0, y0 = 0, moving = false;
    const at = (x: number) => { const b = bar.getBoundingClientRect(); return Math.min(1, Math.max(0, (x - b.left) / Math.max(1, b.width))); };
    const apply = (x: number) => {
      const v = Math.round(at(x) * 255) / 255, r = get();
      if (which === "low") set(Math.min(v, r.high), r.high); else set(r.low, Math.max(v, r.low));
      show(); edit();
    };
    bar.addEventListener("pointerdown", (e) => {
      const v = at(e.clientX), r = get();
      // The nearer handle; when both are at one spot, the one on the side of the touch.
      which = r.low === r.high ? (v < r.low ? "low" : "high") : Math.abs(v - r.low) < Math.abs(v - r.high) ? "low" : "high";
      x0 = e.clientX; y0 = e.clientY; moving = false;
      bar.setPointerCapture(e.pointerId);
    });
    bar.addEventListener("pointermove", (e) => {
      if (!which) return;
      if (!moving) {
        if (Math.hypot(e.clientX - x0, e.clientY - y0) < 6) return;
        if (Math.abs(e.clientY - y0) > Math.abs(e.clientX - x0)) { which = undefined; return; } // a scroll
        moving = true;
      }
      apply(e.clientX);
    });
    bar.addEventListener("pointerup", (e) => { if (which && !moving) apply(e.clientX); which = undefined; });
    bar.addEventListener("pointercancel", () => { which = undefined; });
    show();
    return bar;
  }

  // ------------------------------------------------------------------ properties
  let curvesUi: ReturnType<typeof createToneCurves> | undefined;
  function renderProps() {
    curvesUi = undefined;
    const l = sel();
    if (!l) {
      if (selected !== "blur" && selected !== "toneEq" && selected !== "contrastEq" && selected !== "film") selected = "develop";
      if (selected !== "blur") ctx.leftBlur?.();
      if (selected !== "toneEq") ctx.toneEq?.leave();
      if (selected !== "contrastEq") ctx.contrastEq?.leave();
      const own = selected === "toneEq" ? ctx.toneEq : selected === "contrastEq" ? ctx.contrastEq : selected === "film" ? ctx.film : undefined;
      if (own) { props.replaceChildren(own.el); own.render(); }
      else props.replaceChildren(selected === "blur" ? ctx.blur : ctx.develop);
      applyMaskView();
      return;
    }
    ctx.leftBlur?.();
    ctx.toneEq?.leave();
    ctx.contrastEq?.leave();
    const name = el("input", { class: "lay-title", value: layerName(l), "aria-label": t("lay.name") });
    name.onchange = () => { l.name = name.value.trim() || layerName(l); (l as Layer & { renamed?: boolean }).renamed = true; edit(); renderDock(); };
    const act = (ic: Parameters<typeof icon>[0], label: string, fn: () => void) => { const b = el("button", { class: "btn small icon ghost", title: label, "aria-label": label }, icon(ic, 19)); b.onclick = fn; return b; };
    const p = ctx.params()!;
    const titleBox = el("div", { class: "lay-titlebox" }, name);
    if (l.auto) titleBox.append(el("span", { class: "lay-sub", text: `${typeName(l.type)} · ${t("lay.auto")}` }));
    else titleBox.append(el("span", { class: "lay-sub", text: typeName(l.type) }));
    const head = el("div", { class: "lay-head" }, el("span", { class: "lay-glyph big" }, icon(l.type, 20)), titleBox,
      act(l.visible ? "eye" : "eyeOff", t("lay.visible"), () => { l.visible = !l.visible; edit(t(l.visible ? "hist.show" : "hist.hide", { name: layerName(l) })); render(); }),
      act("copy", t("lay.duplicate"), () => {
        const c = { ...structuredClone(l), id: newId(), auto: undefined, name: `${layerName(l)} 2` };
        p.layers.splice(p.layers.indexOf(l) + 1, 0, c);
        selected = c.id; ctx.changed(t("hist.duplicate", { name: layerName(l) })); render();
      }),
      act("trash", t("lay.delete"), () => {
        const i = p.layers.indexOf(l);
        p.layers.splice(i, 1);
        selected = p.layers[Math.min(i, p.layers.length - 1)]?.id ?? "develop";
        ctx.changed(t("hist.delete", { name: layerName(l) })); render();
      }),
    );
    if (l.auto) {
      const orig = ctx.auto()?.layers.find((x) => x.auto === l.auto);
      if (orig) head.append(act("reset", t("lay.resetAuto"), () => {
        delete (l as Layer & { renamed?: boolean }).renamed;
        Object.assign(l, structuredClone({ ...orig, id: l.id }));
        ctx.changed(t("hist.reset", { name: layerName(l) })); render();
      }));
    }
    const segs = el("div", { class: "seg" }, ...(["adjust", "mask", "blend"] as const).map((k) => {
      const b = el("button", { class: k === tab ? "on" : "", text: t(`lay.tab.${k}`) });
      b.onclick = () => { tab = k; renderProps(); };
      return b;
    }));
    const body = tab === "adjust" ? adjustBody(l) : tab === "mask" ? maskBody(l) : blendBody(l);
    const flareRow: HTMLElement[] = [];
    if (l.flare) {
      const set = l.flare.set;
      const move = el("button", { class: "btn small" + (placing === set ? " primary" : ""), text: placing === set ? t("flare.tapLight") : t("flare.move") });
      move.onclick = () => { setPlacing(placing === set ? undefined : set); renderProps(); };
      const del = el("button", { class: "btn small", text: t("flare.remove") });
      del.onclick = () => {
        p.layers = p.layers.filter((x) => x.flare?.set !== set);
        if (placing === set) setPlacing(undefined);
        selected = "develop"; ctx.changed(t("flare.remove")); render();
      };
      flareRow.push(el("div", { class: "actions flare-row" }, move, del));
    }
    props.replaceChildren(head, ...flareRow, segs, ...body);
    applyMaskView();
  }

  function render() { renderDock(); renderProps(); }

  return {
    render,
    /** New histograms: redraw the curve box only. */
    refreshHistogram() { curvesUi?.refreshHistogram(); },
    /** The panel became visible / hidden (mask view only while it is shown). */
    setVisible(v: boolean) { visible = v; applyMaskView(); },
    selectDevelop() { selected = "develop"; render(); },
    /** A tap on the photo while a lens flare waits for its light: taken (true), or not a placing tap. */
    placeAt(x: number, y: number): boolean { return placeAt(x, y); },
    /** The engine's answer to a tap in pick mode. */
    onPick(info: PickInfo, tapped?: { layer: string; as: "object" | "color" | "depth" }): boolean { return onPick(info, tapped); },
    /** What a tap on the photo edits now: the selected layer (index among the live layers) and whether taps select objects. */
    pickTarget(): { layer: number; object: boolean; id: string; as: "object" | "color" | "depth" } | undefined {
      const l = sel();
      if (!picking || !l) return undefined;
      return { layer: liveIndex(l.id), object: pickAs === "object", id: l.id, as: pickAs };
    },
    /** Pick mode ended from outside (another photo tool took the taps). */
    stopPicking() { if (picking) { setPicking(false); renderProps(); } },
    /** A new photo: nothing of the previous photo's editing state survives (tab, open piece, picking, a flare waiting for its light). */
    reset() {
      tab = "adjust"; open = undefined; maskSent = null; placing = undefined;
      if (picking) { picking = false; ctx.pickMode(false); }
    },
  };
}

