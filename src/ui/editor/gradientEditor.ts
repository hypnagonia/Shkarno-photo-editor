/**
 * Gradient editor, as easy as it can be:
 *   bar       the gradient the layer has
 *   quick     New palette (one tap: another set of colours that belong together),
 *             From photo, and how many colours (3 · 4 · 5)
 *   palettes  every preset as a tile of colour blocks: one tap applies
 *   edit      (folded) harmony by base colour and rule; the stops under the bar
 *             (tap the bar to add one, drag to move, drag down to remove) and the
 *             selected stop's colour, opacity and location
 */
import {
  GRADIENT_GROUPS, GRADIENT_PRESETS, HARMONY_RULES, gradientAt, withAlphaOf, gradientCss, gradientFrom, harmonyPalette, hexToOklch, oklchToHex, paletteCss, paletteFromColors, rgbToHex,
  type Gradient, type HarmonyRule,
} from "../../layers/gradient.ts";
import { t } from "../i18n.ts";
import { icon } from "./icons.ts";
import { el } from "../dom.ts";

type Target = { gradient: Gradient; reverse: boolean; preset?: string };


/** Optional extras from the app. */
export interface GradientEditorOptions {
  /** The photo's dominant colours (hex): enables "From photo". */
  photoColors?: () => Promise<string[]>;
}

/**
 * `changed(label?)`: the gradient was edited (a label = a finished step, e.g. a preset).
 * `slider` builds the app's standard slider row.
 */
export function createGradientEditor(target: () => Target, changed: (label?: string) => void,
  slider: (label: string, min: number, max: number, step: number, get: () => number, set: (v: number) => void, fmt: (v: number) => string, def: number) => HTMLElement,
  opts: GradientEditorOptions = {}): HTMLElement {
  const root = el("div", { class: "grad" });
  let selected = 0;
  /** A palette becomes the gradient: a fresh copy (later edits must not change the source), smooth interpolation. */
  const apply = (colors: string[], preset: string | undefined, label: string) => {
    const tg = target();
    // (Over a gradient that fades, the fade stays: only the colours change.)
    tg.gradient = withAlphaOf(gradientFrom(colors, "oklab"), tg.gradient); tg.preset = preset; selected = 0;
    changed(label);
    render();
  };

  // ---- palettes: every preset as a tile of colour blocks, in group order (Palettes first)
  const presets = el("div", { class: "grad-tiles" });
  function renderPresets() {
    const all = GRADIENT_GROUPS.flatMap((g) => GRADIENT_PRESETS.filter((p) => p.group === g));
    presets.replaceChildren(...all.map((p) => {
      const name = t(`grad.p.${p.id}` as never);
      const b = el("button", { class: "grad-tile" + (p.id === target().preset ? " on" : ""), title: name, "aria-label": name, "data-id": p.id });
      b.style.background = paletteCss(p.colors);
      b.onclick = () => apply(p.colors, p.id, t("grad.applied", { name }));
      return b;
    }));
  }

  // ---- harmony: palettes that belong together, from one base colour
  const mostChromatic = (g: Gradient) => [...g.stops].sort((a, b) => hexToOklch(b.color)[1] - hexToOklch(a.color)[1])[0]?.color ?? "#1D6A73";
  let base = mostChromatic(target().gradient);
  let rule: HarmonyRule = "analogous";
  let seed = 0;
  const baseInput = el("input", { type: "color", class: "grad-color", value: base.toLowerCase(), "aria-label": t("grad.base") });
  baseInput.oninput = () => { base = baseInput.value.toUpperCase(); seed = 0; renderHarmony(); };
  const ruleChips = el("div", { class: "chips grad-rules" });
  const preview = el("button", { class: "grad-harmony-preview", "aria-label": t("grad.tapApply") }, el("span", { class: "grad-swatch" }), el("span", { class: "grad-pname", text: t("grad.tapApply") }));
  preview.onclick = () => apply(harmonyPalette(base, rule, seed, count), undefined, t("grad.harmonyApplied", { name: t(`grad.rule.${rule}`) }));
  const vary = el("button", { class: "btn small", text: t("grad.vary") });
  vary.onclick = () => { seed++; renderHarmony(); };
  const harmonyRow = el("div", { class: "grad-stoprow" }, el("label", { class: "grad-colorwrap", title: t("grad.base") }, baseInput), preview, vary);
  // ---- quick: a new palette in one tap, or the photo's, in 3–5 colours
  let count = Math.min(5, Math.max(3, target().gradient.stops.length));
  /** What the last quick tap made, so changing the count remakes it. */
  let last: "new" | "photo" | undefined;
  let genSeed = Math.floor(Math.random() * 1e6);
  const QUICK_RULES: HarmonyRule[] = ["complementary", "split", "triadic", "analogous", "warmCool"];
  const makeNew = () => {
    // Spread over the colour wheel (golden angle) and the harmony rules: every tap is different.
    const hue = (genSeed * 137.508) % 360;
    const rr = QUICK_RULES[genSeed % QUICK_RULES.length];
    return { colors: harmonyPalette(oklchToHex(0.62, 0.14, hue), rr, genSeed % 7, count), rule: rr };
  };
  const newBtn = el("button", { class: "btn small primary", text: t("grad.new") });
  newBtn.onclick = () => { genSeed++; last = "new"; const g = makeNew(); apply(g.colors, undefined, t("grad.harmonyApplied", { name: t(`grad.rule.${g.rule}`) })); };
  const countChips = el("div", { class: "chips grad-count" });
  function renderCount() {
    countChips.replaceChildren(...[3, 4, 5].map((n) => {
      const b = el("button", { class: "chip" + (n === count ? " on" : ""), text: String(n), title: t("grad.count", { n }) });
      b.onclick = () => {
        count = n; renderCount();
        if (last === "new") { const g = makeNew(); apply(g.colors, undefined, t("grad.harmonyApplied", { name: t(`grad.rule.${g.rule}`) })); }
        else if (last === "photo") photoBtn.click();
      };
      return b;
    }));
  }
  const photoBtn = el("button", { class: "btn small", text: t("grad.fromPhoto") });
  photoBtn.hidden = !opts.photoColors;
  photoBtn.onclick = async () => {
    if (!opts.photoColors) return;
    photoBtn.disabled = true;
    try {
      const cols = await opts.photoColors();
      if (!cols.length) return;
      // The photo's own colours as the map; its most colourful one becomes the harmony base.
      base = [...cols].sort((a, b) => hexToOklch(b)[1] - hexToOklch(a)[1])[0].toUpperCase();
      baseInput.value = base.toLowerCase(); seed = 0;
      last = "photo";
      apply(paletteFromColors(cols, count), undefined, t("grad.photoApplied"));
    } finally { photoBtn.disabled = false; }
  };
  function renderHarmony() {
    ruleChips.replaceChildren(...HARMONY_RULES.map((k) => {
      const b = el("button", { class: "chip" + (k === rule ? " on" : ""), text: t(`grad.rule.${k}`) });
      b.onclick = () => { rule = k; seed = 0; renderHarmony(); };
      return b;
    }));
    (preview.firstChild as HTMLElement).style.background = paletteCss(harmonyPalette(base, rule, seed, count));
  }

  // ---- bar + stops
  const bar = el("div", { class: "grad-bar" }, el("div", { class: "grad-fill" }));
  const stopsRow = el("div", { class: "grad-stops" });
  const stopBody = el("div", { class: "grad-stop" });
  const reverseBtn = el("button", { class: "chip", text: t("grad.reverse") });
  reverseBtn.onclick = () => { const tg = target(); tg.reverse = !tg.reverse; changed(t("grad.reverse")); render(); };
  // Smooth: interpolate in OkLab (even perceived steps) instead of classic encoded sRGB.
  const smoothBtn = el("button", { class: "chip", text: t("grad.smooth"), title: t("grad.smoothTip") });
  smoothBtn.onclick = () => { const g = target().gradient; g.space = g.space === "oklab" ? "srgb" : "oklab"; changed(t("grad.smooth")); render(); };
  const head = el("div", { class: "grad-head" }, el("div", { class: "group-title", text: t("grad.gradient") }), smoothBtn);

  const posOf = (clientX: number) => { const r = bar.getBoundingClientRect(); return Math.min(1, Math.max(0, (clientX - r.left) / r.width)); };
  const shown = (pos: number) => (target().reverse ? 1 - pos : pos);

  // Tap the bar: a new stop with the colour already there.
  bar.addEventListener("click", (e) => {
    if (!edit.open) return; // stops are edited under "Edit colours" only
    const tg = target();
    const x = shown(posOf(e.clientX));
    const [r, g, b, a] = gradientAt(tg.gradient, x);
    tg.gradient.stops.push({ pos: x, color: rgbToHex(r, g, b), alpha: Math.round(a * 100) / 100 });
    tg.preset = undefined;
    selected = tg.gradient.stops.length - 1;
    changed(t("grad.addStop"));
    render();
  });

  function renderStops() {
    const tg = target();
    (bar.firstChild as HTMLElement).style.background = gradientCss(tg.gradient, tg.reverse);
    stopsRow.replaceChildren(...tg.gradient.stops.map((s, i) => {
      const h = el("div", { class: "grad-handle" + (i === selected ? " on" : ""), role: "button", "aria-label": `${t("grad.stop")} ${i + 1}` }, el("i"));
      h.style.left = `${shown(s.pos) * 100}%`;
      (h.firstChild as HTMLElement).style.background = s.color;
      let drag: { id: number; y0: number; moved: boolean } | undefined;
      h.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        drag = { id: e.pointerId, y0: e.clientY, moved: false };
        h.setPointerCapture(e.pointerId);
        // Select in place: re-rendering the handles here would drop this pointer's capture.
        if (selected !== i) { selected = i; for (const [k, hh] of [...stopsRow.children].entries()) hh.classList.toggle("on", k === i); renderStop(); }
      });
      h.addEventListener("pointermove", (e) => {
        if (!drag || e.pointerId !== drag.id) return;
        drag.moved = true;
        const off = e.clientY - drag.y0 > 44 && tg.gradient.stops.length > 2;
        h.classList.toggle("remove", off);
        s.pos = shown(posOf(e.clientX));
        h.style.left = `${shown(s.pos) * 100}%`;
        (bar.firstChild as HTMLElement).style.background = gradientCss(tg.gradient, tg.reverse);
        tg.preset = undefined;
        changed();
      });
      const end = (e: PointerEvent) => {
        if (!drag || e.pointerId !== drag.id) return;
        const removed = h.classList.contains("remove");
        const moved = drag.moved;
        drag = undefined;
        if (removed) { tg.gradient.stops.splice(i, 1); selected = Math.max(0, i - 1); changed(t("grad.removeStop")); render(); }
        else if (moved) { changed(t("grad.moveStop")); renderStop(); }
      };
      h.addEventListener("pointerup", end);
      h.addEventListener("pointercancel", end);
      return h;
    }));
  }

  function renderStop() {
    const tg = target();
    const s = tg.gradient.stops[selected];
    if (!s) { stopBody.replaceChildren(); return; }
    const color = el("input", { type: "color", class: "grad-color", value: s.color.toLowerCase(), "aria-label": t("grad.color") });
    const hex = el("input", { class: "text grad-hex", value: s.color.toUpperCase(), maxlength: "7", spellcheck: "false", "aria-label": "Hex" });
    color.oninput = () => { s.color = color.value.toUpperCase(); hex.value = s.color; tg.preset = undefined; renderStops(); changed(); };
    color.onchange = () => changed(t("grad.color"));
    hex.onchange = () => {
      const v = hex.value.trim().replace(/^#?/, "#").toUpperCase();
      if (/^#[0-9A-F]{6}$/.test(v)) { s.color = v; color.value = v.toLowerCase(); tg.preset = undefined; renderStops(); changed(t("grad.color")); }
      else hex.value = s.color;
    };
    const del = el("button", { class: "btn small icon ghost", title: t("grad.removeStop"), "aria-label": t("grad.removeStop") }, icon("trash", 18));
    del.disabled = tg.gradient.stops.length <= 2;
    del.onclick = () => { tg.gradient.stops.splice(selected, 1); selected = Math.max(0, selected - 1); changed(t("grad.removeStop")); render(); };
    stopBody.replaceChildren(
      el("div", { class: "grad-stoprow" }, el("label", { class: "grad-colorwrap" }, color), hex, el("span", { class: "grad-spacer" }), del),
      slider(t("grad.opacity"), 0, 1, 0.01, () => s.alpha, (v) => { s.alpha = v; tg.preset = undefined; renderStops(); }, (v) => `${Math.round(v * 100)}%`, 1),
      // As shown on the bar (reversed gradients are drawn mirrored).
      slider(t("grad.location"), 0, 1, 0.005, () => shown(s.pos), (v) => { s.pos = shown(v); tg.preset = undefined; renderStops(); }, (v) => `${Math.round(v * 100)}%`, shown(s.pos)),
    );
  }

  function render() {
    const tg = target();
    selected = Math.min(selected, tg.gradient.stops.length - 1);
    reverseBtn.classList.toggle("on", tg.reverse);
    smoothBtn.classList.toggle("on", tg.gradient.space === "oklab");
    for (const b of presets.children) b.classList.toggle("on", (b as HTMLElement).dataset.id === tg.preset);
    renderStops();
    renderStop();
  }

  const edit = el("details", { class: "grad-edit" }, el("summary", { text: t("grad.edit") }),
    el("div", { class: "group-title", text: t("grad.harmony") }), ruleChips, harmonyRow,
    el("div", { class: "muted grad-hint", text: t("grad.harmonyHint") }),
    head, stopsRow, el("div", { class: "muted grad-hint", text: t("grad.hint") }), stopBody);
  root.append(bar,
    el("div", { class: "grad-quick" }, newBtn, photoBtn, el("span", { class: "grad-spacer" }), countChips, reverseBtn),
    el("div", { class: "group-title", text: t("grad.palettes") }), presets, edit);
  renderPresets();
  renderHarmony();
  renderCount();
  render();

  return root;
}
