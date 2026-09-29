/**
 * The contrast equalizer's panel (src/tone/contrastEq.ts): seven detail bands from
 * coarse (left) to fine (right), a lightness curve and a colour curve over them —
 * drag a point up for more contrast at that size, down for less — edge awareness,
 * presets.
 */
import { el } from "./dom.ts";
import { t } from "./i18n.ts";
import type { Params } from "../decision/params.ts";
import { BAND_PX, BANDS, CONTRAST_EQ_PRESETS, MAX_BAND_GAIN, neutralContrastEq, type ContrastEq } from "../tone/contrastEq.ts";

export interface ContrastEqContext {
  params: () => Params | undefined;
  changed: (label: string) => void;
}

type Curve = "luma" | "chroma";

export function createContrastEqPanel(ctx: ContrastEqContext) {
  const root = el("div", { class: "teq ceq" });
  let curve: Curve = "luma";
  let band = 3;
  const ceqOf = (p: Params): ContrastEq => (p.contrastEq ??= neutralContrastEq());
  const edit = () => { ctx.changed(t("ceq.title")); draw(); };
  // Coarse on the left, fine on the right (column k shows band BANDS−1−k).
  const bandAt = (col: number) => BANDS - 1 - col;
  const pct = (v: number) => `${v > 0 ? "+" : ""}${Math.round(v * 100)}%`;
  const sizeLabel = (b: number) => `${BAND_PX[b]} px`;

  // ------------------------------------------------------------------ graph
  const cv = el("canvas", { class: "teq-graph" });
  const gx = (col: number, w: number) => ((col + 0.5) / BANDS) * w;
  const gy = (g: number, h: number) => h / 2 - (g / (MAX_BAND_GAIN + 0.15)) * (h / 2 - 12);
  function draw() {
    const p = ctx.params();
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(200, cv.clientWidth || 320), H = 170;
    if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const c = cv.getContext("2d")!;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    const css = getComputedStyle(root);
    const fg = css.getPropertyValue("--text").trim() || "#eee", mute = css.getPropertyValue("--muted").trim() || "#888", line = css.getPropertyValue("--line").trim() || "#444", bg = css.getPropertyValue("--bg").trim() || "#000";
    const eq = p ? ceqOf(p) : neutralContrastEq();
    c.strokeStyle = line; c.lineWidth = 1;
    for (let k = 0; k < BANDS; k++) { const x = Math.round(gx(k, W)) + 0.5; c.beginPath(); c.moveTo(x, 4); c.lineTo(x, H - 14); c.stroke(); }
    c.beginPath(); c.moveTo(0, Math.round(gy(0, H)) + 0.5); c.lineTo(W, Math.round(gy(0, H)) + 0.5); c.stroke();
    // Both curves; the one being edited on top, solid.
    for (const which of (curve === "luma" ? ["chroma", "luma"] : ["luma", "chroma"]) as Curve[]) {
      const vals = eq[which];
      const on = which === curve;
      c.strokeStyle = on ? fg : mute; c.lineWidth = on ? 2 : 1.5; c.setLineDash(which === "chroma" ? [5, 4] : []);
      c.globalAlpha = eq.enabled ? 1 : 0.35;
      c.beginPath();
      for (let k = 0; k < BANDS; k++) { const x = gx(k, W), y = gy(vals[bandAt(k)] ?? 0, H); if (k) c.lineTo(x, y); else c.moveTo(x, y); }
      c.stroke(); c.setLineDash([]);
      if (on) for (let k = 0; k < BANDS; k++) {
        const b = bandAt(k);
        c.beginPath(); c.arc(gx(k, W), gy(vals[b] ?? 0, H), b === band ? 7 : 5, 0, Math.PI * 2);
        c.fillStyle = b === band ? fg : bg; c.fill(); c.strokeStyle = fg; c.lineWidth = 2; c.stroke();
      }
    }
    c.globalAlpha = 1;
    c.fillStyle = mute; c.font = "10px system-ui, sans-serif"; c.textAlign = "center";
    c.fillText(t("ceq.coarse"), gx(0, W), H - 2); c.fillText(t("ceq.fine"), gx(BANDS - 1, W), H - 2);
    for (let k = 1; k < BANDS - 1; k++) c.fillText(sizeLabel(bandAt(k)), gx(k, W), H - 2);
    bandLabel.textContent = `${t(`ceq.${curve}`)} · ${sizeLabel(band)}`;
    bandIn.value = String(eq[curve][band] ?? 0);
    bandOut.textContent = pct(eq[curve][band] ?? 0);
  }
  let drag: { b: number; y0: number; g0: number } | undefined;
  let lastTap = { t: 0, b: -1 };
  cv.addEventListener("pointerdown", (e) => {
    const p = ctx.params(); if (!p) return;
    const r = cv.getBoundingClientRect();
    const col = Math.min(BANDS - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * BANDS)));
    const b = bandAt(col);
    band = b;
    const eq = ceqOf(p);
    const now = performance.now();
    if (now - lastTap.t < 320 && lastTap.b === b) { eq[curve][b] = 0; lastTap.t = 0; edit(); syncSliders(); return; }
    lastTap = { t: now, b };
    drag = { b, y0: e.clientY, g0: eq[curve][b] ?? 0 };
    cv.setPointerCapture(e.pointerId);
    draw();
  });
  cv.addEventListener("pointermove", (e) => {
    const p = ctx.params(); if (!drag || !p) return;
    const perUnit = (cv.getBoundingClientRect().height / 2 - 12) / (MAX_BAND_GAIN + 0.15);
    const g = Math.round(Math.min(MAX_BAND_GAIN, Math.max(-MAX_BAND_GAIN, drag.g0 - (e.clientY - drag.y0) / perUnit)) * 100) / 100;
    const eq = ceqOf(p);
    if (g === eq[curve][drag.b]) return;
    eq[curve][drag.b] = g; eq.enabled = true;
    edit(); syncSliders();
  });
  const endDrag = () => { drag = undefined; };
  cv.addEventListener("pointerup", endDrag); cv.addEventListener("pointercancel", endDrag);

  const bandLabel = el("span");
  const bandOut = el("output");
  const bandIn = el("input", { type: "range", min: String(-MAX_BAND_GAIN), max: String(MAX_BAND_GAIN), step: "0.01" });
  bandIn.oninput = () => { const p = ctx.params(); if (!p) return; const eq = ceqOf(p); eq[curve][band] = parseFloat(bandIn.value); eq.enabled = true; edit(); syncSliders(); };

  // ------------------------------------------------------------------ controls
  const on = el("input", { type: "checkbox" });
  on.onchange = () => { const p = ctx.params(); if (!p) return; ceqOf(p).enabled = on.checked; edit(); };
  const reset = el("button", { class: "btn small ghost", text: t("teq.reset") });
  reset.onclick = () => { const p = ctx.params(); if (!p) return; p.contrastEq = neutralContrastEq(); edit(); renderSliders(); };
  const presets = el("div", { class: "chips" }, ...CONTRAST_EQ_PRESETS.map((pr) => {
    const b = el("button", { class: "chip", text: t(`ceq.preset.${pr.id}`) });
    b.onclick = () => { const p = ctx.params(); if (!p) return; const eq = ceqOf(p); eq.luma = [...pr.luma]; eq.chroma = [...pr.chroma]; eq.enabled = true; edit(); renderSliders(); };
    return b;
  }));
  const curveChips = el("div", { class: "seg teq-seg" });
  const body = el("div", { class: "teq-body" });
  const sliders: Array<() => void> = [];
  const syncSliders = () => { for (const show of sliders) show(); };
  function slider(text: string, min: number, max: number, step: number, get: () => number, set: (v: number) => void, fmt: (v: number) => string, def: number) {
    const input = el("input", { type: "range", min: String(min), max: String(max), step: String(step) });
    const out = el("output");
    const show = () => { const v = get(); input.value = String(v); out.textContent = fmt(v); };
    input.oninput = () => { const p = ctx.params(); if (!p) return; set(parseFloat(input.value)); ceqOf(p).enabled = true; show(); edit(); };
    const lab = el("label", { text });
    lab.addEventListener("dblclick", () => { set(def); show(); edit(); });
    show();
    sliders.push(show);
    return el("div", { class: "row" }, lab, input, out);
  }
  function renderSliders() {
    sliders.length = 0;
    const p = ctx.params();
    if (!p) { body.replaceChildren(); return; }
    const eq = ceqOf(p);
    on.checked = eq.enabled;
    curveChips.replaceChildren(...(["luma", "chroma"] as const).map((k) => {
      const b = el("button", { class: k === curve ? "on" : "", text: t(`ceq.${k}`) });
      b.onclick = () => { curve = k; renderSliders(); };
      return b;
    }));
    body.replaceChildren(
      // Coarse first, as on the graph.
      ...Array.from({ length: BANDS }, (_, k) => bandAt(k)).map((b) => slider(sizeLabel(b), -MAX_BAND_GAIN, MAX_BAND_GAIN, 0.01, () => ceqOf(p)[curve][b] ?? 0, (v) => { ceqOf(p)[curve][b] = v; band = b; }, pct, 0)),
      el("div", { class: "group-title", text: t("ceq.edges") }),
      slider(t("ceq.edgesLabel"), 0, 1, 0.01, () => ceqOf(p).edges, (v) => { ceqOf(p).edges = v; }, (v) => `${Math.round(v * 100)}`, 0.6),
      el("p", { class: "muted", text: t("ceq.edgesTip") }),
    );
    draw();
  }

  root.append(
    el("div", { class: "teq-head" }, el("label", { class: "toggle" }, t("teq.on"), on), reset),
    cv,
    el("p", { class: "muted teq-hint", text: t("ceq.hint") }),
    el("div", { class: "row teq-zone-row" }, el("div", { class: "teq-zone" }, bandLabel, bandOut), bandIn),
    el("div", { class: "group-title", text: t("teq.presets") }), presets,
    curveChips, body,
  );
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => draw()).observe(cv);
  return { el: root, render() { renderSliders(); }, leave() {}, redraw() { draw(); } };
}
