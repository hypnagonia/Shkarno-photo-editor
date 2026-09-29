/**
 * The tone equalizer's panel (src/tone/toneEq.ts): a graph of the nine zones over the
 * photo's mask histogram — drag a node up or down, or tap the photo to find its zone —
 * with the zones as sliders, the mask settings, and presets.
 */
import { el } from "./dom.ts";
import { t } from "./i18n.ts";
import type { Params } from "../decision/params.ts";
import { autoFitMask, compensate, MAX_GAIN, neutralToneEq, TONE_EQ_DETAIL, TONE_EQ_PRESETS, toneEqCurve, ZONE_EV, ZONES, type MaskHist, type ToneEq, type ToneEqDetail } from "../tone/toneEq.ts";

export interface ToneEqContext {
  params: () => Params | undefined;
  /** Something changed (history label). */
  changed: (label: string) => void;
  /** The mask histograms of the open photo, by detail setting (as it renders now). */
  hist: () => Record<ToneEqDetail, MaskHist> | undefined;
  /** Ask for them again (the panel opened: exposure or local tone may have changed). */
  refreshHist: () => void;
  /** Show the mask (its zones) on the photo instead of the photo. */
  showMask: (on: boolean) => void;
  /** Taps on the photo pick a zone (on) or do what they normally do (off). */
  pickMode: (on: boolean) => void;
}

export function createToneEqPanel(ctx: ToneEqContext) {
  const root = el("div", { class: "teq" });
  let zone = 4; // the selected node
  let section: "zones" | "mask" = "zones";
  let maskShown = false;
  const eqOf = (p: Params): ToneEq => (p.toneEq ??= neutralToneEq());
  const label = () => t("teq.title");
  const edit = () => { ctx.changed(label()); draw(); };

  // ------------------------------------------------------------------ graph
  const cv = el("canvas", { class: "teq-graph" });
  const X0 = -9, X1 = 1; // EV shown (a half zone of margin either side)
  const gx = (ev: number, w: number) => ((ev - X0) / (X1 - X0)) * w;
  const gy = (gain: number, h: number) => h / 2 - (gain / (MAX_GAIN + 0.4)) * (h / 2 - 8);
  function draw() {
    const p = ctx.params();
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(200, cv.clientWidth || 320), H = 170;
    if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const c = cv.getContext("2d")!;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    const css = getComputedStyle(root);
    const fg = css.getPropertyValue("--text").trim() || "#eee", mute = css.getPropertyValue("--muted").trim() || "#888", line = css.getPropertyValue("--line").trim() || "#444";
    // The photo's tones on the zones: the mask's histogram at the current exposure and compensation.
    const eq = p ? eqOf(p) : neutralToneEq();
    const h = ctx.hist()?.[eq.detail];
    if (h && p) {
      const step = (h.hi - h.lo) / h.bins.length;
      const cols = new Float64Array(Math.ceil(W));
      h.bins.forEach((n, i) => {
        const ev = compensate(eq, h.lo + (i + 0.5) * step);
        const x = Math.round(gx(Math.min(X1, Math.max(X0, ev)), W));
        if (x >= 0 && x < cols.length) cols[x] += n;
      });
      // A little smoothing (bins land on pixel columns unevenly).
      const sm = cols.map((_, i) => ((cols[i - 1] ?? 0) + 2 * cols[i] + (cols[i + 1] ?? 0)) / 4);
      const mx = Math.max(1e-9, ...sm);
      c.fillStyle = mute; c.globalAlpha = 0.28;
      c.beginPath(); c.moveTo(0, H);
      sm.forEach((v, x) => c.lineTo(x, H - Math.pow(v / mx, 0.6) * (H - 10)));
      c.lineTo(W, H); c.fill(); c.globalAlpha = 1;
    }
    // Zones and the 0 line.
    c.strokeStyle = line; c.lineWidth = 1;
    for (const ev of ZONE_EV) { const x = Math.round(gx(ev, W)) + 0.5; c.beginPath(); c.moveTo(x, 4); c.lineTo(x, H - 4); c.stroke(); }
    c.beginPath(); c.moveTo(0, Math.round(gy(0, H)) + 0.5); c.lineTo(W, Math.round(gy(0, H)) + 0.5); c.stroke();
    // The curve.
    const f = toneEqCurve(eq);
    c.strokeStyle = fg; c.lineWidth = 2; c.globalAlpha = eq.enabled ? 1 : 0.35;
    c.beginPath();
    for (let x = 0; x <= W; x += 2) { const y = gy(f(X0 + ((X1 - X0) * x) / W), H); if (x) c.lineTo(x, y); else c.moveTo(x, y); }
    c.stroke();
    // Nodes (the selected one filled).
    ZONE_EV.forEach((ev, i) => {
      const x = gx(ev, W), y = gy(eq.gains[i] ?? 0, H);
      c.beginPath(); c.arc(x, y, i === zone ? 7 : 5, 0, Math.PI * 2);
      c.fillStyle = i === zone ? fg : css.getPropertyValue("--bg").trim() || "#000"; c.fill();
      c.strokeStyle = fg; c.lineWidth = 2; c.stroke();
    });
    c.globalAlpha = 1;
    c.fillStyle = mute; c.font = "10px system-ui, sans-serif"; c.textAlign = "center";
    for (const ev of [-8, -6, -4, -2, 0]) c.fillText(`${ev}`, gx(ev, W), H - 2);
    zoneRow.replaceChildren(el("span", { text: t("teq.zone", { ev: String(ZONE_EV[zone]) }) }), zoneOut);
    zoneIn.value = String(eq.gains[zone] ?? 0);
    zoneOut.textContent = fmtEv(eq.gains[zone] ?? 0);
  }
  // Drag a node vertically; a tap selects the nearest; a double tap sets it to 0.
  let drag: { i: number; y0: number; g0: number } | undefined;
  let lastTap = { t: 0, i: -1 };
  cv.addEventListener("pointerdown", (e) => {
    const p = ctx.params(); if (!p) return;
    const r = cv.getBoundingClientRect();
    const ev = X0 + ((e.clientX - r.left) / r.width) * (X1 - X0);
    const i = Math.min(ZONES - 1, Math.max(0, Math.round(ev + 8)));
    zone = i;
    const now = performance.now();
    if (now - lastTap.t < 320 && lastTap.i === i) { eqOf(p).gains[i] = 0; lastTap.t = 0; edit(); renderSliders(); return; }
    lastTap = { t: now, i };
    drag = { i, y0: e.clientY, g0: eqOf(p).gains[i] ?? 0 };
    cv.setPointerCapture(e.pointerId);
    draw(); renderSliders();
  });
  cv.addEventListener("pointermove", (e) => {
    const p = ctx.params(); if (!drag || !p) return;
    const perEv = (cv.getBoundingClientRect().height / 2 - 8) / (MAX_GAIN + 0.4);
    const g = Math.round(Math.min(MAX_GAIN, Math.max(-MAX_GAIN, drag.g0 - (e.clientY - drag.y0) / perEv)) * 100) / 100;
    const eq = eqOf(p);
    if (g === eq.gains[drag.i]) return;
    eq.gains[drag.i] = g; eq.enabled = true;
    edit(); syncSliders();
  });
  const endDrag = () => { drag = undefined; };
  cv.addEventListener("pointerup", endDrag); cv.addEventListener("pointercancel", endDrag);

  // The selected zone, finely.
  const fmtEv = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(2)} EV`;
  const zoneOut = el("output");
  const zoneRow = el("div", { class: "teq-zone" });
  const zoneIn = el("input", { type: "range", min: String(-MAX_GAIN), max: String(MAX_GAIN), step: "0.01" });
  zoneIn.oninput = () => { const p = ctx.params(); if (!p) return; const eq = eqOf(p); eq.gains[zone] = parseFloat(zoneIn.value); eq.enabled = true; edit(); syncSliders(); };

  // ------------------------------------------------------------------ controls
  const on = el("input", { type: "checkbox" });
  on.onchange = () => { const p = ctx.params(); if (!p) return; eqOf(p).enabled = on.checked; edit(); };
  const reset = el("button", { class: "btn small ghost", text: t("teq.reset") });
  reset.onclick = () => { const p = ctx.params(); if (!p) return; const d = p.toneEq?.detail; p.toneEq = { ...neutralToneEq(), detail: d ?? "balanced" }; edit(); renderSliders(); };
  const presets = el("div", { class: "chips" }, ...TONE_EQ_PRESETS.map((pr) => {
    const b = el("button", { class: "chip", text: t(`teq.preset.${pr.id}`) });
    b.onclick = () => {
      const p = ctx.params(); if (!p) return;
      const eq = eqOf(p);
      eq.gains = [...pr.gains]; eq.enabled = true;
      // A preset assumes the photo's tones spread over the zones: fit the mask the first time.
      if (eq.maskExposure === 0 && eq.maskContrast === 1) fitMask(false);
      edit(); renderSliders();
    };
    return b;
  }));

  const sectionChips = el("div", { class: "seg teq-seg" });
  const body = el("div", { class: "teq-body" });
  /** The visible sliders' refreshers (a drag on the graph moves them too). */
  const sliders: Array<() => void> = [];
  function slider(text: string, min: number, max: number, step: number, get: () => number, set: (v: number) => void, fmt: (v: number) => string, def: number) {
    const input = el("input", { type: "range", min: String(min), max: String(max), step: String(step) });
    const out = el("output");
    const show = () => { const v = get(); input.value = String(v); out.textContent = fmt(v); };
    input.oninput = () => { const p = ctx.params(); if (!p) return; set(parseFloat(input.value)); eqOf(p).enabled = true; show(); edit(); };
    const lab = el("label", { text });
    lab.addEventListener("dblclick", () => { set(def); show(); edit(); });
    show();
    sliders.push(show);
    return el("div", { class: "row" }, lab, input, out);
  }
  const syncSliders = () => { for (const show of sliders) show(); };
  function fitMask(record = true) {
    const p = ctx.params(), h = ctx.hist(); if (!p || !h) return;
    const eq = eqOf(p);
    Object.assign(eq, autoFitMask(h[eq.detail]));
    if (record) { edit(); renderSliders(); }
  }
  function renderSliders() {
    sliders.length = 0;
    const p = ctx.params();
    if (!p) { body.replaceChildren(); return; }
    const eq = eqOf(p);
    on.checked = eq.enabled;
    sectionChips.replaceChildren(...(["zones", "mask"] as const).map((s) => {
      const b = el("button", { class: s === section ? "on" : "", text: t(`teq.${s}`) });
      b.onclick = () => { section = s; renderSliders(); };
      return b;
    }));
    if (section === "zones") {
      body.replaceChildren(
        ...ZONE_EV.map((ev, i) => slider(`${ev} EV`, -MAX_GAIN, MAX_GAIN, 0.01, () => eqOf(p).gains[i] ?? 0, (v) => { eqOf(p).gains[i] = v; zone = i; }, fmtEv, 0)),
        slider(t("teq.smoothing"), 0, 1, 0.01, () => eqOf(p).smoothing, (v) => { eqOf(p).smoothing = v; }, (v) => `${Math.round(v * 100)}`, 0),
      );
    } else {
      const detail = el("div", { class: "chips" }, ...TONE_EQ_DETAIL.map((d) => {
        const b = el("button", { class: "chip" + (d === eq.detail ? " on" : ""), text: t(`teq.detail.${d}`) });
        b.onclick = () => { eqOf(p).detail = d; edit(); renderSliders(); };
        return b;
      }));
      const fit = el("button", { class: "btn small", text: t("teq.fit") });
      fit.onclick = () => fitMask();
      const show = el("input", { type: "checkbox" });
      show.checked = maskShown;
      show.onchange = () => { maskShown = show.checked; ctx.showMask(maskShown && eqOf(p).enabled); };
      body.replaceChildren(
        el("div", { class: "group-title", text: t("teq.detail") }), detail,
        el("p", { class: "muted", text: t("teq.detailTip") }),
        slider(t("teq.maskExposure"), -6, 6, 0.05, () => eqOf(p).maskExposure, (v) => { eqOf(p).maskExposure = v; }, (v) => `${v > 0 ? "+" : ""}${v.toFixed(2)} EV`, 0),
        slider(t("teq.maskContrast"), 0.5, 4, 0.01, () => eqOf(p).maskContrast, (v) => { eqOf(p).maskContrast = v; }, (v) => `×${v.toFixed(2)}`, 1),
        el("div", { class: "actions" }, fit),
        el("p", { class: "muted", text: t("teq.fitTip") }),
        el("label", { class: "toggle" }, t("teq.showMask"), show),
      );
    }
    draw();
  }

  root.append(
    el("div", { class: "teq-head" }, el("label", { class: "toggle" }, t("teq.on"), on), reset),
    cv,
    el("p", { class: "muted teq-hint", text: t("teq.hint") }),
    el("div", { class: "row teq-zone-row" }, zoneRow, zoneIn),
    el("div", { class: "group-title", text: t("teq.presets") }), presets,
    sectionChips, body,
  );
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => draw()).observe(cv);

  return {
    el: root,
    /** The panel became visible (or the photo / its params changed). */
    render() { renderSliders(); ctx.refreshHist(); ctx.pickMode(true); if (maskShown) ctx.showMask(true); },
    /** Left: taps and the mask view go back to normal. */
    leave() { ctx.pickMode(false); if (maskShown) ctx.showMask(false); },
    /** A tap on the photo landed in this zone (0 = −8 EV … 8 = 0 EV). */
    selectZone(i: number) { zone = Math.min(ZONES - 1, Math.max(0, i)); draw(); },
    redraw() { draw(); },
  };
}
