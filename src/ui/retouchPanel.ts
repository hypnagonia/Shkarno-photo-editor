/**
 * The Brush card (magic brush, src/retouch): while it is open one finger paints over what
 * should disappear; releasing fills it in. Brush size, undo the last stroke, clear all.
 */
import { el } from "./dom.ts";
import { t } from "./i18n.ts";
import type { Params } from "../decision/params.ts";

export interface RetouchContext {
  params: () => Params | undefined;
  changed: (label: string) => void;
  /** Painting on the photo: on while the card is open, with this brush radius (share of the long side). */
  brush: { on: boolean; radius: number };
}

export function createRetouchPanel(ctx: RetouchContext) {
  const root = el("div", { class: "retouch" });
  const size = el("input", { type: "range", min: "0.005", max: "0.08", step: "0.001" });
  const sizeOut = el("output");
  size.oninput = () => { ctx.brush.radius = parseFloat(size.value); sizeOut.textContent = `${Math.round(ctx.brush.radius * 200 * 10) / 10}%`; };
  const count = el("p", { class: "muted" });
  const undo = el("button", { class: "btn small", text: t("brush.undo") });
  const clear = el("button", { class: "btn small ghost", text: t("brush.clear") });
  undo.onclick = () => { const p = ctx.params(); if (!p?.retouch?.length) return; p.retouch = p.retouch.slice(0, -1); ctx.changed(t("brush.undo")); render(); };
  clear.onclick = () => { const p = ctx.params(); if (!p?.retouch?.length) return; p.retouch = []; ctx.changed(t("brush.clear")); render(); };

  function render() {
    ctx.brush.on = true;
    size.value = String(ctx.brush.radius);
    sizeOut.textContent = `${Math.round(ctx.brush.radius * 200 * 10) / 10}%`;
    const n = ctx.params()?.retouch?.length ?? 0;
    count.textContent = n ? t("brush.count", { n }) : t("brush.none");
    undo.disabled = clear.disabled = !n;
  }

  root.append(
    el("p", { class: "muted", text: t("brush.hint") }),
    el("div", { class: "row" }, el("label", { text: t("brush.size") }), size, sizeOut),
    el("div", { class: "chips" }, undo, clear),
    count,
  );
  return { el: root, render, leave: () => { ctx.brush.on = false; } };
}
