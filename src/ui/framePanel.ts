/**
 * The Frame card (src/geometry/frame.ts): crop, straighten, turn and mirror. While it is
 * open the preview shows the whole turned photo and the crop is drawn over it — dragged
 * by its corners and edges, moved from inside; proportions locked by a chip.
 */
import { el } from "./dom.ts";
import { t } from "./i18n.ts";
import type { Params } from "../decision/params.ts";
import { FULL_CROP, cropInside, cropOfAspect, fitCrop, orientedSize, type Frame } from "../geometry/frame.ts";

export interface FrameContext {
  params: () => Params | undefined;
  changed: (label: string) => void;
  /** The photo's size before the frame (px, any scale). */
  srcSize: () => [number, number];
  /** Framing on the photo: the preview shows the whole turned photo while on. */
  framing: (on: boolean) => void;
  /** The on-screen rectangle of the preview, and the stage it sits in. */
  imageRect: () => { left: number; top: number; width: number; height: number };
  stage: HTMLElement;
}

/** Width : height as named; laid the way the photo lies (a portrait photo gets 4:5, a landscape one 5:4), ⇅ turns it. */
const RATIOS = [["free", 0], ["orig", -1], ["1:1", 1], ["4:5", 4 / 5], ["3:2", 3 / 2], ["16:9", 16 / 9]] as const;
type RatioId = (typeof RATIOS)[number][0];

export function createFramePanel(ctx: FrameContext) {
  const root = el("div", { class: "frame-panel" });
  let ratio: RatioId = "free";
  /** Locked proportions turned against the photo's own orientation (⇅). */
  let tall = false;
  const frame = (): Frame => {
    const p = ctx.params()!;
    return (p.frame ??= { quarter: 0, flip: false, angle: 0, crop: [...FULL_CROP] });
  };
  const WH = () => ctx.srcSize();
  /** The locked proportions (px width / height of the crop), or none. */
  const lock = (): number | undefined => {
    const r = RATIOS.find((x) => x[0] === ratio)![1];
    if (r === 0) return undefined;
    const [W, H] = WH(), [ow, oh] = orientedSize(frame(), W, H);
    if (r < 0) return tall ? oh / ow : ow / oh;
    const lying = Math.max(r, 1 / r), portrait = (oh > ow) !== tall;
    return portrait ? 1 / lying : lying;
  };

  // ---- the crop on the photo
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("crop-guide");
  svg.style.display = "none";
  ctx.stage.append(svg);
  function draw() {
    const p = ctx.params();
    if (!p || svg.style.display === "none") return;
    const r = ctx.imageRect(), st = ctx.stage.getBoundingClientRect();
    const [x, y, w, h] = frame().crop;
    const L = r.left - st.left, T = r.top - st.top;
    const X0 = L + x * r.width, Y0 = T + y * r.height, X1 = X0 + w * r.width, Y1 = Y0 + h * r.height;
    const W = st.width, H = st.height;
    const thirds = [1, 2].map((k) => `M${X0 + ((X1 - X0) * k) / 3} ${Y0}V${Y1}M${X0} ${Y0 + ((Y1 - Y0) * k) / 3}H${X1}`).join("");
    const c = 16;
    const corners = [[X0, Y0, 1, 1], [X1, Y0, -1, 1], [X0, Y1, 1, -1], [X1, Y1, -1, -1]].map(([a, b, sx, sy]) => `M${a} ${b + sy * c}V${b}H${a + sx * c}`).join("");
    svg.innerHTML = `<path class="shade" fill-rule="evenodd" d="M0 0H${W}V${H}H0zM${X0} ${Y0}V${Y1}H${X1}V${Y0}z"/>`
      + `<path class="thirds" d="${thirds}"/><rect class="edge" x="${X0}" y="${Y0}" width="${X1 - X0}" height="${Y1 - Y0}"/><path class="corner" d="${corners}"/>`;
  }

  let drag: { mode: string; x0: number; y0: number; c0: [number, number, number, number] } | undefined;
  svg.addEventListener("pointerdown", (e) => {
    e.stopPropagation(); e.preventDefault();
    if (!ctx.params()) return;
    const r = ctx.imageRect();
    const [x, y, w, h] = frame().crop;
    const px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
    const tx = 22 / r.width, ty = 22 / r.height;
    const nl = Math.abs(px - x) < tx, nr = Math.abs(px - (x + w)) < tx, nt = Math.abs(py - y) < ty, nb = Math.abs(py - (y + h)) < ty;
    const inX = px > x - tx && px < x + w + tx, inY = py > y - ty && py < y + h + ty;
    let mode = "";
    if (inX && inY) {
      if (nl) mode += "l"; else if (nr) mode += "r";
      if (nt) mode += "t"; else if (nb) mode += "b";
      // Locked proportions: corners only (an edge would break them).
      if (lock() && mode.length === 1) mode = "";
      if (!mode && px > x && px < x + w && py > y && py < y + h) mode = "move";
    }
    if (!mode) return;
    svg.setPointerCapture(e.pointerId);
    drag = { mode, x0: px, y0: py, c0: [x, y, w, h] };
  });
  svg.addEventListener("pointermove", (e) => {
    if (!drag) return;
    e.stopPropagation();
    const r = ctx.imageRect();
    const dx = (e.clientX - r.left) / r.width - drag.x0, dy = (e.clientY - r.top) / r.height - drag.y0;
    let [x, y, w, h] = drag.c0;
    const m = drag.mode, min = 0.05;
    if (m === "move") {
      x = Math.min(1 - w, Math.max(0, x + dx)); y = Math.min(1 - h, Math.max(0, y + dy));
    } else {
      let x1 = x + w, y1 = y + h;
      if (m.includes("l")) x = Math.min(x1 - min, Math.max(0, x + dx));
      if (m.includes("r")) x1 = Math.max(x + min, Math.min(1, x1 + dx));
      if (m.includes("t")) y = Math.min(y1 - min, Math.max(0, y + dy));
      if (m.includes("b")) y1 = Math.max(y + min, Math.min(1, y1 + dy));
      w = x1 - x; h = y1 - y;
      const a = lock();
      if (a) {
        // Keep the proportions: the height follows the width (in px), anchored at the far corner.
        const [W, H] = WH(), [ow, oh] = orientedSize(frame(), W, H);
        const nh = (w * ow) / a / oh;
        if (m.includes("t")) y = y1 - nh;
        h = nh;
        if (y < 0 || y + h > 1) return;
      }
    }
    const f = frame(), [W, H] = WH();
    const c: [number, number, number, number] = [x, y, w, h];
    if (!cropInside(f, W, H, c)) return; // (never past the straightened photo's edges)
    f.crop = c;
    draw();
  });
  const end = (e: PointerEvent) => {
    if (!drag) return;
    e.stopPropagation();
    drag = undefined;
    ctx.changed(t("frame.crop"));
  };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);

  // ---- the card
  const chips = el("div", { class: "chips" });
  const swap = el("button", { class: "chip", title: t("frame.swap"), "aria-label": t("frame.swap"), text: "⇅" });
  swap.onclick = () => { tall = !tall; applyRatio(); };
  function applyRatio() {
    const a = lock(), p = ctx.params();
    if (p && a) { const [W, H] = WH(); frame().crop = cropOfAspect(frame(), W, H, a); ctx.changed(t("frame.crop")); }
    render();
  }
  const angle = el("input", { type: "range", min: "-45", max: "45", step: "0.1" });
  const angleOut = el("output");
  let crop0: [number, number, number, number] | undefined;
  angle.addEventListener("pointerdown", () => { crop0 = [...frame().crop]; });
  angle.oninput = () => {
    const f = frame(), [W, H] = WH();
    crop0 ??= [...f.crop];
    f.angle = parseFloat(angle.value);
    // The crop shrinks, about its centre, to stay inside the turned photo.
    f.crop = fitCrop({ ...f, crop: crop0 }, W, H);
    angleOut.textContent = `${f.angle.toFixed(1)}°`;
    ctx.changed(t("frame.straighten"));
    draw();
  };
  angle.onchange = () => { crop0 = undefined; };
  angle.ondblclick = () => { angle.value = "0"; angle.oninput?.(new Event("input")); crop0 = undefined; };
  const btn = (text: string, title: string, fn: () => void) => { const b = el("button", { class: "btn small", title, "aria-label": title, text }); b.onclick = () => { if (ctx.params()) fn(); }; return b; };
  const turn = (cw: boolean) => {
    const f = frame(), [x, y, w, h] = f.crop;
    f.quarter = (f.quarter + (cw ? 1 : 3)) & 3;
    f.crop = cw ? [1 - (y + h), x, h, w] : [y, 1 - (x + w), h, w];
    ctx.changed(t("frame.turn"));
    render();
  };
  const mirror = () => {
    // Left ↔ right as seen: under an odd number of turns the photo's own mirror is up ↔ down,
    // so half a turn more.
    const f = frame(), [x, y, w, h] = f.crop;
    f.flip = !f.flip;
    if (f.quarter & 1) f.quarter = (f.quarter + 2) & 3;
    f.angle = -f.angle;
    f.crop = [1 - x - w, y, w, h];
    ctx.changed(t("frame.flip"));
    render();
  };
  const reset = () => { const p = ctx.params()!; delete p.frame; ratio = "free"; tall = false; ctx.changed(t("frame.reset")); render(); };
  root.append(
    el("p", { class: "muted", text: t("frame.hint") }),
    chips,
    el("div", { class: "row" }, el("label", { text: t("frame.straighten") }), angle, angleOut),
    el("div", { class: "chips" }, btn("↺", t("frame.left"), () => turn(false)), btn("↻", t("frame.right"), () => turn(true)), btn("⇋", t("frame.flip"), mirror), btn(t("frame.reset"), t("frame.reset"), reset)),
  );

  function render() {
    svg.style.display = "";
    ctx.framing(true);
    const f = ctx.params()?.frame;
    angle.value = String(f?.angle ?? 0);
    angleOut.textContent = `${(f?.angle ?? 0).toFixed(1)}°`;
    chips.replaceChildren(...RATIOS.map(([id]) => {
      const b = el("button", { class: "chip" + (id === ratio ? " on" : ""), text: t(`frame.r.${id}`) });
      b.onclick = () => { ratio = id; applyRatio(); };
      return b;
    }), swap);
    swap.hidden = !lock() || ratio === "1:1";
    draw();
  }
  function leave() {
    if (svg.style.display === "none") return;
    svg.style.display = "none";
    drag = undefined;
    ctx.framing(false);
  }
  return { el: root, render, leave, draw };
}
