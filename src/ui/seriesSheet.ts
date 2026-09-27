/**
 * Opening several photos at once: a sheet offering to merge them as one series
 * (src/burst) — which shot is the reference (default: the sharpest), how many
 * frames are used on this device — or to open just the first.
 */
import { el } from "./dom.ts";
import { t } from "./i18n.ts";
import { isPhone } from "../device.ts";

/** Most frames merged: phone memory and time (each frame is developed twice). */
export const maxFrames = () => (isPhone() ? 12 : 24);

/** `n` of the files, evenly spread over the series (the first and last kept). */
export function spread<T>(files: T[], n: number): T[] {
  if (files.length <= n) return files;
  return Array.from({ length: n }, (_, i) => files[Math.round((i * (files.length - 1)) / (n - 1))]);
}

export function openSeriesSheet(all: File[], act: { merge: (files: File[], ref?: number) => void; single: (f: File) => void }) {
  const files = spread([...all].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true })), maxFrames());
  let ref: number | undefined;
  const root = el("div", { class: "more series-sheet" });
  const close = () => root.remove();
  root.onclick = (e) => { if (e.target === root) close(); };
  const refs = el("div", { class: "chips" });
  const drawRefs = () => {
    refs.replaceChildren(...[undefined, ...files.map((_, i) => i)].map((i) => {
      const b = el("button", { class: "chip" + (i === ref ? " on" : ""), text: i === undefined ? t("series.refAuto") : files[i].name.replace(/\.[^.]+$/, "") });
      b.onclick = () => { ref = i; drawRefs(); };
      return b;
    }));
  };
  drawRefs();
  const merge = el("button", { class: "btn primary", text: t("series.merge", { n: String(files.length) }) });
  merge.onclick = () => { close(); act.merge(files, ref); };
  const first = el("button", { class: "btn ghost", text: t("series.firstOnly") });
  first.onclick = () => { close(); act.single(files[0]); };
  const pane = el("div", { class: "pane series-pane" },
    el("div", { class: "group-title", text: t("series.title", { n: String(all.length) }) }),
    el("p", { class: "muted", text: t("series.body") }),
    el("div", { class: "chips" }, el("button", { class: "chip on", text: t("series.mode.clean") })),
    el("div", { class: "group-title", text: t("series.ref") }),
    refs,
    ...(all.length > files.length ? [el("p", { class: "muted", text: t("series.limit", { n: String(files.length), total: String(all.length) }) })] : []),
    el("div", { class: "series-actions" }, first, merge),
  );
  root.append(pane);
  document.body.append(root);
}
