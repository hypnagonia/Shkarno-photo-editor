/**
 * The Film card (src/film/film.ts): a character of analog rendering — off, clean
 * analog, negative film, cinema texture — one strength, and the negative's format.
 * Made for a thumb: three rows, no expert controls.
 */
import { el } from "./dom.ts";
import { t } from "./i18n.ts";
import type { FilmCharacter, Params } from "../decision/params.ts";
import { FILM_FORMATS, defaultFilm } from "../film/film.ts";

export interface FilmContext {
  params: () => Params | undefined;
  changed: (label: string) => void;
}

const CHARACTERS: FilmCharacter[] = ["off", "clean", "negative", "cinema"];

export function createFilmPanel(ctx: FilmContext) {
  const root = el("div", { class: "film" });
  const film = (p: Params) => (p.film ??= defaultFilm());
  const edit = () => { ctx.changed(t("film.title")); render(); };

  const chars = el("div", { class: "chips film-chars" });
  const note = el("p", { class: "muted film-note" });
  const strength = el("input", { type: "range", min: "0", max: "1.5", step: "0.01" });
  const strengthOut = el("output");
  strength.oninput = () => {
    const p = ctx.params(); if (!p) return;
    const f = film(p);
    f.strength = parseFloat(strength.value);
    if (f.character === "off") f.character = "negative";
    strengthOut.textContent = `${Math.round(f.strength * 100)}%`;
    edit();
  };
  // Double-tap: back to the film's own strength.
  const strengthLabel = el("label", { text: t("film.strength") });
  strengthLabel.addEventListener("dblclick", () => { const p = ctx.params(); if (!p) return; film(p).strength = 1; edit(); });
  const formats = el("div", { class: "chips film-formats" });

  function render() {
    const p = ctx.params();
    const f = p?.film ?? defaultFilm();
    chars.replaceChildren(...CHARACTERS.map((c) => {
      const b = el("button", { class: "chip" + (f.character === c ? " on" : ""), text: t(`film.${c}`) });
      b.onclick = () => { const q = ctx.params(); if (!q) return; film(q).character = c; edit(); };
      return b;
    }));
    note.textContent = t(`film.${f.character}Note`);
    strength.value = String(f.strength);
    strength.disabled = !p;
    strengthOut.textContent = `${Math.round(f.strength * 100)}%`;
    formats.replaceChildren(...FILM_FORMATS.map((fm) => {
      const b = el("button", { class: "chip" + (f.format === fm.mm ? " on" : ""), text: t(`film.format.${fm.id}`) });
      b.onclick = () => { const q = ctx.params(); if (!q) return; film(q).format = fm.mm; edit(); };
      return b;
    }));
    const off = f.character === "off";
    root.classList.toggle("off", off);
  }

  root.append(
    chars,
    note,
    el("div", { class: "row" }, strengthLabel, strength, strengthOut),
    el("div", { class: "group-title", text: t("film.format") }),
    formats,
    el("p", { class: "muted", text: t("film.hint") }),
  );
  render();
  return { el: root, render, leave: () => {} };
}
