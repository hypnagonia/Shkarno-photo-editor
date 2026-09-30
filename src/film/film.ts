/**
 * Film: three characters of analog rendering, each one strength. What they are made of
 * is physical, so the numbers mean the same on every photo:
 *
 *   grain      RMS granularity (σ_D × 1000 through a 48 µm aperture at density 1, as
 *              film datasheets give it) and the dye-cloud size, both on the film: a
 *              larger negative (format) spreads the same emulsion over more of the
 *              image, so its grain is finer; a higher-resolution photo shows it in more
 *              pixels, each one noisier (the same grain, looked at more closely)
 *   halation   light through the emulsion, back from the film base into the red layer:
 *              a red-orange rim around bright lights, a fixed distance on the film
 *   bloom      a diffusion filter's glow on the highlights (optical: relative to the frame)
 *   softness   the film's MTF: a blur of a fixed size on the film
 *   shoulder   print paper's gentler highlights
 *
 * Pure numbers here; render_film.wgsl draws them.
 */
import type { Film, FilmCharacter, Params } from "../decision/params.ts";

interface Character {
  /** Diffuse RMS granularity (σ_D × 1000 at a 48 µm aperture, density 1). */
  rms: number;
  /** Dye cloud diameter on the film, µm. */
  cloud: number;
  /** Correlation of the three layers' grain (1 = monochrome grain). */
  mono: number;
  /** Clumping (micro-structure) 0 … 1. */
  clump: number;
  /** Halation amount and its radius on the film (mm). */
  halation: number; halationMm: number;
  /** Bloom amount and radius (share of the long side). */
  bloom: number; bloomR: number;
  /** Softness: σ of the film's blur on the film, µm. */
  softUm: number;
  /** Highlight shoulder 0 … 1. */
  shoulder: number;
}

export const FILM_CHARACTERS: Record<Exclude<FilmCharacter, "off">, Character> = {
  // Nearly invisible grain, a little diffusion: a fine 100-speed stock through a light filter.
  clean: { rms: 3, cloud: 7, mono: 0.85, clump: 0.25, halation: 0, halationMm: 0.2, bloom: 0.08, bloomR: 0.02, softUm: 2.5, shoulder: 0.15 },
  // Physical grain, soft highlights, a delicate halation: a 400-speed colour negative, printed.
  negative: { rms: 7, cloud: 10, mono: 0.8, clump: 0.45, halation: 0.12, halationMm: 0.22, bloom: 0.04, bloomR: 0.02, softUm: 3.5, shoulder: 0.4 },
  // Pronounced grain, optical glow, softer micro-texture: fast motion-picture stock, no anti-halation layer.
  cinema: { rms: 11, cloud: 13, mono: 0.7, clump: 0.65, halation: 0.35, halationMm: 0.3, bloom: 0.12, bloomR: 0.03, softUm: 5, shoulder: 0.25 },
};

export const FILM_FORMATS = [{ mm: 36, id: "35" }, { mm: 70, id: "67" }, { mm: 125, id: "45" }] as const;

/** What a new photo starts with (defaultParams). */
export const defaultFilm = (): Film => ({ character: "cinema", strength: 1, format: 36 });

/** The film to render, if any (grain from edits saved before the Film card renders on its own). */
export function filmOf(p: Params): Film | undefined {
  if (p.film) return p.film.character !== "off" && p.film.strength > 0 ? p.film : undefined;
  return undefined;
}

export interface FilmUniforms {
  /** σ_D per full-image pixel (0 = no grain), dye cloud in full-image px, clumping, layer correlation. */
  grain: [number, number, number, number];
  /** Softness σ in full-image px, shoulder. */
  softPx: number; shoulder: number;
  /** Halation / bloom amounts, their σ as a share of the long side. */
  halation: number; bloom: number; halationR: number; bloomR: number;
}

/** The numbers for an image whose long side is `longPx` full-image pixels. */
export function filmUniforms(f: Film, longPx: number): FilmUniforms | undefined {
  if (f.character === "off" || f.strength <= 0) return undefined;
  const c = FILM_CHARACTERS[f.character];
  const s = Math.min(f.strength, 1.5);
  const pixelUm = (f.format * 1000) / Math.max(longPx, 1);
  // A grain field correlated over the cloud size has, per pixel, the 48 µm RMS scaled by
  // 48 / (the larger of cloud and pixel): smaller than a pixel, clouds average out in it.
  const sigma = (c.rms / 1000) * (48 / Math.max(c.cloud, pixelUm)) * s;
  return {
    grain: [sigma, c.cloud / pixelUm, c.clump, c.mono],
    softPx: Math.min((c.softUm / pixelUm) * Math.min(s, 1), 1.5),
    shoulder: c.shoulder * Math.min(s, 1),
    halation: c.halation * s, bloom: c.bloom * s,
    halationR: c.halationMm / f.format, bloomR: c.bloomR,
  };
}
