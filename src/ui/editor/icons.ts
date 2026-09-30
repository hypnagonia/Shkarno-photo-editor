/** Line icons (24 × 24, stroked with currentColor) shared by the header and the layers editor. */
export const ICON = {
  undo: "M9 14 4 9l5-5M4 9h10.5a5.5 5.5 0 0 1 0 11H11",
  redo: "m15 14 5-5-5-5M20 9H9.5a5.5 5.5 0 0 0 0 11H13",
  more: "M5 12h.01M12 12h.01M19 12h.01",
  check: "M4 4h16v16H4zM8 12.5l2.8 2.8L16.5 9",
  full: "M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3",
  plus: "M12 5v14M5 12h14",
  close: "M6 6l12 12M18 6 6 18",
  eye: "M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z",
  eyeOff: "M3 3l18 18M10.6 5.1Q11.3 5 12 5c6.5 0 10 7 10 7a17 17 0 0 1-3.2 4.1M6.6 6.6C3.9 8.3 2 12 2 12s3.5 7 10 7c1.7 0 3.2-.5 4.5-1.2M9.9 9.9a3 3 0 0 0 4.2 4.2",
  copy: "M9 9h11v11H9zM5 15H4V4h11v1",
  trash: "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3",
  reset: "M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5",
  // Layer types
  develop: "M4 7h9m4 0h3M4 17h3m4 0h9M15 5v4M9 15v4",
  flare: "M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM2 12h5M17 12h5M12 3v3M12 18v3M19.5 19.5l-1.5-1.5",
  toneEq: "M5 20v-5M5 11V4M12 20v-9M12 7V4M19 20v-3M19 13V4M3 15h4M10 7h4M17 13h4",
  contrastEq: "M3 12c1.5-6 3-6 4.5 0s3 6 4.5 0 3-6 4.5 0 3 6 4.5 0",
  film: "M4 4h16v16H4zM8 4v16M16 4v16M4 8h4M4 12h4M4 16h4M16 8h4M16 12h4M16 16h4",
  retouch: "M14.5 4.5l5 5-8.5 8.5H6v-5zM12 7l5 5M4 21c1.5-1 2-2.5 2-3",
  blur: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM12 3v2M12 19v2M3 12h2M19 12h2",
  curves: "M4 20C11 20 10 4 20 4M4 4v16h16",
  hueSat: "M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z",
  basic: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4",
  brightContrast: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 3v18M12 8h4M12 12h5M12 16h4",
  exposure: "M4 4h16v16H4zM4 20 20 4M7 8h4M9 6v4M13 16h4",
  gradientMap: "M3 7h18v7H3zM7 14v3M12 14v3M17 14v3M9 7v7M15 7v7",
  gradientFill: "M4 4h16v16H4zM4 9h16M4 13h16M4 16.5h16M4 19h16",
  fog: "M3 8c2-1.5 4-1.5 6 0s4 1.5 6 0 4-1.5 6 0M3 13c2-1.5 4-1.5 6 0s4 1.5 6 0 4-1.5 6 0M5 18c2-1.5 4-1.5 6 0s4 1.5 6 0",
  light: "M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0 0 12 3z",
} as const;
export type IconName = keyof typeof ICON;

export function icon(name: IconName, size = 20): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24"); svg.setAttribute("width", String(size)); svg.setAttribute("height", String(size));
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = `<path d="${ICON[name]}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>`;
  return svg;
}
