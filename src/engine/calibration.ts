/** Automatic development calibrated to the camera's rendering, on what the preview shows. */
import type { Engine } from "./engine.ts";
import type { Session } from "./session.ts";
import type { Params } from "../decision/params.ts";
import { MEDIAN, REF_QS, shadowMatch } from "../decode/preview.ts";
import { srgbEotf } from "../color/transfer.ts";

/**
 * Calibration to the camera's own rendering (the JPEG inside a ProRAW DNG), one step per
 * final preview after opening: exposure (≤ 2 rounds), contrast, the black point and
 * highlights (a curve the page adds as a layer), then colourfulness. `calib` / `calibHi`:
 * this preview's quantiles (REF_QS / HI_QS), `oursC` its mid-tone chroma, all measured
 * without the film.
 */
export function calibrateToCamera(eng: Engine, s: Session, p: Params, calib: number[], calibHi: number[] | undefined, oursC: { mean: number; p95: number } | undefined) {
  const oursChroma = oursC?.mean;
  const cal = s.calib!;
    // The display model misjudges some scenes (backlight, night): correct on what was rendered.
    const ours = calib[MEDIAN], ref = cal.ref[MEDIAN];
    if (cal.rounds < 2 && Math.abs(ours - ref) > 6 / 255) {
      cal.rounds++;
      const d = Math.log2(Math.max(srgbEotf(ref), 1e-4) / Math.max(srgbEotf(ours), 1e-4));
      const ev = Math.round(Math.min(2.5, Math.max(-3, p.exposure + 0.9 * d)) * 100) / 100;
      const note = `measured on the preview: median ${Math.round(ours * 255)}/255 vs the camera's ${Math.round(ref * 255)}/255 → ${ev > 0 ? "+" : ""}${ev} EV`;
      eng.log(`exposure calibration: ${note}`);
      s.decision.params.exposure = ev;
      // Not over an edit that arrived while this frame rendered.
      if (s.params.exposure === p.exposure) s.params = { ...s.params, exposure: ev };
      eng.post({ type: "exposureCalibrated", exposure: ev, note });
      eng.requestRender(true);
    } else if (!cal.contrast && !cal.black && calibHi && cal.refHi) {
      // Exposure settled, but the camera's shadows much deeper / highlights much brighter
      // (a contrasty backlit or golden-hour scene we flattened): take back the automatic
      // shadow lift and highlight compression first — a curve alone moves tones too little.
      cal.contrast = true;
      const deeper = calib[REF_QS.indexOf(0.05)] - cal.ref[REF_QS.indexOf(0.05)]; // > 0: ours lifted
      const brighter = cal.refHi[2] - calibHi[2]; // > 0: the camera's highlights brighter (98 %)
      const tone = { ...p.tone }, local = { ...p.local };
      const why: string[] = [];
      if (deeper > 10 / 255 && tone.shadows > 0) { const k = Math.max(0, 1 - deeper / (40 / 255)); tone.shadows = Math.round(tone.shadows * k * 100) / 100; local.compression = Math.round(local.compression * (0.5 + 0.5 * k) * 100) / 100; why.push(`shadows ${Math.round(deeper * 255)}/255 lighter than the camera's → shadow lift ${tone.shadows}`); }
      if (brighter > 20 / 255 && tone.highlights < 0) { tone.highlights = Math.round(tone.highlights * Math.max(0, 1 - brighter / (60 / 255)) * 100) / 100; why.push(`highlights ${Math.round(brighter * 255)}/255 dimmer than the camera's → highlight compression ${tone.highlights}`); }
      if (why.length) {
        const note = `contrast matched to the camera's rendering: ${why.join("; ")}`;
        eng.log(note);
        s.decision.params.tone = tone; s.decision.params.local = local;
        // Each value only where not edited while this frame rendered (the page decides the same way).
        const cur = s.params;
        const shadows = cur.tone.shadows === p.tone.shadows ? tone.shadows : cur.tone.shadows;
        const highlights = cur.tone.highlights === p.tone.highlights ? tone.highlights : cur.tone.highlights;
        const compression = cur.local.compression === p.local.compression ? local.compression : cur.local.compression;
        s.params = { ...cur, tone: { ...cur.tone, shadows, highlights }, local: { ...cur.local, compression } };
        const applied = [shadows === tone.shadows && "tone.shadows", highlights === tone.highlights && "tone.highlights", compression === local.compression && "local.compression"].filter((k): k is string => !!k);
        eng.post({ type: "autoAdjusted", changes: { "tone.shadows": tone.shadows, "tone.highlights": tone.highlights, "local.compression": local.compression }, from: { "tone.shadows": p.tone.shadows, "tone.highlights": p.tone.highlights, "local.compression": p.local.compression }, applied, note });
      }
      eng.requestRender(true);
    } else if (cal.black && !cal.color) {
      // Exposure and black point settled: colourfulness, to the camera's (the automatic
      // saturation, never beyond ±0.35; not over a saturation set by hand meanwhile).
      cal.color = true;
      const ref = cal.chroma ?? 0;
      if (oursChroma !== undefined && oursChroma > 0.01 && ref > 0.01) {
        const r = ref / oursChroma;
        if (Math.abs(Math.log(r)) > 0.08) {
          // Paler than the camera: more vibrance (it lifts weak colour and spares strong,
          // so nothing is pushed out of gamut); more colourful: less saturation.
          const sat0 = p.color.saturation, vib0 = p.color.vibrance;
          let sat = sat0, vib = vib0;
          // Only as far as the strongest colours stay within the camera's own: an average
          // pulled down by grey surfaces must not push the vivid ones out of gamut.
          const room = oursC && cal.chroma95 ? cal.chroma95 / Math.max(oursC.p95, 1e-6) : r;
          const rb = Math.min(r, Math.max(1, 1 + (room - 1) * 2));
          // (Vibrance multiplies weak chroma by ≈ 1 + vibrance: the shortfall itself.)
          if (r > 1) vib = Math.round(Math.min(0.6, vib0 + (rb - 1) * 0.9) * 100) / 100;
          else sat = Math.round(Math.max(-0.3, (1 + sat0) * Math.pow(r, 0.8) - 1) * 100) / 100;
          const note = `colour matched to the camera's rendering: mid-tone chroma ${oursChroma.toFixed(3)} vs ${ref.toFixed(3)} → ${r > 1 ? `vibrance ${vib > 0 ? "+" : ""}${Math.round(vib * 100)}` : `saturation ${Math.round(sat * 100)}`}`;
          eng.log(note);
          s.decision.params.color = { ...s.decision.params.color, saturation: sat, vibrance: vib };
          if (s.params.color.saturation === sat0 && s.params.color.vibrance === vib0) { s.params = { ...s.params, color: { ...s.params.color, saturation: sat, vibrance: vib } }; eng.requestRender(true); }
          eng.post({ type: "colorCalibrated", saturation: sat, vibrance: vib, from: [sat0, vib0], note });
        }
      }
    } else {
      // Exposure settled: now the black point, on the same rendering.
      cal.rounds = 2;
      cal.black = true;
      const points = shadowMatch(calib, cal.ref, calibHi, cal.refHi);
      if (points) {
        const note = `tones matched to the camera's rendering: shadows ${calib.slice(0, MEDIAN).map((v) => Math.round(v * 255)).join("/")} → ${cal.ref.slice(0, MEDIAN).map((v) => Math.round(v * 255)).join("/")}, highlights ${calibHi!.map((v) => Math.round(v * 255)).join("/")} → ${(cal.refHi ?? []).map((v) => Math.round(v * 255)).join("/")}`;
        eng.log(note);
        eng.post({ type: "blackPointMatched", points, note });
        // (The colour step reads the next final preview — with this curve, which the
        // page adds and sends back: measured before it, the colour would be off.)
      } else eng.requestRender(true); // no curve: the colour step needs a preview of its own
    }
}
