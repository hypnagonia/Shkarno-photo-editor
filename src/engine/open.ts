/** Opening a photo: decode, develop, scene analysis, decisions — up to the first preview. */
import type { Engine } from "./engine.ts";
import { halvesToFloats } from "../gpu/half.ts";
import { decodeFile } from "../decode/decode.ts";
import type { DecodedImage } from "../decode/types.ts";
import { develop, type WorkingImage } from "../raw/develop.ts";
import { analyseScene, analyseSceneIsolated, neutralScene, applyAppleMattes, type AnalysisLevel, GROUPS, type Group } from "../neural/scene.ts";
import { decideUpscale, measureQuality, type ImageQualityReport, type UpscaleMode } from "../analysis/quality.ts";
import { downsample, refine, releaseRefined } from "../refine/refine.ts";
import { blurReport, lumPercentiles, measureBlocks, measureRegions, noiseProfile } from "../analysis/analysis.ts";
import type { AnalysisReport } from "../analysis/types.ts";
import { decide } from "../decision/engine.ts";
import { autoFocus } from "../decision/focus.ts";
import { buildAutoLayers } from "../layers/auto.ts";
import { embeddedPreviewStats, MEDIAN, REF_QS } from "../decode/preview.ts";
import { displayQuantiles, applyAutoCurves, autoCurves } from "../decision/autoCurves.ts";
import { allMask, makeLayer } from "../layers/model.ts";
import { depthZones } from "../decision/zones.ts";
import { cellCoverage } from "../analysis/previewHist.ts";
import { measureSkin, naturalSkin } from "../decision/skinTone.ts";
import { neutralToTempTint } from "../color/wb.ts";
import { Profiler } from "./profiler.ts";
import type { Summary } from "./protocol.ts";
import { srgbEotf, srgbOetf } from "../color/transfer.ts";
import { phoneForced } from "../device.ts";
import { sceneEV, type Session, DEFAULT_DOF_STRENGTH, UPSCALE_MAX_MP, ANALYSIS_LONG, ANALYSIS_LONG_LIGHT, isMobile, protectGroup, exposureGain, vanishingPoint } from "./session.ts";

export async function openInner(eng: Engine, file: File, resolution: "auto" | "full" | "half", autoExposure: boolean, autoDof: boolean, upscaleMode: UpscaleMode, safeAnalysis: boolean, level: AnalysisLevel | undefined,
  track: (free: () => void) => void, commit: () => void) {
  const gen = ++eng.generation;
  eng.closeSession();
  eng.gpu.flushStaging(); // the previous photo's readback sizes
  const P = new Profiler(eng.gpu);
  eng.profiler = P;
  const gpu = eng.gpu;
  // Working resolution: iPhone memory decides, not desktop assumptions.
  const factorFor = (w: number, h: number) => {
    const mp = (w * h) / 1e6;
    let f = 1;
    if (resolution === "half") f = 2;
    else if (resolution === "auto") f = isMobile() && mp > 16 ? 2 : 1;
    while (Math.max(w, h) / f > gpu.info.maxTextureDimension2D) f++;
    return f;
  };
  let decoded: DecodedImage, work: WorkingImage;
  {
    eng.progress("decode", file.name);
    // The file bytes are only needed by the decoder (which copies them): no
    // reference is kept here, so 30–80 MB can be collected during development.
    decoded = await P.time("decode", async () => decodeFile(new Uint8Array(await file.arrayBuffer()), file.name, file.type), (d) => `${d.format} via ${d.source.kind === "rgb" ? d.source.decoder : "LibRaw"}`);
    const dec = decoded;
    track(() => dec.close());
    for (const [k, v] of Object.entries(decoded.timings)) eng.log(`  ${k}: ${v.toFixed(0)} ms`);
    const src0 = decoded.source;
    const factor = factorFor(src0.width, src0.height);
    eng.progress("develop", `${src0.width}×${src0.height}${factor > 1 ? ` → 1/${factor}` : ""}`);
    const w0 = await P.time("raw development", () => develop(gpu, dec, { factor }), (w) => `${w.width}×${w.height}`);
    work = w0;
    track(() => gpu.release(w0.tex));
    work.log.forEach((l) => eng.log(l));
    // The sensor data now lives on the GPU; free the decoder's wasm heap. The
    // raw view points into that heap, so it must be dropped as well — otherwise
    // the whole LibRaw memory (≈150 MB for 12 MP, ≈460 MB for 48 MP) stays
    // alive for as long as the photo is open.
    decoded.close();
    decoded.close = () => {};
    if (decoded.source.kind !== "rgb") decoded.source.data = new Uint16Array(0);
  }
  const src = decoded.source;
  if (gen !== eng.generation) return;

  // --- reduced analysis image + exposure normalisation gain ------------------
  eng.progress("analysis image");
  // The networks see a ~1036 px image: segmentation slides 512 px windows over
  // it and depth adds high-resolution detail tiles to a global pass. Still a
  // reduced image — never the 12/48 MP original.
  // Phones: "light" at most — one 512 px segmentation pass and a 392 px depth pass
  // (a sixth of the segmentation work, about half the depth memory). The page
  // may ask for less after the tab died during analysis on this device.
  const lvl: AnalysisLevel = isMobile() && (!level || level === "full") ? "light" : (level ?? "full");
  const aScale = Math.min(1, (lvl === "full" ? ANALYSIS_LONG : ANALYSIS_LONG_LIGHT) / Math.max(work.width, work.height));
  const gw = Math.max(16, Math.round(work.width * aScale)), gh = Math.max(16, Math.round(work.height * aScale));
  const lin = await downsample(gpu, work.tex, work.width, work.height, gw, gh, false, 1, "analysis.lin");
  const linHalf = new Uint16Array(await gpu.readTexture(lin, 0, 0, gw, gh, 8));
  gpu.release(lin);
  const linF = halvesToFloats(linHalf);
  const gain = exposureGain(linF);
  const analysisRgba = new Float32Array(gw * gh * 4);
  for (let i = 0; i < gw * gh * 4; i += 4) {
    for (let c = 0; c < 3; c++) {
      const v = Math.min(1, Math.max(0, linF[i + c] * gain));
      analysisRgba[i + c] = srgbOetf(v);
    }
    analysisRgba[i + 3] = 1;
  }
  eng.log(`analysis image ${gw}×${gh}; normalisation gain ${gain.toFixed(3)} (${Math.log2(gain).toFixed(2)} EV)`);

  // --- semantic segmentation + depth (reduced image only) ------------------------
  // Phones (and devices where analysis crashed before): on the CPU in a worker of its
  // own that is terminated afterwards — the model runtime's memory is returned at
  // once instead of staying for the tab's life. Elsewhere: WebGPU, in this worker.
  const isolated = isMobile() || safeAnalysis;
  const withDepth = lvl !== "seg";
  const depthLong = lvl === "full" ? 518 : 392;
  const detailTiles = lvl === "full" && !isMobile(); // 4 more depth passes, too heavy for phones
  const inHere = () => analyseScene(eng.neural, { rgba: analysisRgba, width: gw, height: gh }, (s) => eng.progress(s), withDepth, detailTiles, safeAnalysis || isMobile() ? "wasm" : eng.neural.backend, depthLong);
  const img = { rgba: analysisRgba, width: gw, height: gh };
  const inWorker = () => analyseSceneIsolated(img, eng.base, detailTiles, (s) => eng.progress(s), withDepth, depthLong, phoneForced());
  if (lvl !== "full") eng.log(`scene analysis level: ${lvl}${lvl === "none" ? " (the tab stopped during segmentation before on this device)" : lvl === "seg" ? " (the tab stopped during depth before on this device)" : ""}`);
  const scene = await P.time("segmentation + depth", async () => {
    if (lvl === "none") return neutralScene(img, "skipped on this device");
    if (!isolated) return inHere();
    try { return await inWorker(); }
    catch (e) {
      eng.log(`analysis worker failed (${e instanceof Error ? e.message : e}); trying once more`);
      try { return await inWorker(); }
      catch (e2) {
        // On a phone a second failure is memory: analysing in this worker would keep
        // the model runtime's memory for good. Open without the analysis instead.
        if (isMobile()) return neutralScene(img, e2 instanceof Error ? e2.message : String(e2));
        return inHere();
      }
    }
  }, (s) => Object.entries(s.timings).map(([k, v]) => `${k} ${v.toFixed(0)}ms`).join(", "));
  if (import.meta.env.DEV) {
    // Dev only: dump the analysis image and distance map (PGM) for offline inspection.
    const pgm = (w: number, h: number, v: (i: number) => number) => {
      const head = new TextEncoder().encode(`P5 ${w} ${h} 255\n`);
      const out = new Uint8Array(head.length + w * h);
      out.set(head);
      for (let k = 0; k < w * h; k++) out[head.length + k] = Math.max(0, Math.min(255, Math.round(v(k) * 255)));
      return out;
    };
    const d = scene.depth;
    fetch("/__debug/save?name=depth.pgm", { method: "POST", body: pgm(d.width, d.height, (k) => d.dist[k]) }).catch(() => undefined);
    fetch("/__debug/save?name=analysis.pgm", { method: "POST", body: pgm(gw, gh, (k) => analysisRgba[k * 4 + 1]) }).catch(() => undefined);
  }
  // A ProRAW file carries Apple's own sky / skin mattes: sharper edges than
  // the network can produce on a reduced image, and already computed.
  if (decoded.masks?.length) scene.log.push(...applyAppleMattes(scene.seg, decoded.masks));
  // The skin matte also goes to the GPU: the look's skin protection uses it
  // directly instead of guessing skin from the person mask and its colour.
  // With it (g), the portrait subject matte: exact to the hair, it keeps a person in
  // focus whole under depth of field (the depth map is too coarse for flyaway hair).
  const skinMatte = decoded.masks?.find((m) => m.kind === "skin");
  const subjMatte = decoded.masks?.find((m) => m.kind === "subject");
  let skinTex: GPUTexture | undefined;
  const size = skinMatte ?? subjMatte;
  if (size) {
    const w = size.width, h = size.height, rg = new Uint8Array(w * h * 2);
    const put = (m: typeof size | undefined, c: number) => {
      if (!m) return;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) rg[(y * w + x) * 2 + c] = m.data[Math.min(m.height - 1, Math.floor((y * m.height) / h)) * m.width + Math.min(m.width - 1, Math.floor((x * m.width) / w))];
    };
    put(skinMatte, 0); put(subjMatte, 1);
    skinTex = gpu.tex("apple.mattes", w, h, "rg8unorm", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST);
    const t = skinTex;
    track(() => gpu.release(t));
    gpu.device.queue.writeTexture({ texture: skinTex }, rg, { bytesPerRow: w * 2, rowsPerImage: h }, { width: w, height: h });
    eng.log(`Apple mattes ${w}×${h}: ${[skinMatte && "skin (look's skin protection)", subjMatte && "portrait subject (kept sharp under depth of field)"].filter(Boolean).join(", ")}`);
  }
  scene.log.forEach((l) => eng.log(l));
  if (gen !== eng.generation) return; // (freed by open())

  // --- refinement ---------------------------------------------------------------
  eng.progress("refine masks");
  const maps = await P.time("mask/depth refinement", () => refine(gpu, work.tex, work.width, work.height, gain, scene), (m) => `guide ${m.w}×${m.h}, r=${m.params.maskRadius}`);
  track(() => releaseRefined(gpu, maps));

  // --- statistics -------------------------------------------------------------------
  eng.progress("statistics");
  const report = await P.time("image statistics", async () => {
    const b0 = await measureBlocks(gpu, work.tex, work.width, work.height, gain, 0.02);
    const noise = noiseProfile(b0);
    const thr = Math.max(0.02, 8 * noise.mid);
    const blocks = thr > 0.0201 ? await measureBlocks(gpu, work.tex, work.width, work.height, gain, thr) : b0;
    const blur = blurReport(blocks, noise);
    const reg = await measureRegions(gpu, maps, gain);
    const r: AnalysisReport = {
      width: work.width, height: work.height, gain, referred: work.referred,
      isProRaw: src.kind !== "rgb" && src.isProRaw, iso: decoded.meta.iso,
      global: reg.regions.global, groups: reg.regions as AnalysisReport["groups"],
      histLum: reg.histLum, histRGB: reg.histRGB, atmosphere: reg.atmosphere, noise, blur, blocks,
      lum: lumPercentiles(reg.histLum), timings: {},
    };
    return r;
  }, (r) => `noise σ ${(r.noise.mid * 255).toFixed(2)}/255, blur ${r.blur.median.toFixed(2)}px`);
  eng.log(`noise bins (y: σ/255 luma, chroma, blocks): ` + report.noise.bins.map((b) => `${b.y.toFixed(2)}: ${(b.sigma * 255).toFixed(2)}, ${(b.sigmaC * 255).toFixed(2)}, ${b.blocks}`).join(" | "));
  eng.log(`regions: ` + Object.entries(report.groups).filter(([, g]) => g.area > 0.01).map(([k, g]) => `${k} ${(g.area * 100).toFixed(0)}% ${g.meanEV.toFixed(2)}EV C${g.chroma.toFixed(3)} d${g.dist.toFixed(2)}`).join("; "));
  eng.log(`noise σ (encoded, /255): mid ${(report.noise.mid * 255).toFixed(2)}, shadows ${(report.noise.shadow * 255).toFixed(2)}, chroma ${(report.noise.chroma * 255).toFixed(2)}; blur median ${report.blur.median.toFixed(2)} px over ${report.blur.edgeBlocks} edge blocks`);

  // --- decisions ----------------------------------------------------------------------
  const colorInput = src.kind !== "rgb" ? src.color : undefined;
  // The camera's own rendering (the JPEG inside a DNG): the brightness reference.
  const reference = src.kind !== "rgb" && /\.dng$/i.test(file.name) ? await P.time("camera rendering", () => embeddedPreviewStats(file, REF_QS, isMobile() ? 24 : Infinity)) : undefined;
  if (reference) eng.log(`camera rendering: ${reference.width}×${reference.height} embedded JPEG, median ${(reference.q[MEDIAN] * 255).toFixed(0)}/255, shadows ${reference.q.slice(0, MEDIAN).map((v) => Math.round(v * 255)).join("/")}`);
  const decideWith = (referenceExposure?: { ev: number; note: string }) => decide({
    report,
    camera: work.camera,
    referred: work.referred,
    isProRaw: report.isProRaw,
    iso: decoded.meta.iso,
    sceneEV: sceneEV(decoded.meta),
    autoExposure,
    solveNeutral: colorInput ? (n) => neutralToTempTint(colorInput, n) : undefined,
    referenceExposure,
  });
  const decision = await P.time("decision engine", () => {
    let d = decideWith();
    if (!reference || !autoExposure) return d;
    // Exposure at which our rendering's median matches the camera's: solved on the
    // display model, then decided again (tone and local settings follow exposure).
    for (let k = 0; k < 2; k++) {
      const m = (e: number) => displayQuantiles(report.global.hist, { tone: d.params.tone, exposure: e, local: { ...d.params.local, anchorEV: d.params.local.anchorEV + d.params.exposure - e } }, [0.5])[0];
      let lo = -1.5, hi = 2.5;
      for (let it = 0; it < 30; it++) { const mid = (lo + hi) / 2; if (m(mid) < reference.q[MEDIAN]) lo = mid; else hi = mid; }
      const ev = Math.round(((lo + hi) / 2) * 100) / 100;
      d = decideWith({ ev, note: `matched to the camera's rendering (median ${(reference.q[MEDIAN] * 255).toFixed(0)}/255)` });
    }
    return d;
  });
  const params = structuredClone(decision.params);
  // Atmospheric light: dark-channel estimate is in the analysis encoding → linear working.
  const A = decision.params.dehaze.light.map((v) => srgbEotf(v) / gain) as [number, number, number];

  const s: Session = { name: file.name, file, decoded, work, denoised: work.tex, gain, scene, maps, report, decision, params, lightLinear: A, scale: 1, skin: skinTex };
  if (reference && autoExposure) s.calib = { ref: reference.q, refHi: reference.qHi, rounds: 0, black: false, chroma: reference.chroma, chroma95: reference.chroma95 };
  eng.s = s;
  commit(); // from here closeSession() frees it
  await eng.cacheDistance();
  // Automatic focus: subject from refined depth + segmentation + composition.
  const af = autoFocus(s.distCPU!, scene.seg, { blur: { bw: report.blocks.bw, bh: report.blocks.bh, data: report.blur.perBlock }, longPx: Math.max(work.width, work.height) });
  // Without a depth map (it failed on this device) nothing can be separated by distance.
  const flatDepth = !!scene.depth.flat;
  decision.dofSuggestion = { justified: af.justified && !flatDepth, focus: af.focus, strength: af.strength, reason: flatDepth ? "no depth map on this device — no automatic blur" : af.reason, x: af.x, y: af.y };
  // Depth zones by natural breaks of this photo's depth (boundaries fall in the
  // gaps between layers). Initial blur per zone follows the automatic focus
  // curve at the zone's mean distance, so switching to zones is seamless.
  {
    const zones = depthZones(s.distCPU!, scene.seg, 5);
    const smooth = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
    const curve = (c: number) => Math.max(smooth(0, 0.55, c - af.focus), 0.6 * smooth(0, 0.4, af.focus - c));
    decision.dofSuggestion.zoneEdges = [0, ...zones.slice(1).map((z) => z.lo), 1];
    decision.dofSuggestion.zones = zones.map((z) => ({ share: z.share, label: z.label, lo: z.lo, hi: z.hi }));
    for (const p of [decision.params, params]) {
      p.dof.zoneBounds = zones.slice(1).map((z) => z.lo);
      p.dof.zones = zones.map((z) => Math.round(curve(z.center) * 100) / 100);
      p.dof.mode = "focus";
    }
    eng.log("depth zones (natural breaks): " + zones.map((z, i) => `${i + 1}: ${z.lo.toFixed(2)}–${z.hi.toFixed(2)} ${Math.round(z.share * 100)}% ${z.label}`).join(" | "));
    // Distance bands for curves (near / middle / far): the same natural breaks, in three.
    const z3 = depthZones(s.distCPU!, scene.seg, 3);
    const b1 = z3[1].lo, b2 = Math.max(z3[2].lo, b1 + 0.02);
    for (const p of [decision.params, params]) p.depthBands = [b1, b2];
    decision.dofSuggestion.bands = z3.map((z) => ({ share: z.share, label: z.label, lo: z.lo, hi: z.hi }));
    decision.cellCoverage = cellCoverage(scene.seg, s.distCPU!, [b1, b2]);

    // Atmospheric perspective, only in outdoor scenes with real depth and clear air
    // (hazy scenes already have it): the distance a little quieter and cooler.
    const nearShare = z3[0].share, farShare = z3[2].share;
    // Outdoors only (visible sky): the far side of a room is not atmosphere.
    if (!flatDepth && nearShare >= 0.1 && farShare >= 0.2 && params.dehaze.strength < 0.15 && report.groups.sky.area >= 0.03) {
      for (const p of [decision.params, params]) p.distance = { ...p.distance, far: { ...p.distance.far, saturation: -0.06, warmth: -0.04, clarity: 0.9, texture: 0.9 } };
      decision.decisions.push({ id: "distance.far", value: "atmospheric perspective", reason: `depth: ${Math.round(nearShare * 100)}% near, ${Math.round(farShare * 100)}% far, clear air → the distance slightly less saturated, textured and cooler`, inputs: { near: nearShare, far: farShare } });
    }
    eng.log("distance bands for curves: " + z3.map((z, i) => `${["near", "middle", "far"][i]} ${z.lo.toFixed(2)}–${z.hi.toFixed(2)} ${Math.round(z.share * 100)}% ${z.label}`).join(" | "));

    // Natural skin: measured on the skin itself; pulled back only when overdriven.
    {
      const m = s.maps;
      const lin = new Float32Array(m.w * m.h * 4);
      halvesToFloats(new Uint16Array(await eng.gpu.readTexture(m.lin, 0, 0, m.w, m.h, 8)), lin);
      const apple = decoded.masks?.find((k) => k.kind === "skin");
      const st = measureSkin(lin, m.w, m.h, { ...scene.seg, plane: GROUPS.indexOf("person") * scene.seg.width * scene.seg.height }, apple);
      if (st) {
        const fix = naturalSkin(st, params, params.skin);
        if (fix.reasons.length) {
          for (const p of [decision.params, params]) p.skin = { ...p.skin, saturation: Math.round((p.skin.saturation + fix.saturation) * 100) / 100, hue: Math.round((p.skin.hue + fix.hue) * 10) / 10 };
          decision.decisions.push({ id: "skin", value: [params.skin.saturation, params.skin.hue], reason: fix.reasons.join("; "), inputs: { share: Math.round(st.share * 1000) / 10, L: Math.round(st.L * 100) / 100, C: Math.round(st.C * 1000) / 1000, hue: Math.round(st.hue) } });
        } else decision.decisions.push({ id: "skin", value: "natural", reason: `skin already natural (chroma ${st.C.toFixed(3)}, hue ${st.hue.toFixed(0)}°) — left alone`, inputs: {} });
      }
    }
    // Automatic curves: the photo, regions, skin and distance, measured through the rendering.
    const bandHist = await eng.depthBandHistograms(s, b1, b2);
    const st = (g: Group) => ({ hist: report.groups[g].hist, area: report.groups[g].area, localContrast: report.groups[g].localContrast });
    const ac = autoCurves({
      tone: params.tone, exposure: params.exposure, local: params.local, clipHi: report.global.clipHi,
      chroma: report.global.chroma, haze: params.dehaze.strength,
      photo: { hist: report.global.hist, area: 1 },
      regions: { sky: st("sky"), vegetation: st("vegetation"), water: st("water"), building: st("building"), person: st("person"), ground: st("ground") },
      bands: flatDepth ? undefined : bandHist, // one distance everywhere: no near / far curves
    });
    decision.autoCurves = { photo: ac.photo, regions: ac.regions, depth: ac.depth };
    for (const p of [decision.params, params]) applyAutoCurves(p, decision.autoCurves, p.autoCurves ?? 1);
    decision.decisions.push(...ac.notes);
    if (!ac.notes.length) decision.decisions.push({ id: "curves", value: "flat", reason: "every region, skin and distance already renders within its comfortable range — no automatic curves", inputs: {} });
    eng.log(`automatic curves: ${ac.notes.map((n) => n.id).join(", ") || "none"}`);
  }
  // Subject priority: the main subject may get +0.1…+0.25 EV — only when it does
  // not already stand out from its surroundings.
  {
    const g = af.kind === "person" || af.kind === "animal" || af.kind === "vehicle" || af.kind === "building" ? af.kind : undefined;
    const st = g ? report.groups[g] : undefined;
    if (g && st && st.area >= 0.01 && st.area <= 0.45) {
      const sep = st.meanEV - report.global.meanEV;
      if (sep < 0.3) {
        const add = Math.round(Math.min(0.25, Math.max(0.1, 0.1 + (0.3 - sep) * 0.25)) * 100) / 100;
        // A visible, editable layer: "Subject priority" (Exposure on the subject's region).
        for (const p of [decision.params, params]) {
          p.layers = [...(p.layers ?? []), makeLayer("exposure", "Subject priority", { auto: "subject", mask: { ...allMask(), kind: "region", region: g }, params: { exposure: add, offset: 0, gamma: 1 } })];
        }
        decision.decisions.push({ id: "subject", value: add, reason: `main subject (${g}) only ${sep.toFixed(2)} EV above the scene → +${add} EV luminance priority`, inputs: { separationEV: Math.round(sep * 100) / 100, area: st.area } });
      } else decision.decisions.push({ id: "subject", value: 0, reason: `main subject (${g}) already stands out (${sep.toFixed(2)} EV above the scene) — not brightened`, inputs: {} });
    }
  }
  // Keep the decision trace consistent with what auto focus found.
  const dofNote = decision.decisions.find((d) => d.id === "dof");
  if (dofNote) { dofNote.value = af.justified ? "justified" : "not justified"; dofNote.reason = `auto focus at distance ${af.focus.toFixed(2)}: ${af.reason}`; }
  for (const p of [decision.params, params]) {
    // Blur amount from the lens model (auto focus) when the photo calls for it; the
    // default is only a starting point for turning depth of field on by hand.
    p.dof = { ...p.dof, focus: af.focus, focusSpan: af.span, protect: protectGroup(af.kind), strength: af.justified ? af.strength : DEFAULT_DOF_STRENGTH, points: [] };
    if (autoDof && af.justified) p.enable = { ...p.enable, dof: true };
  }
  eng.log(`auto focus: distance ${af.focus.toFixed(2)} at (${af.x.toFixed(2)}, ${af.y.toFixed(2)}) — ${af.justified ? "blur justified" : "no blur"}: ${af.reason}${autoDof && af.justified ? " — applied (Auto depth of field)" : ""}`);
  // The automatic grade becomes layers (src/layers/auto.ts): visible, editable, removable.
  Object.assign(decision.params, buildAutoLayers(decision.params));
  Object.assign(params, buildAutoLayers(params));
  eng.post({ type: "analysis", summary: eng.summary(file.name), decisions: decision.decisions, auto: decision.params, params, dof: decision.dofSuggestion, exposureSuggestion: decision.exposureSuggestion, autoCurves: decision.autoCurves, cellCoverage: decision.cellCoverage, cameraWB: { temp: work.camera?.temp ?? 6504, tint: work.camera?.tint ?? 0 }, noDepth: flatDepth ? (scene.log.find((l) => /Depth skipped|Depth unavailable|Scene analysis unavailable/.test(l)) ?? "no depth map") : undefined });
  eng.post({ type: "profile", stages: P.stages });

  // --- first preview (before neural restoration) --------------------------------------
  await P.time("preview proxy", () => eng.makeProxy());
  await eng.renderNow(false);
  eng.post({ type: "profile", stages: P.stages });
  if (gen !== eng.generation) return;

  // --- neural restoration (tiled, only where needed) -------------------------------------
  await eng.restore(gen);
  if (gen !== eng.generation) return;

  // --- image quality → optional 2× upscale ----------------------------------------------
  // Measured on the restored image (after denoise), before any tone or
  // look. When the source already has enough detail this is the only cost:
  // the upscaling model is never downloaded or loaded.
  const q = await P.time("quality analysis", () => eng.analyseQuality(resolution === "half", upscaleMode), (r) => `${r.megapixels} MP, edge σ ${r.metrics.edgeSigma.toFixed(2)} px, noise ${(r.metrics.noiseSigma * 255).toFixed(2)}/255 → ${r.needsUpscale ? "2×" : "skip"}`);
  await P.time("preview proxy (restored)", () => eng.makeProxy());
  await eng.renderNow(true);
  eng.post({ type: "profile", stages: P.stages });
  // The upscale runs as its own queued job in short slices (see runUpscale), so
  // the photo is fully editable while it works.
  if (q.needsUpscale) void eng.runUpscale(gen);
}

export async function analyseQuality(eng: Engine, reducedByUser: boolean, mode: UpscaleMode) : Promise<ImageQualityReport> {
  const s = eng.s!;
  const { width: W, height: H } = s.work;
  const metrics = await measureQuality(eng.gpu, s.denoised, W, H, s.gain);
  const q = decideUpscale(metrics, {
    width: W, height: H, iso: s.decoded.meta.iso, reducedByUser, mode,
    maxOutputMP: UPSCALE_MAX_MP(), maxTextureDimension: eng.gpu.info.maxTextureDimension2D, mobile: isMobile(),
  });
  s.upscale = { state: q.needsUpscale ? "pending" : "skipped", upscaleApplied: false, upscaleFactor: 1, upscaleReason: q.reason, code: q.code, vars: q.vars, report: q };
  const m = q.metrics;
  eng.log(`quality: ${q.megapixels} MP; sharpness ${q.sharpnessScore.toFixed(2)} (edge σ ${m.edgeSigma.toFixed(2)} px sharpest quartile, ${m.edgeSigmaMedian.toFixed(2)} px median, ${m.edgeBlocks} edge blocks in ${m.patches} patches); ` +
    `noise ${(m.noiseSigma * 255).toFixed(2)}/255 (score ${q.noiseScore.toFixed(2)}); detail ${(m.detailDensity * 100).toFixed(1)}%; Laplacian var ${m.laplacianVar.toExponential(2)}; Tenengrad ${m.tenengrad.toExponential(2)}`);
  eng.log(`upscale: ${q.needsUpscale ? "2× planned" : "skipped"} — ${q.reason}`);
  eng.post({ type: "upscale", info: s.upscale });
  return q;
}

/** Keeps a CPU copy of the refined distance map for tap-to-focus. */
export async function cacheDistance(eng: Engine) {
  const s = eng.s!;
  const m = s.maps;
  const raw = new Float32Array(await eng.gpu.readTexture(m.masks[2], 0, 0, m.w, m.h, 16));
  const d = new Float32Array(m.w * m.h);
  for (let i = 0; i < d.length; i++) d[i] = raw[i * 4 + 3];
  s.distCPU = { w: m.w, h: m.h, data: d };
  eng.renderer.vanishing = vanishingPoint(s.distCPU);
}

export function summary(eng: Engine, name: string) : Summary {
  const s = eng.s!;
  const m = s.decoded.meta;
  const src = s.decoded.source;
  const meta: Record<string, string | number> = {};
  if (m.make) meta.camera = `${m.make} ${m.model ?? ""}`.trim();
  if (m.iso) meta.ISO = m.iso;
  if (m.exposureTime) meta.shutter = m.exposureTime >= 1 ? `${m.exposureTime}s` : `1/${Math.round(1 / m.exposureTime)}s`;
  if (m.fNumber) meta.aperture = `f/${m.fNumber.toFixed(1)}`;
  if (m.focalLength) meta.focal = `${m.focalLength.toFixed(1)}mm${m.focalLength35 ? ` (${m.focalLength35}mm eq.)` : ""}`;
  if (s.work.camera) { meta["as-shot WB"] = `${Math.round(s.work.camera.temp)}K / ${s.work.camera.tint.toFixed(1)}`; meta["baseline exposure"] = `${s.work.camera.baselineExposure.toFixed(2)} EV`; }
  return {
    file: name,
    format: s.decoded.format,
    source: src.kind === "rgb" ? `display-referred RGB (${src.decoder})` : src.isProRaw ? "Apple ProRAW (LinearRaw)" : src.kind === "bayer" ? "Bayer RAW" : "LinearRaw DNG",
    width: src.width,
    height: src.height,
    working: { width: s.work.width, height: s.work.height, factor: s.work.factor },
    meta,
    coverage: Object.fromEntries(Object.entries(s.scene.coverage).map(([k, v]) => [k, Math.round(v * 1000) / 10])),
  };
}
