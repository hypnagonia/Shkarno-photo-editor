/** Restoration after the first preview: GPU denoise, and the optional 2× upscale (Swin2SR). */
import type { Engine } from "./engine.ts";
import { Neural, MODELS } from "../neural/ort.ts";
import { denoiseGPU } from "../restore/denoise.ts";
import { UpscaleJob, probeUpscaler } from "../restore/upscale.ts";
import { decideUpscale } from "../analysis/quality.ts";
import { UPSCALE_MAX_MP } from "./session.ts";

/** "Upscale 2× now" from the Upscale tab: overrides the decision (never the memory budget). */
export function forceUpscale(eng: Engine) {
  const s = eng.s;
  const info = s?.upscale;
  if (!s || !info || s.scale !== 1 || info.state === "running" || info.state === "pending") return;
  const q = decideUpscale(info.report.metrics, {
    width: s.work.width, height: s.work.height, iso: s.decoded.meta.iso, reducedByUser: false, mode: "always",
    maxOutputMP: UPSCALE_MAX_MP(), maxTextureDimension: eng.gpu.info.maxTextureDimension2D,
  });
  s.upscale = { state: q.needsUpscale ? "pending" : "skipped", upscaleApplied: false, upscaleFactor: 1, upscaleReason: q.reason, code: q.code, vars: q.vars, report: q };
  eng.log(`upscale: requested — ${q.needsUpscale ? "2× planned" : "not possible"} (${q.reason})`);
  eng.post({ type: "upscale", info: s.upscale });
  if (q.needsUpscale) void eng.runUpscale(eng.generation);
}

/**
 * 2× upscale of the restored working image, as a chain of short exclusive
 * jobs: each slice runs tiles for ~250 ms, then preview renders and other
 * requests queued meanwhile get their turn. The result replaces the working
 * image atomically at the end, so every later stage (tone, look, semantic,
 * sharpening, depth of field, export) runs at the new resolution. Any
 * failure leaves the photo exactly as it was.
 */
export async function runUpscale(eng: Engine, gen: number) {
  const s = eng.s;
  if (!s?.upscale) return;
  const src = s.denoised;
  const { width: W, height: H } = s.work;
  const P = eng.profiler;
  const info = s.upscale;
  const post = () => eng.post({ type: "upscale", info });
  info.state = "running";
  post();
  let session: Awaited<ReturnType<Neural["session"]>> | undefined;
  let job: UpscaleJob | undefined;
  let backend = eng.neural.backend;
  const t0 = performance.now();
  try {
    eng.progress("detail enhancement", "loading model");
    // Load, then self-test on one probe tile: WebGPU first, WASM if either fails.
    const open = async (b: typeof backend) => {
      const ses = await eng.neural.session(MODELS.swin2sr, false, b);
      const bad = await probeUpscaler(ses);
      if (bad) { await ses.release(); throw new Error(bad); }
      return ses;
    };
    try {
      session = await open(backend);
    } catch (e) {
      if (backend !== "webgpu") throw e;
      eng.log(`Swin2SR on WebGPU failed (${e instanceof Error ? e.message : e}); retrying on WASM`);
      backend = "wasm";
      session = await open("wasm");
    }
    const live = () => gen === eng.generation && eng.s === s && s.denoised === src;
    for (let first = true; ; first = false) {
      const done = await eng.exclusive(async () => {
        if (!live()) throw new Error("cancelled");
        if (!job) job = new UpscaleJob(eng.gpu, session!, src, W, H, s.gain);
        try {
          return await job.step(first ? 0 : 250);
        } catch (e) {
          // A WebGPU failure mid-run: redo the whole image on WASM rather than give up.
          if (backend !== "webgpu") throw e;
          eng.log(`Swin2SR WebGPU inference failed (${e instanceof Error ? e.message : e}); retrying on WASM`);
          job.release();
          await session!.release();
          backend = "wasm";
          session = await eng.neural.session(MODELS.swin2sr, false, "wasm");
          job = new UpscaleJob(eng.gpu, session, src, W, H, s.gain);
          return false;
        }
      });
      const pr = job!.progress;
      eng.progress("detail enhancement", `tile ${pr.done}/${pr.total}`, pr.done / pr.total);
      if (done) break;
    }
    await eng.exclusive(async () => {
      if (!live()) throw new Error("cancelled");
      const out = job!.out;
      const tiles = job!.total;
      const netMs = job!.progress.msPerTile;
      job!.finish();
      job = undefined;
      // Swap in the 2× image: it is both base and restored image (restoration is baked in).
      if (s.denoised !== s.work.tex) eng.gpu.release(s.denoised);
      eng.gpu.release(s.work.tex);
      s.work = { ...s.work, tex: out, width: 2 * W, height: 2 * H };
      s.denoised = out;
      s.scale = 2;
      eng.scaleRetouch(s, 2);
      eng.dropThumb();
      P.add("2× upscale (Swin2SR)", performance.now() - t0, `${W}×${H} → ${2 * W}×${2 * H} on ${backend}, ${tiles} tiles`);
      Object.assign(info, { state: "applied", upscaleApplied: true, upscaleFactor: 2, width: 2 * W, height: 2 * H });
      eng.log(`upscale: 2× applied on ${backend} in ${((performance.now() - t0) / 1000).toFixed(1)} s — ${W}×${H} → ${2 * W}×${2 * H}, ${tiles} tiles, network ${netMs.toFixed(0)} ms/tile`);
      await eng.makeProxy();
      await eng.renderNow(true);
      post();
      eng.post({ type: "profile", stages: P.stages });
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    job?.release();
    if (msg === "cancelled") {
      eng.log("upscale: cancelled (photo changed or restoration re-run)");
      Object.assign(info, { state: "cancelled" });
    } else {
      // Never fail the photo because of the optional stage: keep the 1× image.
      eng.log(`upscale: failed (${msg}) — continuing without it`);
      Object.assign(info, { state: "failed", upscaleReason: `${info.upscaleReason}; model could not run: ${msg}` });
    }
    if (eng.s === s) post();
  } finally {
    await session?.release().catch(() => {});
  }
}

/** Noise-adaptive GPU denoise of the working image, when the decision engine asked for it. */
export async function restore(eng: Engine, _gen: number) {
  const s = eng.s!;
  const gpu = eng.gpu;
  const P = eng.profiler;
  const { width: W, height: H } = s.work;
  if (!s.decision.plan.denoise) { eng.log("denoise: noise below visibility — not needed"); return; }
  eng.progress("denoise", "GPU");
  s.denoised = await P.time("GPU denoise", () => denoiseGPU(gpu, s.work.tex, W, H, s.gain, s.report.noise), () => `${W}×${H}`);
  eng.log(`GPU denoise: full frame, noise-adaptive (σ mid ${(s.report.noise.mid * 255).toFixed(2)}/255)`);
  eng.dropThumb(); // look previews must see the restored image
}
