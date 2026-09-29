#!/usr/bin/env node
/**
 * Memory guard: runs the app's iPhone path on sample photos and fails when a step
 * needs more memory than the budget (scripts/memcheck.budget.json) — so an update
 * that would crash phones is caught before it ships.
 *
 *   npm run memcheck                         Chrome (headless), the default photos
 *   npm run memcheck -- --photo IMG_1514.DNG --steps open,select
 *   npm run memcheck -- --browser safari     real WebKit (needs WebGPU: macOS 26)
 *   npm run memcheck -- --no-build
 *
 * How: `vite preview` serves the production build; the browser opens
 * /?autotest&phone&… (src/autotest.ts), which behaves exactly as on an iPhone
 * (src/device.ts), opens the sample by itself and reports every step to
 * .samples/out/autotest.jsonl, including the engine's own GPU bookkeeping for the
 * step (every texture and buffer it allocates: exact, the same in every browser).
 * Meanwhile this script samples the physical footprint (what iOS's memory limit
 * counts) of the page's process — the app's workers run inside it — and of the
 * browser's GPU process, and keeps the peak of each per step.
 */
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : d; };
const flag = (k) => args.includes(`--${k}`);
const budgetAll = JSON.parse(readFileSync(new URL("./memcheck.budget.json", import.meta.url), "utf8"));
const browser = opt("browser", "chrome");
const budget = budgetAll[browser];
if (!budget) { console.error(`no budget for browser "${browser}"`); process.exit(2); }
const photos = (opt("photo") ?? budgetAll.photos.join(",")).split(",");
const stepsOpt = opt("steps");
const stepsFor = () => stepsOpt ?? budgetAll.steps.join(",");
const port = Number(opt("port", "5399"));
const REPORT = ".samples/out/autotest.jsonl";

const sh = (cmd, a) => { try { return execFileSync(cmd, a, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }); } catch { return ""; } };
const pids = (re) => sh("pgrep", ["-f", re]).split("\n").filter(Boolean).map(Number);
/** Physical footprint in MB (what iOS's memory limit counts), 0 when the process is gone. */
function footprint(pid) {
  const m = /Footprint:\s+([\d.]+)\s+(KB|MB|GB)/.exec(sh("footprint", ["-p", String(pid)]));
  if (!m) return 0;
  return Number(m[1]) * (m[2] === "GB" ? 1024 : m[2] === "KB" ? 1 / 1024 : 1);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
/** Child processes of `parent` whose command line contains `has`. */
const children = (parent, has) => sh("ps", ["-axo", "pid=,ppid=,command="]).split("\n")
  .map((l) => l.trim().match(/^(\d+)\s+(\d+)\s+(.*)$/)).filter((m) => m && Number(m[2]) === parent && m[3].includes(has)).map((m) => Number(m[1]));

/** Starts the browser on `url`; returns how to find its page and GPU processes, and how to stop it. */
function launch(url) {
  if (browser === "safari") {
    const WC = "StagedFrameworks/Safari/WebKit.framework.*com.apple.WebKit.WebContent";
    const before = new Set(pids(WC));
    execFileSync("open", ["-g", "-a", "Safari", url]);
    return { page: () => pids(WC).filter((p) => !before.has(p)), gpu: () => pids("StagedFrameworks/Safari/WebKit.framework.*com.apple.WebKit.GPU").slice(0, 1), stop: async () => {} };
  }
  const bin = ["/Applications/Google Chrome.app", "/Applications/Google Chrome 2.app", "/Applications/Chromium.app"]
    .map((a) => join(a, "Contents/MacOS", a.includes("Chromium") ? "Chromium" : "Google Chrome")).find(existsSync);
  if (!bin) { console.error("Chrome not found in /Applications"); process.exit(2); }
  const profile = mkdtempSync(join(tmpdir(), "memcheck-"));
  const proc = spawn(bin, ["--headless=new", "--enable-unsafe-webgpu", "--enable-blink-features=ForceEagerMeasureMemory", "--no-first-run", "--no-default-browser-check", `--user-data-dir=${profile}`, url], { stdio: "ignore" });
  return {
    // Chrome's helpers do not carry the profile on their command line: they are this Chrome's children.
    page: () => children(proc.pid, "--type=renderer"),
    gpu: () => children(proc.pid, "--type=gpu-process"),
    stop: async () => {
      const exited = new Promise((r) => proc.once("exit", r));
      proc.kill();
      await Promise.race([exited, sleep(5000)]);
      try { rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 300 }); } catch { /* left for the OS to clean */ }
    },
  };
}

if (!flag("no-build")) {
  console.log("building…");
  execFileSync("npm", ["run", "build"], { stdio: "ignore" });
}
const server = spawn("npx", ["vite", "preview", "--port", String(port), "--strictPort", "--host", "localhost"], { stdio: "ignore" });
process.on("exit", () => server.kill());
await sleep(2500);

const fmt = (v) => String(Math.round(v)).padStart(5);
let failed = false;
for (const photo of photos) {
  if (!existsSync(`.samples/${photo}`)) { console.log(`skip ${photo}: not in .samples`); continue; }
  rmSync(REPORT, { force: true });
  const what = `photo=${encodeURIComponent(photo)}`;
  const b = launch(`http://localhost:${port}/?autotest&phone&close&${what}&steps=${stepsFor(photo)}${flag("save") ? "&save" : ""}${opt("teq") ? `&teq=${opt("teq")}` : ""}${opt("ceq") ? `&ceq=${opt("ceq")}` : ""}${flag("logs") ? "&logs" : ""}&run=${Date.now()}`);
  console.log(`\n${photo} (${browser}): ${stepsFor(photo)}`);
  const peaks = new Map(); // step → { page, gpu, tracked }
  let stage = "load", sub = "", done = "", message = "", gpuBase = -1, seen = 0;
  const subPeaks = new Map(); // "step › engine stage" → page MB (--timeline)
  const t0 = Date.now();
  while (Date.now() - t0 < budgetAll.timeoutSec * 1000) {
    await sleep(250);
    const page = Math.max(0, ...b.page().map(footprint));
    const gpuNow = Math.max(0, ...b.gpu().map(footprint));
    if (existsSync(REPORT)) {
      const lines = readFileSync(REPORT, "utf8").trim().split("\n").filter(Boolean);
      for (const l of lines.slice(seen)) {
        const r = JSON.parse(l);
        if (r.stage === "boot") {
          gpuBase = gpuNow; // the GPU process's own overhead, before the app did anything
          if (!r.gpu || !r.adapter || !r.isolated) { done = "failed"; message = `browser: WebGPU ${r.gpu}/${r.adapter}, cross-origin isolated ${r.isolated}`; }
        }
        if (r.stage.endsWith(":start")) { stage = r.stage.slice(0, -6); sub = ""; }
        if (r.stage.startsWith("progress:") && r.stage.length > 9) sub = r.stage.slice(9) + (r.detail && /^(encode|decode|filter)$/.test(r.detail) ? `:${r.detail}` : "");
        if (flag("timeline") && r.stage.endsWith(":done")) {
          // What the page process's footprint is made of (top categories).
          const pid = b.page().map((p) => [p, footprint(p)]).sort((x, y) => y[1] - x[1])[0]?.[0];
          const txt = pid ? sh("footprint", ["-p", String(pid)]) : "";
          const cats = [...txt.matchAll(/^\s*([\d.]+ (?:KB|MB|GB))\s+[\d.]+ (?:B|KB|MB|GB)\s+[\d.]+ (?:B|KB|MB|GB)\s+\d+\s+(.+)$/gm)]
            .map((m) => [m[2].trim(), Number(m[1].split(" ")[0]) * (m[1].endsWith("GB") ? 1024 : m[1].endsWith("KB") ? 1 / 1024 : 1)])
            .sort((x, y) => y[1] - x[1]).slice(0, 6);
          console.log(`    ${r.stage} page footprint by category: ${cats.map(([k, v]) => `${k} ${Math.round(v)} MB`).join(" · ")}`);
        }
        if (flag("timeline") && r.js && r.stage.endsWith(":done")) console.log(`    ${r.stage} JS/wasm by worker (MB): ${Object.entries(r.js).filter(([, v]) => v >= 5).map(([k, v]) => `${k} ${v}`).join(" · ")}`);
        if (r.stage.endsWith(":done")) {
          // A step shorter than one sample still gets this sample (never a 0 row).
          const s = r.stage.slice(0, -5), p = peaks.get(s) ?? { page: 0, gpu: 0, tracked: 0 };
          const g = Math.max(0, gpuNow - Math.max(0, gpuBase));
          peaks.set(s, { page: Math.max(p.page, page), gpu: Math.max(p.gpu, g), tracked: Math.max(p.tracked, r.gpuPeakMB ?? 0) });
        }
        if (r.stage === "done" || r.stage === "failed") { done = done || r.stage; message = message || (r.message ?? ""); }
        if (r.stage === "error") message = r.message;
        // A GPU validation error: a dispatch was dropped and its result is wrong, whatever the memory says.
        if (r.stage === "gpuerror") { done = "failed"; message = r.message; }
      }
      seen = lines.length;
    }
    const gpu = Math.max(0, gpuNow - Math.max(0, gpuBase));
    if (sub) { const k = `${stage} › ${sub}`; subPeaks.set(k, Math.max(subPeaks.get(k) ?? 0, page)); }
    const p = peaks.get(stage) ?? { page: 0, gpu: 0, tracked: 0 };
    peaks.set(stage, { ...p, page: Math.max(p.page, page), gpu: Math.max(p.gpu, gpu) });
    if (done) break;
  }
  await b.stop();
  if (!done) { done = "failed"; message = `timeout after ${budgetAll.timeoutSec}s in ${stage}`; }
  console.log("  step          page MB   gpu proc MB   engine GPU MB");
  for (const [s, p] of peaks) {
    if (s === "load") continue;
    const lim = { ...budget, ...(budget.steps?.[s] ?? {}) };
    const over = [p.page > lim.pageMB && `page > ${lim.pageMB}`, p.gpu > lim.gpuProcMB && `gpu proc > ${lim.gpuProcMB}`, p.tracked > lim.engineGpuMB && `engine GPU > ${lim.engineGpuMB}`].filter(Boolean);
    if (over.length) failed = true;
    console.log(`  ${s.padEnd(10)} ${fmt(p.page)}      ${fmt(p.gpu)}        ${fmt(p.tracked)}${over.length ? "   ✗ " + over.join(", ") : ""}`);
  }
  if (flag("timeline")) for (const [k, v] of subPeaks) console.log(`    ${k.padEnd(34)} page ${fmt(v)} MB`);
  if (done === "failed") { failed = true; console.log(`  ✗ ${message}`); } else console.log("  ✓ every step ran");
  await sleep(1500);
}
server.kill();
console.log(failed ? "\nmemcheck FAILED" : "\nmemcheck passed");
process.exit(failed ? 1 : 0);
