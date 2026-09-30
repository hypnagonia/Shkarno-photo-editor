#!/usr/bin/env node
/**
 * Quality benchmark of the automatic development: every photo in .samples/bench
 * is opened by the app as it opens on an iPhone (?phone), and scored on
 *   - the Check tab's findings on the automatic result (bad / warn)
 *   - its difference from the iPhone's own rendering of the same file (the JPEG
 *     embedded in the DNG; a JPEG / HEIC is its own reference): exposure (median L),
 *     colour (mean ΔE), colourfulness (chroma ratio)
 * and put side by side (Apple | ours) in a contact sheet.
 *
 *   npm run bench                      all photos, compared with the saved baseline
 *   npm run bench -- --save-baseline   … and make this run the baseline
 *   npm run bench -- --photo IMG_1847.DNG
 *   npm run bench -- --desktop         the desktop path instead of the phone's
 *   npm run bench -- --jobs 1          one photo at a time (default 3 in parallel)
 *
 * Output: .samples/out/bench/ (report.json, sheet.jpg, per photo ours / ref JPEGs).
 */
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : d; };
const flag = (k) => args.includes(`--${k}`);
const port = 5398;
const OUT = ".samples/out/bench";
const REPORT = ".samples/out/autotest.jsonl";
const SRGB = "/System/Library/ColorSync/Profiles/sRGB Profile.icc";
mkdirSync(OUT, { recursive: true });

const photos = (opt("photo")?.split(",") ?? readdirSync(".samples/bench").filter((f) => /\.(dng|heic|heif|jpe?g)$/i.test(f)).sort());
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ image maths
/** A photo as sRGB 8-bit pixels at a fixed small size (colour-managed by sips first). */
function pixels(file, w = 160, h = 160) {
  const tmp = join(OUT, "_px.jpg");
  execFileSync("sips", ["-s", "format", "jpeg", "--matchTo", SRGB, file, "--out", tmp], { stdio: "ignore" });
  const raw = execFileSync("magick", [tmp, "-auto-orient", "-resize", `${w}x${h}!`, "-depth", "8", "rgb:-"], { maxBuffer: 1 << 26 });
  return new Uint8Array(raw);
}
const lin = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
function lab(px) {
  const out = new Float64Array((px.length / 3) * 3);
  const f = (t) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
  for (let i = 0, j = 0; i < px.length; i += 3, j += 3) {
    const r = lin(px[i]), g = lin(px[i + 1]), b = lin(px[i + 2]);
    const X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047, Y = 0.2126 * r + 0.7152 * g + 0.0722 * b, Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    const fx = f(X), fy = f(Y), fz = f(Z);
    out[j] = 116 * fy - 16; out[j + 1] = 500 * (fx - fy); out[j + 2] = 200 * (fy - fz);
  }
  return out;
}
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
function compare(ours, ref) {
  const A = lab(ours), B = lab(ref), n = A.length / 3;
  const LA = [], LB = [];
  let dE = 0, cA = 0, cB = 0, clip = 0, crush = 0;
  for (let i = 0; i < n; i++) {
    const a = [A[i * 3], A[i * 3 + 1], A[i * 3 + 2]], b = [B[i * 3], B[i * 3 + 1], B[i * 3 + 2]];
    LA.push(a[0]); LB.push(b[0]);
    dE += Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    cA += Math.hypot(a[1], a[2]); cB += Math.hypot(b[1], b[2]);
    if (a[0] > 97) clip++;
    if (a[0] < 3) crush++;
  }
  return { dL: median(LA) - median(LB), dE: dE / n, chroma: cA / Math.max(cB, 1e-6), clip: (clip / n) * 100, crush: (crush / n) * 100 };
}

/**
 * The iPhone's own rendering of a DNG: the largest JPEG embedded in it (what the phone
 * showed). Segments are walked to the scan, then the scan to its end marker (an EXIF
 * thumbnail inside the JPEG cannot end it early).
 */
function embeddedJpeg(file) {
  const b = readFileSync(file);
  let best;
  for (let i = 0; i < b.length - 4; i++) {
    if (b[i] !== 0xff || b[i + 1] !== 0xd8 || b[i + 2] !== 0xff) continue;
    let p = i + 2, w = 0, h = 0, end = -1;
    while (p < b.length - 4 && b[p] === 0xff) {
      const m = b[p + 1], len = (b[p + 2] << 8) | b[p + 3];
      if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) { h = (b[p + 5] << 8) | b[p + 6]; w = (b[p + 7] << 8) | b[p + 8]; }
      if (m === 0xda) {
        let q = p + 2 + len;
        while (q < b.length - 1 && !(b[q] === 0xff && b[q + 1] !== 0 && !(b[q + 1] >= 0xd0 && b[q + 1] <= 0xd7))) q++;
        if (b[q + 1] === 0xd9) end = q + 2;
        break;
      }
      p += 2 + len;
    }
    if (end > 0 && w * h > (best?.w ?? 0) * (best?.h ?? 0) && Math.max(w, h) >= 1000) best = { start: i, end, w, h };
  }
  return best ? b.subarray(best.start, best.end) : undefined;
}

// ------------------------------------------------------------------ run the app
function chrome(url) {
  const bin = ["/Applications/Google Chrome.app", "/Applications/Google Chrome 2.app", "/Applications/Chromium.app"]
    .map((a) => join(a, "Contents/MacOS", a.includes("Chromium") ? "Chromium" : "Google Chrome")).find(existsSync);
  if (!bin) { console.error("Chrome not found in /Applications"); process.exit(2); }
  const profile = mkdtempSync(join(tmpdir(), "bench-"));
  const proc = spawn(bin, ["--headless=new", "--enable-unsafe-webgpu", "--no-first-run", "--no-default-browser-check", `--user-data-dir=${profile}`, url], { stdio: "ignore" });
  return async () => { proc.kill(); await sleep(1500); try { rmSync(profile, { recursive: true, force: true }); } catch { /* the OS cleans up */ } };
}

if (!flag("no-build")) { console.log("building…"); execFileSync("npm", ["run", "build"], { stdio: "ignore" }); }
const server = spawn("npx", ["vite", "preview", "--port", String(port), "--strictPort", "--host", "localhost"], { stdio: "ignore" });
process.on("exit", () => server.kill());
await sleep(2500);

const results = {};
// --reuse: the app's renders and findings of the last run, only the comparison again.
const last = flag("reuse") && existsSync(join(OUT, "report.json")) ? JSON.parse(readFileSync(join(OUT, "report.json"), "utf8")) : undefined;
// Photos run a few at a time (--jobs, default 3), each in a browser of its own; their
// report lines are told apart by the photo they name.
rmSync(REPORT, { force: true });
async function runPhoto(photo) {
  const tag = `bench-${photo.replace(/\.[^.]+$/, "")}`;
  if (last?.[photo] && !last[photo].failed && existsSync(join(OUT, `${tag}-ours.jpg`))) {
    execFileSync("cp", [join(OUT, `${tag}-ours.jpg`), join(".samples/out", `${tag}.jpg`)]);
    const L = last[photo];
    await score(photo, tag, L.checkItems ?? [...L.bad.map((id) => ({ id, level: "bad" })), ...L.warn.map((id) => ({ id, level: "warn" }))], "done");
    return;
  }
  rmSync(join(".samples/out", `${tag}.jpg`), { force: true });
  const stop = chrome(`http://localhost:${port}/?autotest${flag("desktop") ? "" : "&phone"}&close&save&tag=${tag}&photo=${encodeURIComponent(`bench/${photo}`)}&steps=open,check,snap&run=${Date.now()}`);
  let check, done = "", t0 = Date.now();
  while (Date.now() - t0 < 300_000) {
    await sleep(1000);
    if (!existsSync(REPORT)) continue;
    for (const l of readFileSync(REPORT, "utf8").trim().split("\n").filter(Boolean)) {
      let r;
      try { r = JSON.parse(l); } catch { continue; } // (a line being written)
      if (r.photo !== `bench/${photo}`) continue;
      if (r.stage === "check") check = r.items;
      if (r.stage === "done" || r.stage === "failed") done = r.stage + (r.message ? `: ${r.message}` : "");
    }
    if (done) break;
  }
  await stop();
  await score(photo, tag, check, done);
}
const queue = [...photos];
await Promise.all(Array.from({ length: Math.max(1, Number(opt("jobs", 3))) }, async () => { while (queue.length) await runPhoto(queue.shift()); }));
server.kill();

async function score(photo, tag, check, done) {
  const oursFile = join(".samples/out", `${tag}.jpg`);
  if (!done.startsWith("done") || !existsSync(oursFile)) { console.log(`${photo}: ✗ ${done || "timeout"}`); results[photo] = { failed: done || "timeout" }; return; }
  // The iPhone's own rendering: the JPEG inside the DNG (else macOS's rendering of the
  // file; a JPEG / HEIC is its own reference).
  const refFile = join(OUT, `${tag}-ref.jpg`);
  const src = join(".samples/bench", photo);
  const emb = /\.dng$/i.test(photo) ? embeddedJpeg(src) : undefined;
  if (emb) { writeFileSync(join(OUT, "_emb.jpg"), emb); execFileSync("sips", ["-s", "format", "jpeg", "--matchTo", SRGB, join(OUT, "_emb.jpg"), "--out", refFile], { stdio: "ignore" }); }
  else execFileSync("sips", ["-s", "format", "jpeg", "--matchTo", SRGB, src, "--out", refFile], { stdio: "ignore" });
  execFileSync("cp", [oursFile, join(OUT, `${tag}-ours.jpg`)]);
  const m = compare(pixels(oursFile), pixels(refFile));
  const bad = (check ?? []).filter((i) => i.level === "bad").map((i) => i.id), warn = (check ?? []).filter((i) => i.level === "warn").map((i) => i.id);
  // Penalty: Check findings first, then distance from Apple beyond what a grade may differ.
  const score = 3 * bad.length + warn.length + Math.abs(m.dL) / 3 + Math.max(0, m.dE - 6) / 3;
  results[photo] = { ...m, bad, warn, score, checkItems: check };
  console.log(`${photo.padEnd(16)} score ${score.toFixed(2).padStart(5)}  ΔL ${m.dL.toFixed(1).padStart(5)}  ΔE ${m.dE.toFixed(1).padStart(4)}  chroma ×${m.chroma.toFixed(2)}  clip ${m.clip.toFixed(1)}%  crush ${m.crush.toFixed(1)}%  ${bad.length ? "bad: " + bad.join(",") + "  " : ""}${warn.length ? "warn: " + warn.join(",") : ""}`);
}

// ------------------------------------------------------------------ report
// (In photo order, whichever finished first.)
for (const k of Object.keys(results).sort()) { const v = results[k]; delete results[k]; results[k] = v; }
writeFileSync(join(OUT, "report.json"), JSON.stringify(results, null, 2));
const ok = Object.entries(results).filter(([, r]) => !r.failed);
const total = ok.reduce((a, [, r]) => a + r.score, 0);
console.log(`\ntotal score ${total.toFixed(2)} over ${ok.length} photos (lower is better)`);
const basePath = join(OUT, "baseline.json");
if (existsSync(basePath) && !flag("save-baseline")) {
  const base = JSON.parse(readFileSync(basePath, "utf8"));
  let bt = 0, nt = 0;
  for (const [p, r] of ok) {
    const b = base[p];
    if (!b || b.failed) continue;
    bt += b.score; nt += r.score;
    const d = r.score - b.score;
    if (Math.abs(d) >= 0.25) console.log(`  ${d < 0 ? "better" : "WORSE "} ${p}: ${b.score.toFixed(2)} → ${r.score.toFixed(2)}`);
  }
  console.log(`baseline ${bt.toFixed(2)} → now ${nt.toFixed(2)} (${nt <= bt ? "not worse" : "WORSE"})`);
}
if (flag("save-baseline")) { writeFileSync(basePath, JSON.stringify(results, null, 2)); console.log("saved as the baseline"); }
// Contact sheet: Apple | ours, one row per photo.
const rows = ok.map(([p]) => { const t = `bench-${p.replace(/\.[^.]+$/, "")}`; return [join(OUT, `${t}-ref.jpg`), join(OUT, `${t}-ours.jpg`)]; });
if (rows.length) {
  const tiles = rows.flatMap(([a, b]) => [a, b]);
  execFileSync("magick", ["montage", ...tiles, "-auto-orient", "-geometry", "360x360+4+4", "-tile", "2x", join(OUT, "sheet.jpg")]);
  console.log(`contact sheet: ${join(OUT, "sheet.jpg")} (left: Apple, right: ours)`);
}
process.exit(0);
