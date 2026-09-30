# <img src="../public/icon-192.png" width="36" height="36" alt="" align="absmiddle"> Shkarno — development

Live app: [img.jenyadoesapps.com](https://img.jenyadoesapps.com/)

## How it is tested

Shkarno is checked at three levels. The first runs on every push; the other two need a
real GPU and real photos, so they run on a developer machine — GitHub's hosted runners
have no GPU and no WebGPU, and the sample photos are private.

**1. Every push — [GitHub Actions](../.github/workflows/ci.yml)**

- `npx tsc --noEmit` — the whole codebase type-checks (strict TypeScript).
- `npm test` — 125 unit tests (`test/*.test.ts`, Node's test runner) on the logic that
  decides what the GPU does: colour math and colour spaces, the automatic decisions
  (curves, black point, skin tone, focus, depth zones), look profiles and palette
  matching, layer masks and gradients, tap-to-select, tone equalizer, HDR and Ultra HDR
  gain maps, JPEG/DNG preview decoding, film emulation, magic-brush geometry, the Check.
- `npm run build` — the production bundle builds.

**2. The real app on real photos — `src/autotest.ts`**

The app has a scripted mode (`/?autotest&photo=…&steps=…`, local only) that opens a photo
and runs steps in headless Chrome with WebGPU: open, select, blur, motion blur, fog,
light, magic brush and its undo, film, tone and contrast equalizers, export, reopen. Each
step reports timings, GPU memory and any GPU validation error (which fails the run), and
can save its export for inspection. The scripts below drive it.

- `npm run bench` — **quality benchmark**: 19 iPhone photos opened as on a phone, each
  scored against the iPhone's own rendering of the same file (exposure, colour
  difference, colourfulness) and by the Check's findings, compared with a saved baseline.
  Every change to the automatic development is measured with it.
- `npm run memcheck` — **memory guard**: the phone code path on sample photos, step by
  step, against a memory budget (below). `npm run deploy` runs it and refuses to ship a
  build that breaks the budget.

**3. What is not automated**

WGSL shaders, LibRaw (WebAssembly) and the neural networks are exercised end to end by
level 2, not unit-tested in isolation; their results are judged on the exported images.
Visual changes are reviewed on before/after exports.

## Develop

```sh
npm install
npm run dev          # http://localhost:5173 (COOP/COEP headers set)
npm test             # colour math, profiles, palette/matching, geometry
npm run build        # dist/
npm run bench        # automatic-development quality vs the iPhone's rendering
```

Rebuilding native parts (checked-in outputs, only needed when changing them):

```sh
LIBRAW_SRC=/path/to/LibRaw-0.22.2 npm run build:libraw          # needs emscripten
```

Models are prepared by the scripts in `scripts/models/` (sources, conversion and checks
in each file's header).

## Deploy

`npm run deploy`: build → memory guard → `vercel deploy --prod` (project
`image-improver2`). A deploy that breaks the memory budget does not ship.
`vercel.json` sets the cross-origin-isolation headers that multi-threaded WASM needs.

## Memory guard (phones)

`npm run memcheck` runs the iPhone code path (`?autotest&phone`, see
`src/autotest.ts` and `src/device.ts`) on sample photos in `.samples/` —
open, tap-to-select, blur layer, magic brush, export, reopen — in headless Chrome, and
fails when a step exceeds `scripts/memcheck.budget.json`:

- **page**: physical footprint of the page's process (the app and all its workers),
- **GPU process**: the browser's GPU process above its own overhead,
- **engine GPU**: every texture and buffer the engine allocates (exact, identical
  in every browser — the most reliable regression signal).

`--timeline` prints the peak per engine stage (decode, depth, selection…), Chrome's
JS/WebAssembly memory per worker, and what the page's footprint is made of.
`--browser safari` measures real WebKit instead (needs WebGPU in Safari: macOS 26).
iPhone browsers (Safari and Chrome alike) all run WebKit.

Rules the budget taught us:
- CPU-only model workers use ONNX Runtime's plain WebAssembly build
  (`useCpuRuntime`, ≈ 450 MB for MobileSAM instead of 720 MB with the
  WebGPU-capable build).
- One runtime per job: tap-to-select encodes and decodes in one worker (two
  runtimes side by side peaked near 2 GB).
- Heavy model workers are terminated when done: WebAssembly memory never shrinks.
