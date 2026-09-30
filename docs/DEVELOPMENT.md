# <img src="../public/icon-192.png" width="36" height="36" alt="" align="absmiddle"> Shkarno — development

Live app: [img.jenyadoesapps.com](https://img.jenyadoesapps.com/)

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
