// Copies ONNX Runtime Web's loader + wasm into public/ort. ORT spawns its
// WASM threads from this loader's URL; bundled into our worker it would try to
// spawn the engine worker instead (and hang in production).
import { copyFileSync, mkdirSync } from "node:fs";
const src = "node_modules/onnxruntime-web/dist/";
mkdirSync("public/ort", { recursive: true });
// The WebGPU-capable runtime (engine) and the plain WebAssembly one (CPU-only workers).
for (const f of ["ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) copyFileSync(src + f, "public/ort/" + f);
console.log("ORT runtime copied to public/ort");
