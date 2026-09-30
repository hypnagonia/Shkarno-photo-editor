/**
 * Where downloaded models are kept (Cache Storage), shared by src/neural/ort.ts and the
 * page's idle prefetch (src/pwa.ts) — a module of its own so the page does not pull in
 * ONNX Runtime to know the name.
 */
export const MODEL_CACHE = "image-improver2-models-v1";

/** Tap-to-select (MobileSAM): its models and the plain WebAssembly runtime its worker runs on. */
export const SELECT_MODELS = ["/models/mobile-sam-encoder.onnx", "/models/sam-decoder-multi.onnx"];
export const SELECT_RUNTIME = ["/ort/ort-wasm-simd-threaded.mjs", "/ort/ort-wasm-simd-threaded.wasm"];
