/// <reference types="vite/client" />
declare module "*.wgsl?raw" { const s: string; export default s; }
declare module "*.svg?raw" { const s: string; export default s; }
declare module "libraw-js";
declare module "libheif-js/libheif-wasm/libheif-bundle.mjs";
declare const __ORT_EXTERNAL__: boolean;
/** This build's id (vite.config.ts): device settings learnt from a crash are retried after a deploy. */
declare const __BUILD__: string;
