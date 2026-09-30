/**
 * ONNX Runtime Web session management.
 *
 * Backend order: WebGPU EP (sharing the engine's GPUDevice, so tensors can stay
 * on the GPU as storage buffers) → WASM SIMD EP. Model precision follows the
 * device: true fp16 graphs when the adapter exposes shader-f16, otherwise
 * fp16-weight/fp32-compute graphs (same download size, runs everywhere).
 *
 * Models are fetched from this origin only and cached with the Cache API, so
 * after the first visit the app works offline and nothing leaves the device.
 */
import * as ortWebgpu from "onnxruntime-web";
import type * as Ort from "onnxruntime-web";
import { MODEL_CACHE } from "./modelCache.ts";
import * as ortCpu from "onnxruntime-web/wasm";
import * as ortNative from "onnxruntime-web/webgpu";

/**
 * The ONNX Runtime in use. The engine has the WebGPU-capable build; workers that
 * only run on the CPU (scene analysis, tap-to-select) switch to the plain
 * WebAssembly build first (useCpuRuntime): measured on MobileSAM, 450 MB instead of
 * 720 MB and faster — the WebGPU-capable runtime costs that much even on the CPU.
 */
export let ort = ortWebgpu;
export function useCpuRuntime() { ort = ortCpu as unknown as typeof ortWebgpu; }
/**
 * ORT's native WebGPU execution provider (the C++ one, not the JavaScript JSEP kernels the
 * engine uses): the JSEP kernels break on LaMa's Fourier units (a reshape and an Add fail),
 * the native ones run it. For a worker of its own (the magic brush on computers), before
 * anything loads a runtime; WebAssembly stays available in the same build.
 */
export function useNativeWebgpuRuntime() { ort = ortNative as unknown as typeof ortWebgpu; }
import type { Gpu } from "../gpu/gpu.ts";
import { isIOS } from "../device.ts";

export type Backend = "webgpu" | "wasm";

export interface ModelSpec {
  id: "segformer" | "depth" | "swin2sr" | "samEncoder" | "samDecoder" | "lama" | "migan";
  /** File for WebGPU with shader-f16. */
  f16: string;
  /** File for everything else. */
  f32: string;
  /** Approximate bytes for progress reporting. */
  bytes: number;
  /** Stored in this many parts (`<file>.part0` …): files over 100 MB are refused by the host; joined on download. */
  parts?: number;
}

export const MODELS: Record<ModelSpec["id"], ModelSpec> = {
  segformer: { id: "segformer", f16: "segformer-b0-ade.fp16.onnx", f32: "segformer-b0-ade.fp32.onnx", bytes: 8e6 },
  depth: { id: "depth", f16: "depth-anything-v2-small.q4f16.onnx", f32: "depth-anything-v2-small.q4.onnx", bytes: 20e6 },
  // Fixed 256×256 input, shape logic folded (see scripts/models/swin2sr.py). fp32 on every
  // backend: an fp16 graph ran ~25% faster but produced a 2-pixel checkerboard on ORT WebGPU.
  swin2sr: { id: "swin2sr", f16: "swin2sr-lightweight-x2.onnx", f32: "swin2sr-lightweight-x2.onnx", bytes: 15.3e6 },
  // Tap-to-select (src/neural/sam.ts): MobileSAM's image encoder (TinyViT, windowed
  // attention only: ≈ 240 MB peak on the CPU, where SAM-B's global attention needs
  // ≈ 2 GB) and SAM's prompt/mask decoder. Downloaded on first use of Pick.
  samEncoder: { id: "samEncoder", f16: "mobile-sam-encoder.onnx", f32: "mobile-sam-encoder.onnx", bytes: 28.2e6 },
  samDecoder: { id: "samDecoder", f16: "sam-decoder-multi.onnx", f32: "sam-decoder-multi.onnx", bytes: 16.5e6 },
  // Magic brush (src/retouch): LaMa on computers — big-lama, fixed 512², weights stored fp16 and
  // computed fp32 (scripts/models/lama.py; a true fp16 graph overflows) — in three parts.
  lama: { id: "lama", f16: "lama.fp16.onnx", f32: "lama.fp16.onnx", bytes: 106.5e6, parts: 3 },
  // … and MI-GAN on phones: a tenth of LaMa's cost, any size (it resizes to 512 inside).
  migan: { id: "migan", f16: "migan.onnx", f32: "migan.onnx", bytes: 28.1e6 },
};

const CACHE = MODEL_CACHE;

/** A download that retrying will not fix (wrong URL, a web page instead of the model). */
class ModelError extends Error {}

export class Neural {
  readonly backend: Backend;
  readonly f16: boolean;
  /** True when ORT runs on the engine's own device (zero-copy GPU tensors). */
  readonly sharedDevice: boolean;
  private base: string;
  onProgress?: (id: string, loaded: number, total: number) => void;

  private constructor(backend: Backend, f16: boolean, shared: boolean, base: string) {
    this.backend = backend;
    this.f16 = f16;
    this.sharedDevice = shared;
    this.base = base;
  }

  static async create(gpu: Gpu | undefined, base: string, forceWasm = false): Promise<Neural> {
    ort.env.logLevel = "error";
    const iso = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
    const cores = (globalThis.navigator?.hardwareConcurrency ?? 2) | 0;
    // iOS WebKit: threaded wasm memory never shrinks and has been unstable; 2 threads at most.
    const ios = isIOS();
    ort.env.wasm.numThreads = iso ? Math.max(1, Math.min(ios ? 2 : 4, cores - 1)) : 1;
    ort.env.wasm.simd = true;
    // Production loads ORT's runtime from /ort (see scripts/copy-ort.mjs); dev uses the package directly.
    if (__ORT_EXTERNAL__) ort.env.wasm.wasmPaths = base + "ort/";
    if (gpu && !forceWasm) {
      try {
        // ORT's WebGPU backend (JSEP) requests its own device from an adapter and
        // ignores env.webgpu.device. Hand it an adapter whose requestDevice()
        // returns the engine's device, so both share one GPUDevice and tile
        // tensors never leave the GPU.
        const a = gpu.adapter as GPUAdapter & { info?: GPUAdapterInfo };
        const shim = {
          limits: a.limits,
          features: a.features,
          info: a.info,
          isFallbackAdapter: false,
          requestDevice: async () => gpu.device,
          requestAdapterInfo: async () => a.info,
        };
        ort.env.webgpu.adapter = shim as unknown as GPUAdapter;
        return new Neural("webgpu", gpu.info.f16, true, base);
      } catch (e) {
        console.warn("ORT WebGPU unavailable, falling back to WASM", e);
      }
    }
    return new Neural("wasm", false, false, base);
  }

  modelFile(spec: ModelSpec, backend: Backend = this.backend): string {
    return backend === "webgpu" && this.f16 ? spec.f16 : spec.f32;
  }

  async fetchModel(spec: ModelSpec, backend: Backend = this.backend): Promise<Uint8Array> {
    const url = this.base + "models/" + this.modelFile(spec, backend);
    // A model is binary and at least a sizeable fraction of its expected size; an
    // HTML page (a dev-server or SPA fallback answering 200) must never be cached
    // as a model — it would fail to parse on every later visit.
    const plausible = (n: number) => n >= spec.bytes * 0.3;
    let cache: Cache | undefined;
    try { cache = await caches.open(CACHE); } catch { /* private mode */ }
    const hit = await cache?.match(url);
    if (hit) {
      const b = new Uint8Array(await hit.arrayBuffer());
      if (plausible(b.byteLength)) return b;
      await cache?.delete(url);
    }
    // Mobile networks drop connections (also mid-download): the whole download is
    // retried a few times with a growing pause before giving up.
    let out: Uint8Array | undefined;
    for (let attempt = 0; !out; attempt++) {
      try {
        out = spec.parts ? await this.downloadParts(url, spec) : await this.download(url, spec);
      } catch (e) {
        const permanent = e instanceof ModelError;
        if (permanent || attempt >= 3) throw e;
        await new Promise((r) => setTimeout(r, 500 * 2 ** attempt));
      }
    }
    // The bytes themselves (no copy): a copy of a 28 MB model is exactly the kind of
    // transient peak that ends a phone's tab.
    try { await cache?.put(url, new Response(out as Uint8Array<ArrayBuffer>, { headers: { "content-type": "application/octet-stream" } })); } catch { /* quota */ }
    return out;
  }

  /** A model stored in parts: each downloaded in turn (progress over the whole), then joined. */
  private async downloadParts(url: string, spec: ModelSpec): Promise<Uint8Array> {
    const parts: Uint8Array[] = [];
    let done = 0;
    for (let i = 0; i < spec.parts!; i++) {
      const p = await this.download(`${url}.part${i}`, spec, done, 0);
      parts.push(p); done += p.byteLength;
    }
    const out = new Uint8Array(done);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.byteLength; }
    if (done < spec.bytes * 0.9) throw new Error(`Could not download ${spec.id} model (${done} bytes)`);
    return out;
  }

  /**
   * One download attempt. A wrong answer from the server (a page, a 404) is a ModelError: no retry.
   * (`before`: bytes of earlier parts, for progress over a model in parts; `min`: the least a whole file may be.)
   */
  private async download(url: string, spec: ModelSpec, before = 0, min = spec.bytes * 0.3): Promise<Uint8Array> {
    const res = await fetch(url);
    if (!res.ok || !res.body) {
      if (res.status >= 500) throw new Error(`Could not download ${spec.id} model (${res.status})`);
      throw new ModelError(`Could not download ${spec.id} model (${res.status})`);
    }
    if ((res.headers.get("content-type") ?? "").includes("text/html")) throw new ModelError(`Could not download ${spec.id} model (got a web page instead)`);
    const total = spec.parts ? spec.bytes : Number(res.headers.get("content-length")) || spec.bytes;
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      this.onProgress?.(spec.id, before + loaded, total);
    }
    const out = new Uint8Array(loaded);
    let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.byteLength; }
    if (loaded < min || loaded === 0) throw new Error(`Could not download ${spec.id} model (${loaded} bytes)`);
    return out;
  }

  /** `backend` forces WASM for one model (fallback when its WebGPU session fails). */
  async session(spec: ModelSpec, gpuOutput: boolean, backend: Backend = this.backend): Promise<Ort.InferenceSession> {
    const bytes = await this.fetchModel(spec, backend);
    const opts: Ort.InferenceSession.SessionOptions = {
      executionProviders: backend === "webgpu" ? ["webgpu"] : ["wasm"],
      graphOptimizationLevel: "all",
      enableMemPattern: backend === "wasm",
      // The CPU arena rounds every allocation up and keeps it: on a phone that is
      // the difference between fitting and the tab being killed during depth.
      enableCpuMemArena: backend !== "wasm",
    };
    if (backend === "webgpu" && gpuOutput) opts.preferredOutputLocation = "gpu-buffer";
    return ort.InferenceSession.create(bytes, opts);
  }
}

