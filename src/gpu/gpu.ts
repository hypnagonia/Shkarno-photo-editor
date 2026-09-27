/**
 * WebGPU device wrapper: pipeline cache, tracked resources and small helpers.
 *
 * Every texture/buffer the engine creates goes through `tex()` / `buf()` so
 * that live GPU memory is accounted for per stage (the profiler reports it)
 * and can be released deterministically — Safari on iPhone kills the tab long
 * before a desktop browser would, so nothing may linger.
 */
import commonWgsl from "./shaders/common.wgsl?raw";

export interface GpuInfo {
  vendor: string;
  architecture: string;
  f16: boolean;
  maxTextureDimension2D: number;
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
}

type Tracked = { label: string; bytes: number; kind: "texture" | "buffer" };

const BYTES_PER_TEXEL: Partial<Record<GPUTextureFormat, number>> = {
  "rgba8unorm": 4, "rgba16float": 8, "rgba32float": 16, "r16uint": 2, "rgba16uint": 8, "r32float": 4,
  "rg32float": 8, "r16float": 2, "rgba8uint": 4, "r32uint": 4, "bgra8unorm": 4, "rg16float": 4,
};

export class Gpu {
  readonly device: GPUDevice;
  readonly adapter: GPUAdapter;
  readonly info: GpuInfo;
  private pipelines = new Map<string, GPUComputePipeline>();
  private live = new Map<GPUTexture | GPUBuffer, Tracked>();
  peakBytes = 0;
  lost = false;
  /** Receives validation/out-of-memory errors (the engine forwards them to the log). */
  onError?: (message: string) => void;
  /** The device was lost (driver reset, memory pressure); the engine must restart. */
  onLost?: (message: string) => void;
  private errorsSeen = new Set<string>();

  private constructor(adapter: GPUAdapter, device: GPUDevice, info: GpuInfo) {
    this.adapter = adapter;
    this.device = device;
    this.info = info;
    device.addEventListener("uncapturederror", (ev) => {
      const msg = (ev as GPUUncapturedErrorEvent).error.message;
      if (this.errorsSeen.has(msg)) return; // one report per distinct error
      this.errorsSeen.add(msg);
      console.error("WebGPU:", msg);
      this.onError?.(msg);
    });
    device.lost.then((e) => {
      this.lost = true;
      console.error("GPU device lost", e.reason, e.message);
      if (e.reason !== "destroyed") this.onLost?.(e.message);
    });
  }

  static async create(): Promise<Gpu | undefined> {
    const nav = (globalThis as { navigator?: Navigator }).navigator;
    if (!nav?.gpu) return undefined;
    const adapter = await nav.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return undefined;
    let f16 = adapter.features.has("shader-f16");
    const want = (k: keyof GPUSupportedLimits) => adapter.limits[k] as number;
    const limits = {
      maxBufferSize: want("maxBufferSize"),
      maxStorageBufferBindingSize: want("maxStorageBufferBindingSize"),
      maxTextureDimension2D: want("maxTextureDimension2D"),
      maxComputeWorkgroupStorageSize: want("maxComputeWorkgroupStorageSize"),
      maxComputeInvocationsPerWorkgroup: want("maxComputeInvocationsPerWorkgroup"),
      maxStorageTexturesPerShaderStage: want("maxStorageTexturesPerShaderStage"),
      maxStorageBuffersPerShaderStage: want("maxStorageBuffersPerShaderStage"),
    };
    // The adapter's own maxima first; some devices / browser versions refuse a
    // request that is technically within them, so fall back step by step to
    // the defaults (everything the pipeline needs fits the WebGPU defaults on
    // a phone-sized working image) and finally without shader-f16.
    const attempts: GPUDeviceDescriptor[] = [
      { requiredFeatures: f16 ? ["shader-f16"] : [], requiredLimits: limits },
      { requiredFeatures: f16 ? ["shader-f16"] : [], requiredLimits: { maxBufferSize: limits.maxBufferSize, maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize, maxTextureDimension2D: limits.maxTextureDimension2D } },
      { requiredFeatures: f16 ? ["shader-f16"] : [] },
      {},
    ];
    let device: GPUDevice | undefined;
    let lastErr: unknown;
    for (const [i, d] of attempts.entries()) {
      try {
        device = await adapter.requestDevice(d);
        if (i > 0) console.warn(`WebGPU device created with fallback settings #${i}`, lastErr);
        if (i === attempts.length - 1) f16 = false;
        break;
      } catch (e) { lastErr = e; }
    }
    if (!device) throw new Error(`The GPU could not be set up on this device: ${lastErr instanceof Error ? lastErr.message : String(lastErr)}`);
    const ai = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
    return new Gpu(adapter, device, {
      vendor: ai?.vendor ?? "unknown",
      architecture: ai?.architecture ?? "unknown",
      f16,
      maxTextureDimension2D: device.limits.maxTextureDimension2D,
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    });
  }

  // -------------------------------------------------------------------------
  // Resources

  tex(label: string, width: number, height: number, format: GPUTextureFormat, usage?: number, dimension: GPUTextureDimension = "2d", depth = 1): GPUTexture {
    const u = usage ?? (GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST);
    const t = this.device.createTexture({ label, size: { width, height, depthOrArrayLayers: depth }, format, usage: u, dimension });
    this.track(t, { label, kind: "texture", bytes: width * height * depth * (BYTES_PER_TEXEL[format] ?? 8) });
    return t;
  }

  buf(label: string, size: number, usage: number): GPUBuffer {
    const b = this.device.createBuffer({ label, size: Math.max(16, Math.ceil(size / 16) * 16), usage });
    this.track(b, { label, kind: "buffer", bytes: b.size });
    return b;
  }

  /**
   * Released uniform buffers, by size, for reuse: a render makes 5–8 of them. Reuse
   * right after the work using one was submitted is safe — queue writes and
   * submits run in order, so that work still reads the old contents.
   */
  private uniformPool = new Map<number, GPUBuffer[]>();
  private uniforms = new WeakSet<GPUBuffer>();
  uniform(data: ArrayBufferView | ArrayBuffer, label = "uniforms"): GPUBuffer {
    const bytes = data instanceof ArrayBuffer ? data.byteLength : data.byteLength;
    const size = Math.max(16, Math.ceil(bytes / 16) * 16);
    let b = this.uniformPool.get(size)?.pop();
    if (!b) {
      b = this.device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.uniforms.add(b);
    }
    this.device.queue.writeBuffer(b, 0, data as BufferSource);
    // Uniforms are tiny and short-lived: handed back by the caller via `release`.
    this.track(b, { label, kind: "buffer", bytes: b.size });
    return b;
  }

  private track(r: GPUTexture | GPUBuffer, t: Tracked) {
    this.live.set(r, t);
    const total = this.liveBytes();
    if (total > this.peakBytes) this.peakBytes = total;
    if (total > this.stepPeak) this.stepPeak = total;
  }

  /** Peak since the last takeStepPeak(), independent of the profiler's per-stage resets (the memory guard's steps). */
  private stepPeak = 0;
  takeStepPeak(): number {
    const live = this.liveBytes();
    const p = Math.max(this.stepPeak, live);
    this.stepPeak = live;
    return p;
  }

  release(...rs: Array<GPUTexture | GPUBuffer | undefined | null>) {
    for (const r of rs) {
      if (!r) continue;
      if (!this.live.delete(r)) continue;
      if (r instanceof GPUBuffer && this.uniforms.has(r)) {
        const pool = this.uniformPool.get(r.size) ?? [];
        if (pool.length < 32) { pool.push(r); this.uniformPool.set(r.size, pool); continue; }
      }
      r.destroy();
    }
  }

  liveBytes(): number {
    let s = 0;
    for (const t of this.live.values()) s += t.bytes;
    return s;
  }

  liveReport(): Array<{ label: string; mb: number }> {
    return [...this.live.values()].filter((t) => t.bytes > 1 << 20).map((t) => ({ label: t.label, mb: +(t.bytes / 1048576).toFixed(1) }));
  }

  resetPeak() {
    this.peakBytes = this.liveBytes();
  }

  // -------------------------------------------------------------------------
  // Pipelines

  pipeline(name: string, code: string, entryPoint = "main", constants?: Record<string, number>): GPUComputePipeline {
    const key = name + "|" + entryPoint + "|" + JSON.stringify(constants ?? {});
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ label: name, code: commonWgsl + "\n" + code });
      // Compile errors are otherwise silent (the pass just renders nothing): report them by shader.
      void module.getCompilationInfo?.().then((info) => {
        const errs = info.messages.filter((m) => m.type === "error");
        if (errs.length) this.onError?.(`shader "${name}": ` + errs.map((m) => `line ${m.lineNum}: ${m.message}`).join("; "));
      }).catch(() => {});
      p = this.device.createComputePipeline({ label: name, layout: "auto", compute: { module, entryPoint, constants } });
      this.pipelines.set(key, p);
    }
    return p;
  }

  /** Records one compute dispatch. `bindings[i]` is bound to @group(0) @binding(i);
   * holes (undefined) are skipped, for entry points that use a subset. */
  dispatch(
    enc: GPUCommandEncoder,
    pipe: GPUComputePipeline,
    bindings: Array<GPUBuffer | GPUTextureView | GPUSampler | undefined>,
    x: number, y = 1, z = 1,
  ) {
    const entries: GPUBindGroupEntry[] = [];
    bindings.forEach((b, i) => {
      if (b) entries.push({ binding: i, resource: b instanceof GPUBuffer ? { buffer: b } : b });
    });
    const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    const pass = enc.beginComputePass({ label: pipe.label });
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(x)), Math.max(1, Math.ceil(y)), Math.max(1, Math.ceil(z)));
    pass.end();
  }

  /** Runs `record` in a fresh encoder, submits, and releases per-call uniforms. */
  async run(label: string, record: (enc: GPUCommandEncoder, temp: Array<GPUBuffer | GPUTexture>) => void, wait = false) {
    const enc = this.device.createCommandEncoder({ label });
    const temp: Array<GPUBuffer | GPUTexture> = [];
    record(enc, temp);
    this.device.queue.submit([enc.finish()]);
    if (wait) await this.device.queue.onSubmittedWorkDone();
    // Destroying after submit is valid: WebGPU defers destruction until the queue is done with it.
    this.release(...temp);
  }

  // Readback staging buffers are pooled: previews and export strips read back
  // the same sizes over and over.
  private stagingPool: GPUBuffer[] = [];
  /** Most memory kept in pooled readback buffers (phones: less). */
  stagingLimitMB = 48;
  /** Drops every pooled readback buffer (after an export: its strip sizes will not come back). */
  flushStaging() { for (const b of this.stagingPool.splice(0)) b.destroy(); }
  private takeStaging(size: number): GPUBuffer {
    const i = this.stagingPool.findIndex((b) => b.size >= size && b.size <= size * 2);
    if (i >= 0) return this.stagingPool.splice(i, 1)[0];
    return this.device.createBuffer({ label: "staging", size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }
  private giveStaging(b: GPUBuffer) {
    this.stagingPool.push(b);
    let total = this.stagingPool.reduce((a, x) => a + x.size, 0);
    while (total > this.stagingLimitMB * 1048576 && this.stagingPool.length) {
      const old = this.stagingPool.shift()!;
      total -= old.size;
      old.destroy();
    }
  }

  async readBuffer(src: GPUBuffer, size: number, offset = 0): Promise<ArrayBuffer> {
    const staging = this.device.createBuffer({ size: Math.ceil(size / 4) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, offset, staging, 0, Math.ceil(size / 4) * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = staging.getMappedRange().slice(0, size);
    staging.unmap();
    staging.destroy();
    return out;
  }

  /** Reads a region of a texture. `bpp` bytes per texel. Rows are unpadded in the result. */
  async readTexture(t: GPUTexture, x: number, y: number, w: number, h: number, bpp: number): Promise<ArrayBuffer> {
    const rowBytes = w * bpp;
    const padded = Math.ceil(rowBytes / 256) * 256;
    const staging = this.takeStaging(padded * h);
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: t, origin: { x, y } }, { buffer: staging, bytesPerRow: padded, rowsPerImage: h }, { width: w, height: h });
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(staging.getMappedRange());
    const out = new Uint8Array(rowBytes * h);
    for (let r = 0; r < h; r++) out.set(src.subarray(r * padded, r * padded + rowBytes), r * rowBytes);
    staging.unmap();
    this.giveStaging(staging);
    return out.buffer;
  }

  destroy() {
    for (const r of this.live.keys()) r.destroy();
    this.live.clear();
    this.pipelines.clear();
  }
}

/** Packs mixed numeric fields into a Float32/Uint32 uniform block (all 4-byte scalars). */
export class Uniforms {
  private f: Float32Array;
  private u: Uint32Array;
  private i: Int32Array;
  private n = 0;
  constructor(capacity = 64) {
    const b = new ArrayBuffer(capacity * 4);
    this.f = new Float32Array(b);
    this.u = new Uint32Array(b);
    this.i = new Int32Array(b);
  }
  f32(...v: number[]) { for (const x of v) this.f[this.n++] = x; return this; }
  u32(...v: number[]) { for (const x of v) this.u[this.n++] = x >>> 0; return this; }
  i32(...v: number[]) { for (const x of v) this.i[this.n++] = x | 0; return this; }
  /** Aligns to a multiple of `words` 4-byte words (vec4 = 4, vec3 = 4, vec2 = 2). */
  align(words: number) { while (this.n % words) this.n++; return this; }
  mat3(m: ArrayLike<number>) {
    this.align(4);
    for (let c = 0; c < 3; c++) { for (let r = 0; r < 3; r++) this.f[this.n++] = m[r * 3 + c]; this.n++; }
    return this;
  }
  bytes(): ArrayBuffer {
    const words = Math.ceil(this.n / 4) * 4;
    return this.f.buffer.slice(0, Math.max(16, words * 4)) as ArrayBuffer;
  }
}

export const wg = (n: number, size: number) => Math.ceil(n / size);
