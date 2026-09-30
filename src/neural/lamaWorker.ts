/// <reference lib="webworker" />
/**
 * Magic brush on computers: LaMa on ORT's native WebGPU provider (see useNativeWebgpuRuntime
 * in ort.ts), in a worker of its own; WebAssembly when WebGPU is not there or fails.
 * "run": image float32 [1,3,512,512] 0…1 with the hole zeroed, mask [1,1,512,512] 1 = hole
 * → the filled image, 0…255. The session is kept for the worker's life (it ends a minute
 * after the last stroke). Download progress is reported ("progress").
 */
import { Neural, MODELS, ort, useNativeWebgpuRuntime, type Backend } from "./ort.ts";
import type * as Ort from "onnxruntime-web";

useNativeWebgpuRuntime();

type In = { type: "run"; id: number; base: string; image: Float32Array; mask: Float32Array };

const post = (m: unknown, transfer: Transferable[] = []) => (self as unknown as DedicatedWorkerGlobalScope).postMessage(m, transfer);
const N = 512;

let neural: Neural | undefined;
let ses: Ort.InferenceSession | undefined;
let backend: Backend | undefined;

async function open(): Promise<Ort.InferenceSession> {
  const order: Backend[] = backend ? [backend] : "gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"];
  let err: unknown;
  for (const b of order) {
    try { const s = await neural!.session(MODELS.lama, false, b); backend = b; post({ type: "log", text: `magic brush: LaMa on ${b}` }); return s; }
    catch (e) { err = e; post({ type: "log", text: `magic brush: LaMa on ${b} unavailable (${e instanceof Error ? e.message : e})` }); }
  }
  throw err instanceof Error ? err : new Error("LaMa unavailable");
}

async function infer(image: Float32Array, mask: Float32Array): Promise<Float32Array> {
  ses ??= await open();
  const tI = new ort.Tensor("float32", image, [1, 3, N, N]), tM = new ort.Tensor("float32", mask, [1, 1, N, N]);
  try {
    const res = await ses.run({ image: tI, mask: tM });
    const y = Float32Array.from((await res[ses.outputNames[0]].getData()) as Float32Array);
    for (const t of Object.values(res)) t.dispose();
    return y;
  } finally { tI.dispose(); tM.dispose(); }
}

function sane(y: Float32Array): boolean {
  let sum = 0;
  for (let i = 0; i < y.length; i += 97) { const v = y[i]; if (!Number.isFinite(v) || v < -20 || v > 300) return false; sum += v; }
  return sum > 0;
}

self.onmessage = async (ev: MessageEvent<In>) => {
  const m = ev.data;
  try {
    if (!neural) {
      neural = await Neural.create(undefined, m.base, true);
      neural.onProgress = (_id, loaded, total) => post({ type: "progress", loaded, total });
    }
    let out: Float32Array | undefined;
    try { out = await infer(m.image, m.mask); } catch (e) { if (backend !== "webgpu") throw e; post({ type: "log", text: `magic brush: LaMa on WebGPU failed (${e instanceof Error ? e.message.slice(0, 160) : e}), using the CPU` }); }
    // A WebGPU run that fails or returns nonsense: the CPU from now on.
    if (backend === "webgpu" && (!out || !sane(out))) {
      await ses?.release().catch(() => {}); ses = undefined; backend = "wasm";
      out = await infer(m.image, m.mask);
    }
    if (!out || !sane(out)) throw new Error("inpainting returned an invalid image");
    post({ type: "result", id: m.id, result: out }, [out.buffer]);
  } catch (e) {
    post({ type: "error", id: m.id, error: e instanceof Error ? e.message : String(e) });
  }
};
