/// <reference lib="webworker" />
/**
 * Magic brush on phones: MI-GAN on the plain WebAssembly runtime, in a worker of its own
 * (see src/retouch/inpaint.ts). "run": image uint8 CHW + mask (0 = hole) → the filled
 * image, uint8 CHW, same size. The session is kept for the worker's life (it ends 30 s
 * after the last stroke).
 */
import { Neural, MODELS, ort, useCpuRuntime } from "./ort.ts";
import type * as Ort from "onnxruntime-web";
import { forcePhone } from "../device.ts";

useCpuRuntime();

type In = { type: "run"; id: number; base: string; phone?: boolean; image: Uint8Array; mask: Uint8Array; w: number; h: number };

const post = (m: unknown, transfer: Transferable[] = []) => (self as unknown as DedicatedWorkerGlobalScope).postMessage(m, transfer);

let neural: Neural | undefined;
let ses: Ort.InferenceSession | undefined;

self.onmessage = async (ev: MessageEvent<In>) => {
  const m = ev.data;
  if (m.phone) forcePhone(true);
  try {
    if (!neural) {
      neural = await Neural.create(undefined, m.base, true);
      neural.onProgress = (_id, loaded, total) => post({ type: "progress", loaded, total });
    }
    ses ??= await neural.session(MODELS.migan, false, "wasm");
    const tI = new ort.Tensor("uint8", m.image, [1, 3, m.h, m.w]), tM = new ort.Tensor("uint8", m.mask, [1, 1, m.h, m.w]);
    const res = await ses.run({ image: tI, mask: tM });
    const out = Uint8Array.from((await res[ses.outputNames[0]].getData()) as Uint8Array);
    for (const t of Object.values(res)) t.dispose();
    tI.dispose(); tM.dispose();
    post({ type: "result", id: m.id, result: out }, [out.buffer]);
  } catch (e) {
    post({ type: "error", id: m.id, error: e instanceof Error ? e.message : String(e) });
  }
};
