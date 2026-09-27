/// <reference lib="webworker" />
/**
 * Tap-to-select (MobileSAM) on the CPU, in one worker (see sam.ts).
 *
 * One ONNX Runtime for both models: a runtime costs ≈ 200 MB before it runs
 * anything, and an encoder worker and a decoder worker side by side (plus the
 * first one's memory not yet returned) peaked near 2 GB on a phone path. Here the
 * encoder runs, is released, and the small decoder reuses the same memory.
 *
 * "encode": the photo (HWC, 0…255, long side 1024) → image embeddings (kept here,
 *   and sent back so the engine can restart this worker without encoding again).
 * "init":   embeddings from the engine (a restarted worker).
 * "decode": taps → SAM's four 256² masks and their predicted quality.
 */
import { Neural, MODELS, ort, useCpuRuntime } from "./ort.ts";
import type * as Ort from "onnxruntime-web";
import { forcePhone } from "../device.ts";

// CPU only here: the plain WebAssembly runtime (see ort.ts), before anything loads it.
useCpuRuntime();

type In =
  | { type: "encode"; base: string; phone?: boolean; image: Float32Array; w: number; h: number }
  | { type: "init"; base: string; phone?: boolean; emb: Float32Array }
  | { type: "decode"; id: number; coords: Float32Array; labels: Float32Array; w: number; h: number };

const post = (m: unknown, transfer: Transferable[] = []) => (self as unknown as DedicatedWorkerGlobalScope).postMessage(m, transfer);

let neural: Neural | undefined;
let decoder: Ort.InferenceSession | undefined;
let emb: Ort.Tensor | undefined;

self.onmessage = async (ev: MessageEvent<In>) => {
  const m = ev.data;
  if ("phone" in m && m.phone) forcePhone(true);
  try {
    if (m.type === "encode" || m.type === "init") neural ??= await Neural.create(undefined, m.base, true);
    if (m.type === "encode") {
      const s = await neural!.session(MODELS.samEncoder, false, "wasm");
      const out = await s.run({ input_image: new ort.Tensor("float32", m.image, [m.h, m.w, 3]) });
      const e = Float32Array.from((await out[s.outputNames[0]].getData()) as Float32Array);
      for (const t of Object.values(out)) t.dispose();
      await s.release();
      emb = new ort.Tensor("float32", e, [1, 256, 64, 64]);
      post({ type: "embedding", emb: e.slice() });
    } else if (m.type === "init") {
      emb = new ort.Tensor("float32", m.emb, [1, 256, 64, 64]);
      post({ type: "ready" });
    } else if (m.type === "decode") {
      if (!emb || !neural) throw new Error("selection not encoded");
      decoder ??= await neural.session(MODELS.samDecoder, false, "wasm");
      const n = m.labels.length;
      const out = await decoder.run({
        image_embeddings: emb,
        point_coords: new ort.Tensor("float32", m.coords, [1, n, 2]),
        point_labels: new ort.Tensor("float32", m.labels, [1, n]),
        mask_input: new ort.Tensor("float32", new Float32Array(256 * 256), [1, 1, 256, 256]),
        has_mask_input: new ort.Tensor("float32", new Float32Array([0]), [1]),
        orig_im_size: new ort.Tensor("float32", new Float32Array([m.h, m.w]), [2]),
      });
      const low = Float32Array.from((await out.low_res_masks.getData()) as Float32Array);
      const iou = Float32Array.from((await out.iou_predictions.getData()) as Float32Array);
      for (const t of Object.values(out)) t.dispose();
      post({ type: "masks", id: m.id, low, iou }, [low.buffer, iou.buffer]);
    }
  } catch (e) {
    post({ type: "error", id: (m as { id?: number }).id, error: e instanceof Error ? e.message : String(e) });
  }
};
