/**
 * Tap-to-select: MobileSAM (Segment Anything with a TinyViT encoder), for masks
 * of one thing — one person of four, one person's clothes — which the scene
 * analysis (a region per kind of thing) cannot tell apart.
 *
 * The photo is encoded once, in a worker that is terminated right after (see
 * samWorker.ts); taps are decoded in a second small worker that lives until the
 * photo is closed. A selection is its taps (points with add / remove labels),
 * so params stay small and the mask can be rebuilt after a reload.
 */
import { samDims } from "../refine/selection.ts";

import type { SelectPoint } from "../layers/model.ts";
export type { SelectPoint };

type Reply =
  | { type: "embedding"; emb: Float32Array }
  | { type: "ready" }
  | { type: "masks"; id: number; low: Float32Array; iou: Float32Array }
  | { type: "error"; id?: number; error: string };

function spawn(): Worker {
  return new Worker(new URL("./samWorker.ts", import.meta.url), { type: "module" });
}

/** Longest wait for each answer (model download on a slow phone included); past it the worker is stuck. */
const TIMEOUT = { embedding: 180_000, ready: 60_000, masks: 20_000 } as const;

/**
 * One request/answer with a worker (the first message of the matching type, or an
 * error). No answer in time (a worker killed or hung by the OS) terminates the
 * worker and rejects, so a tap never waits forever.
 */
function ask<T extends Reply["type"]>(w: Worker, msg: unknown, want: T, id?: number, transfer: Transferable[] = []): Promise<Extract<Reply, { type: T }>> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { done(); w.terminate(); reject(new Error("Selection timed out")); }, TIMEOUT[want as keyof typeof TIMEOUT] ?? 60_000);
    const onMsg = (ev: MessageEvent<Reply>) => {
      const r = ev.data;
      if (r.type === "error" && (id === undefined || r.id === id || r.id === undefined)) { done(); reject(new Error(r.error)); }
      else if (r.type === want && (id === undefined || (r as { id?: number }).id === id)) { done(); resolve(r as Extract<Reply, { type: T }>); }
    };
    const onErr = (e: ErrorEvent) => { e.preventDefault(); done(); reject(new Error(e.message || "Selection stopped")); };
    const done = () => { clearTimeout(timer); w.removeEventListener("message", onMsg); w.removeEventListener("error", onErr); };
    w.addEventListener("message", onMsg);
    w.addEventListener("error", onErr);
    w.postMessage(msg, transfer);
  });
}

export class SamSelector {
  /** The photo's size inside SAM's 1024 square. */
  readonly dims: [number, number];
  private emb?: Float32Array;
  private encoding?: Promise<void>;
  /** The one worker for both models (samWorker.ts); closed after a while without a tap. */
  private worker?: Worker;
  /** The worker has the embeddings (it encoded them, or was given them). */
  private workerReady?: Promise<unknown>;
  private nextId = 1;
  /** The worker is closed after this long without a tap (its memory back; reopening takes ≈ 1 s). */
  private idle = 0;

  constructor(private base: string, photoW: number, photoH: number, private phone = false) {
    this.dims = samDims(photoW, photoH);
  }

  get encoded(): boolean { return !!this.emb; }

  /** Encodes the photo once (`image`: HWC 0…255 at `dims`); later calls wait for the same run. */
  encode(image: () => Promise<Float32Array>): Promise<void> {
    this.encoding ??= (async () => {
      const px = await image();
      this.worker ??= spawn();
      const [pw, ph] = this.dims;
      const run = ask(this.worker, { type: "encode", base: this.base, phone: this.phone, image: px, w: pw, h: ph }, "embedding", undefined, [px.buffer]);
      this.workerReady = run;
      this.emb = (await run).emb;
    })().catch((e) => { this.encoding = undefined; this.dispose(); throw e; });
    return this.encoding;
  }

  /** SAM's four 256² logit masks and their predicted quality for these taps. */
  async decode(points: SelectPoint[]): Promise<{ low: Float32Array; iou: Float32Array }> {
    if (!this.emb) throw new Error("photo not encoded for selection");
    if (!this.worker) {
      // Closed while idle: a new worker gets the embeddings (no second encode).
      this.worker = spawn();
      // A failed start drops this worker, so the next tap starts a fresh one (not the same rejection forever).
      this.workerReady = ask(this.worker, { type: "init", base: this.base, phone: this.phone, emb: this.emb.slice() }, "ready")
        .catch((e) => { this.dispose(); throw e; });
    }
    await this.workerReady;
    if (!this.worker) throw new Error("Selection stopped");
    const [pw, ph] = this.dims;
    // SAM's point prompt: coordinates in the 1024 square, plus a padding point (label −1) when there is no box.
    const coords = new Float32Array((points.length + 1) * 2);
    const labels = new Float32Array(points.length + 1);
    points.forEach(([x, y, l], i) => { coords[i * 2] = x * pw; coords[i * 2 + 1] = y * ph; labels[i] = l; });
    labels[points.length] = -1;
    const id = this.nextId++;
    clearTimeout(this.idle);
    try {
      const r = await ask(this.worker!, { type: "decode", id, coords, labels, w: pw, h: ph }, "masks", id);
      return { low: r.low, iou: r.iou };
    } catch (e) {
      this.dispose(); // a worker that failed a decode (or was killed) is not reused
      throw e;
    } finally {
      this.idle = setTimeout(() => this.dispose(), 30_000) as unknown as number;
    }
  }

  dispose() {
    clearTimeout(this.idle);
    this.worker?.terminate();
    this.worker = undefined;
    this.workerReady = undefined;
  }
}
