/**
 * Magic brush: the network that fills a painted hole — a region of the photo in display
 * RGB (HWC, 0…1) and its hole (1 = fill) in, the region filled out. Each network runs in a
 * worker of its own (memory apart from the engine's, returned when the worker ends):
 *
 *   LaMa (computers; src/neural/lamaWorker.ts): ORT's native WebGPU provider, else
 *     WebAssembly; a fixed 512² input (the region is resized to it and back), the hole
 *     zeroed in the image, output 0…255. The worker ends a minute after the last stroke.
 *   MI-GAN (phones; src/neural/inpaintWorker.ts): plain WebAssembly; uint8 CHW, 0 = hole,
 *     any size (sent at most 512 px: it works at 512 inside). Ends after 30 s.
 */
import { resize } from "./geometry.ts";

export interface Inpainter {
  run(img: Float32Array, w: number, h: number, hole: Uint8Array): Promise<Float32Array>;
  dispose(): void;
}

type Reply = { type: "result" | "error" | "progress" | "log"; id?: number; result?: Float32Array | Uint8Array; error?: string; loaded?: number; total?: number; text?: string };

const N = 512;

export class WorkerInpainter implements Inpainter {
  private worker?: Worker;
  private idle = 0;
  private seq = 0;
  constructor(
    private model: "lama" | "migan",
    private base: string,
    private phone: boolean,
    private hooks: { log: (t: string) => void; progress: (loaded: number, total: number) => void },
  ) {}

  async run(img: Float32Array, w: number, h: number, hole: Uint8Array): Promise<Float32Array> {
    clearTimeout(this.idle);
    const lama = this.model === "lama";
    // The size the network is given: LaMa's fixed 512², MI-GAN's at most 512 px (it works
    // at 512 inside; a larger input only costs a phone memory).
    const k = lama ? 1 : Math.min(1, 512 / Math.max(w, h));
    const sw = lama ? N : Math.max(1, Math.round(w * k)), sh = lama ? N : Math.max(1, Math.round(h * k));
    const same = sw === w && sh === h;
    const x = same ? img : resize(img, w, h, 3, sw, sh);
    const m = same ? Float32Array.from(hole) : resize(Float32Array.from(hole), w, h, 1, sw, sh);
    const n = sw * sh;
    let msg: Record<string, unknown>;
    let transfer: Transferable[];
    if (lama) {
      const image = new Float32Array(3 * n), mask = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const hm = m[i] > 0.25 ? 1 : 0; // (a thin hole must not vanish in the resize)
        mask[i] = hm;
        for (let c = 0; c < 3; c++) image[c * n + i] = hm ? 0 : x[i * 3 + c];
      }
      msg = { image, mask }; transfer = [image.buffer, mask.buffer];
    } else {
      const image = new Uint8Array(3 * n), mask = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        mask[i] = m[i] > 0.25 ? 0 : 255;
        for (let c = 0; c < 3; c++) image[c * n + i] = Math.round(Math.min(1, Math.max(0, x[i * 3 + c])) * 255);
      }
      msg = { image, mask, w: sw, h: sh, phone: this.phone }; transfer = [image.buffer, mask.buffer];
    }
    this.worker ??= lama
      ? new Worker(new URL("../neural/lamaWorker.ts", import.meta.url), { type: "module" })
      : new Worker(new URL("../neural/inpaintWorker.ts", import.meta.url), { type: "module" });
    const wk = this.worker, id = ++this.seq;
    const out = await new Promise<Float32Array | Uint8Array>((resolve, reject) => {
      // (The first stroke downloads the model: minutes on a slow line.)
      const timer = setTimeout(() => { this.dispose(); reject(new Error("inpainting timed out")); }, 600_000);
      wk.onmessage = (ev: MessageEvent<Reply>) => {
        const r = ev.data;
        if (r.type === "log") { this.hooks.log(r.text ?? ""); return; }
        if (r.type === "progress") { this.hooks.progress(r.loaded ?? 0, r.total ?? 1); return; }
        if (r.id !== id) return;
        clearTimeout(timer);
        if (r.type === "result" && r.result) resolve(r.result); else reject(new Error(r.error ?? "inpainting failed"));
      };
      wk.onerror = (e) => { clearTimeout(timer); this.dispose(); reject(new Error(e.message || "inpainting worker failed")); };
      wk.postMessage({ type: "run", id, base: this.base, ...msg }, transfer);
    });
    this.idle = self.setTimeout(() => this.dispose(), lama ? 60_000 : 30_000);
    // (Both give 0…255: LaMa as floats, MI-GAN as bytes.)
    const hwc = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) hwc[i * 3 + c] = Math.min(1, Math.max(0, out[c * n + i] / 255));
    return same ? hwc : resize(hwc, sw, sh, 3, w, h);
  }

  dispose() { clearTimeout(this.idle); this.worker?.terminate(); this.worker = undefined; }
}
