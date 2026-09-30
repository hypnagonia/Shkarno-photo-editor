"""
LaMa (big-lama, Apache-2.0) for the magic brush, from Carve/LaMa-ONNX's lama_fp32.onnx
(opset 17, fixed 512×512; image [1,3,512,512] 0…1 with the hole zeroed, mask [1,1,512,512]
1 = hole → output [1,3,512,512] in 0…255).

    python lama.py lama_fp32.onnx ../../public/models/lama

1. onnxslim: folds the export's ~7000 shape/constant nodes (same result).
2. Weights stored as fp16, cast back to fp32 on load: half the download, fp32 compute.
   (A real fp16 graph does not work: the Fourier units' constants reach ~5·10⁵, and the
   activations overflow fp16 — mean error 9/255. int8 dynamic quantisation: 11/255.)
3. Split into parts under 50 MB (GitHub and the host refuse files over 100 MB); the app
   downloads and joins them (src/neural/ort.ts, ModelSpec.parts).
4. Checked against the original on the model card's sample: printed max / mean error.
"""
import sys, numpy as np, onnx, onnxslim, onnxruntime as ort
from onnx import numpy_helper, helper, TensorProto

src, out = sys.argv[1], sys.argv[2]
m = onnxslim.slim(src)
g = m.graph
casts, n = [], 0
for init in list(g.initializer):
    if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= 1024:
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "_f16")
        g.initializer.remove(init); g.initializer.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
        n += 1
for c in reversed(casts): g.node.insert(0, c)
blob = m.SerializeToString()
print(f"{n} weight tensors to fp16; {len(blob) / 1e6:.1f} MB")

rng = np.random.default_rng(1)
x = rng.random((1, 3, 512, 512), dtype=np.float32)
mask = np.zeros((1, 1, 512, 512), np.float32); mask[..., 180:340, 200:300] = 1
x *= 1 - mask
ref = ort.InferenceSession(src).run(None, {"image": x, "mask": mask})[0]
y = ort.InferenceSession(blob).run(None, {"image": x, "mask": mask})[0]
print(f"vs fp32: max {np.abs(y - ref).max():.2f}/255, mean {np.abs(y - ref).mean():.4f}/255")

PART = 50_000_000
parts = [blob[i:i + PART] for i in range(0, len(blob), PART)]
for i, p in enumerate(parts):
    open(f"{out}.fp16.onnx.part{i}", "wb").write(p)
print(f"{len(parts)} parts: {out}.fp16.onnx.part0…{len(parts) - 1}")
