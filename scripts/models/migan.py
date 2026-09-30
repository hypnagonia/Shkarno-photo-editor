"""
MI-GAN (Picsart AI Research, MIT) for the magic brush on phones: the "pipeline" export
(andraniksargsyan/migan, mirrored at edgetools/migan) — image uint8 [1,3,H,W] and mask
uint8 [1,1,H,W] with 0 = erase, 255 = keep → result uint8 [1,3,H,W], composited, any size
(the graph resizes to 512 internally). Copied unchanged; checked here on a sample.

    python migan.py migan_pipeline_v2.onnx ../../public/models/migan.onnx
"""
import sys, shutil, numpy as np, onnxruntime as ort

src, out = sys.argv[1], sys.argv[2]
s = ort.InferenceSession(src)
img = (np.random.default_rng(1).random((1, 3, 600, 800)) * 255).astype(np.uint8)
mask = np.full((1, 1, 600, 800), 255, np.uint8); mask[..., 200:400, 300:500] = 0
y = s.run(None, {"image": img, "mask": mask})[0]
keep = mask[0, 0] == 255
print("out", y.shape, y.dtype, "max change outside the hole:", int(np.abs(y[0][:, keep].astype(int) - img[0][:, keep]).max()), "(the app composites only inside the mask)")
shutil.copyfile(src, out)
