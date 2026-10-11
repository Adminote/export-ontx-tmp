"""
Export PP-DocLayout-M inference model → ONNX
KEY: install onnx FIRST (prebuilt wheel), then paddle2onnx --no-deps
"""
import os, sys, subprocess

# Step 1: Install onnx FIRST — pip will find prebuilt wheel, no source build!
print("[1] Installing onnx + onnxruntime (prebuilt wheels)...")
r = subprocess.run(
    ["pip", "install", "onnx", "onnxruntime", "-q"],
    capture_output=True, text=True)
print(f"  onnx install rc={r.returncode}")
if r.returncode != 0:
    print(f"  STDERR: {r.stderr[-300:]}")
    sys.exit(1)

# Step 2: Install paddlepaddle (comes BEFORE paddle2onnx to satisfy deps)
print("[2] Installing paddlepaddle...")
subprocess.run(["pip", "install", "paddlepaddle", "-q"], check=True)

# Step 3: Install paddle2onnx with --no-deps (skip its broken onnx dep)
print("[3] Installing paddle2onnx (no deps)...")
r = subprocess.run(
    ["pip", "install", "paddle2onnx", "--no-deps", "-q"],
    capture_output=True, text=True)
print(f"  paddle2onnx install rc={r.returncode}")
if r.returncode != 0:
    print(f"  STDERR: {r.stderr[-300:]}")
    sys.exit(1)

import paddle2onnx
print(f"[4] paddle2onnx loaded: v{paddle2onnx.__version__}")

model_json = "PP-DocLayout-M_infer/inference.json"
params_file = "PP-DocLayout-M_infer/inference.pdiparams"

assert os.path.exists(model_json), f"Missing {model_json}"
assert os.path.exists(params_file), f"Missing {params_file}"

# paddle2onnx 0.9.2: model_filename, params_filename (NOT model_file!)
paddle2onnx.export(
    model_filename=model_json,
    params_filename=params_file,
    save_file="PP-DocLayout-M.onnx",
    opset_version=13,
)

sz = os.path.getsize("PP-DocLayout-M.onnx") / 1024 / 1024
print(f"[5] ✅ Exported: PP-DocLayout-M.onnx ({sz:.1f}MB)")

# Verify
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("PP-DocLayout-M.onnx", providers=['CPUExecutionProvider'])
print(f"[6] Inputs: {[(i.name, i.shape) for i in sess.get_inputs()]}")
print(f"[6] Outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")

out = sess.run(None, {
    "image": np.random.randn(1, 3, 640, 640).astype(np.float32),
    "scale_factor": np.array([[0.43, 0.30]], dtype=np.float32),
})
print(f"[7] ✅ Verify OK! output shapes = {[o.shape for o in out]}")
print(f"\n🎉🎉🎉 DONE! Download PP-DocLayout-M.onnx from Artifacts")
