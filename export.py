"""
Export PP-DocLayout-M inference model → ONNX
Using paddle2onnx 0.9.2 with CORRECT parameter names (model_filename, params_filename)
"""
import os, sys

print("Installing paddle2onnx...")
# Install paddle2onnx inside the script to ensure Linux wheel
import subprocess
subprocess.run(["pip", "install", "paddle2onnx==0.9.2", "-q"], check=True)

import paddle2onnx

model_json = "PP-DocLayout-M_infer/inference.json"
params_file = "PP-DocLayout-M_infer/inference.pdiparams"

assert os.path.exists(model_json), f"Missing {model_json}"
assert os.path.exists(params_file), f"Missing {params_file}"

print(f"paddle2onnx version: {paddle2onnx.__version__}")
print(f"Exporting: {model_json} + {params_file}")

# paddle2onnx 0.9.2 signature: export(model_filename, params_filename, save_file, ...)
# NO model_format param in 0.9.2 — it auto-detects
paddle2onnx.export(
    model_filename=model_json,
    params_filename=params_file,
    save_file="PP-DocLayout-M.onnx",
    opset_version=13,
)

sz = os.path.getsize("PP-DocLayout-M.onnx") / 1024 / 1024
print(f"[1] ✅ Exported: PP-DocLayout-M.onnx ({sz:.1f}MB)")

# Verify
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("PP-DocLayout-M.onnx", providers=['CPUExecutionProvider'])
print(f"[2] Inputs: {[(i.name, i.shape) for i in sess.get_inputs()]}")
print(f"[2] Outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")

out = sess.run(None, {
    "image": np.random.randn(1, 3, 640, 640).astype(np.float32),
    "scale_factor": np.array([[0.43, 0.30]], dtype=np.float32),
})
print(f"[3] ✅ Verify OK! output shapes = {[o.shape for o in out]}")
print(f"\n🎉🎉🎉 DONE! Download PP-DocLayout-M.onnx")
