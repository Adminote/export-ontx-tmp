"""
Export PP-DocLayout-M inference model → ONNX
Uses paddlepaddle 3.0 native PIR APIs (no paddle2onnx)
"""
import os, sys, shutil
import paddle
import paddle.base.core as core

print(f"paddle version: {paddle.__version__}")

# === Step 1: Deserialize PIR Program from inference.json ===
model_json = "PP-DocLayout-M_infer/inference.json"
params_file = "PP-DocLayout-M_infer/inference.pdiparams"

assert os.path.exists(model_json), f"Missing {model_json}"
assert os.path.exists(params_file), f"Missing {params_file}"

prog = core.pir.Program()
ok = core.deserialize_pir_program(model_json, prog)
print(f"[1] deserialize PIR: ok={ok}, ops={prog.num_ops()}, params={prog.parameters_num()}")

# === Step 2: Serialize to __model__ binary protobuf ===
fix_dir = "/tmp/pir_fix"
if os.path.exists(fix_dir):
    shutil.rmtree(fix_dir)
os.makedirs(fix_dir)

model_bin = os.path.join(fix_dir, "__model__")
try:
    core.serialize_pir_program(prog, model_bin, -1)
    sz = os.path.getsize(model_bin) if os.path.exists(model_bin) else 0
    print(f"[2] serialize PIR → __model__: {sz} bytes")
except Exception as e:
    print(f"[2] serialize failed: {e}")

if not os.path.exists(model_bin) or os.path.getsize(model_bin) == 0:
    print("  Falling back: use JSON as model file for jit.load")
    shutil.copy(model_json, os.path.join(fix_dir, "__model__.json"))

shutil.copy(params_file, os.path.join(fix_dir, "inference.pdiparams"))
print(f"[2] fix_dir contents: {os.listdir(fix_dir)}")

# === Step 3: Try paddle.jit.load ===
model = None
for candidate_name in ["__model__", "__model__.json", "inference.json"]:
    candidate_path = os.path.join(fix_dir if candidate_name.startswith("__") else ".", candidate_name)
    if os.path.exists(candidate_path):
        try:
            load_dir = fix_dir if candidate_name.startswith("__") else "."
            model = paddle.jit.load(load_dir)
            print(f"[3] jit.load OK from {load_dir}")
            break
        except Exception as e:
            print(f"[3] jit.load from {candidate_path} failed: {e}")

if model is None:
    print("❌ jit.load failed completely")
    sys.exit(1)

# === Step 4: paddle.onnx.export ===
onnx_path = "PP-DocLayout-M"
try:
    paddle.onnx.export(
        model,
        onnx_path,
        input_spec=[
            paddle.static.InputSpec([1, 3, 640, 640], 'float32', 'image'),
            paddle.static.InputSpec([1, 2], 'float32', 'scale_factor'),
        ],
        opset_version=13,
        enable_onnx_checker=True,
    )
    final_onnx = onnx_path + ".onnx"
    sz = os.path.getsize(final_onnx) / 1024 / 1024
    print(f"[4] ✅ paddle.onnx.export → {final_onnx} ({sz:.1f}MB)")
except Exception as e:
    print(f"[4] paddle.onnx.export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# === Step 5: Verify with onnxruntime ===
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession(final_onnx, providers=['CPUExecutionProvider'])
out = sess.run(None, {
    "image": np.random.randn(1, 3, 640, 640).astype(np.float32),
    "scale_factor": np.array([[0.43, 0.30]], dtype=np.float32),
})
print(f"[5] ✅ onnxruntime verify OK! output shapes = {[o.shape for o in out]}")
print(f"\n🎉🎉🎉 DONE! Download {final_onnx}")
