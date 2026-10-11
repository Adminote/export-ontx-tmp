"""
Export PP-DocLayout-M inference model → ONNX
paddlepaddle 3.3.1 native APIs
"""
import os, sys, shutil
import paddle
import paddle.base.core as core

print(f"paddle version: {paddle.__version__}")

model_json = "PP-DocLayout-M_infer/inference.json"
params_file = "PP-DocLayout-M_infer/inference.pdiparams"

assert os.path.exists(model_json) and os.path.exists(params_file)

prog = core.pir.Program()
ok = core.deserialize_pir_program(model_json, prog)
print(f"[1] deserialize PIR: ok={ok}, ops={prog.num_ops()}, params={prog.parameters_num()}")

# Approach A: serialize PIR + copy params as __model__.pdiparams
fix_dir = "/tmp/pir_fix"
if os.path.exists(fix_dir): shutil.rmtree(fix_dir)
os.makedirs(fix_dir)

model_bin = os.path.join(fix_dir, "__model__")
core.serialize_pir_program(prog, model_bin, -1)
print(f"[2] serialize PIR: {os.path.getsize(model_bin)} bytes")

# KEY: jit.load expects __model__.pdiparams alongside __model__
shutil.copy(params_file, os.path.join(fix_dir, "__model__.pdiparams"))
print(f"[2] fix_dir: {os.listdir(fix_dir)}")

# Approach A1: jit.load from fix_dir (looks for __model__ + __model__.pdiparams)
model = None
try:
    model = paddle.jit.load(fix_dir)
    print(f"[3A1] jit.load(fix_dir) OK!")
except Exception as e:
    print(f"[3A1] jit.load(fix_dir) failed: {e}")

# Approach A2: explicitly specify params_filename
if model is None:
    try:
        model = paddle.jit.load(fix_dir, params_filename="__model__.pdiparams")
        print(f"[3A2] jit.load w/ params OK!")
    except Exception as e:
        print(f"[3A2] jit.load w/ params failed: {e}")

# Approach B: serialize WITH params embedded
if model is None:
    try:
        fix_dir2 = "/tmp/pir_fix2"
        if os.path.exists(fix_dir2): shutil.rmtree(fix_dir2)
        os.makedirs(fix_dir2)
        # Some paddle versions support serializing params into the program
        core.serialize_pir_program(prog, os.path.join(fix_dir2, "__model__"), -1)
        model = paddle.jit.load(fix_dir2)
        print(f"[3B] jit.load serialized OK!")
    except Exception as e:
        print(f"[3B] failed: {e}")

# Approach C: use inference.json + inference.pdiparams directly (PIR inference format)
if model is None:
    try:
        # Paddle 3.x may have native PIR inference loading
        exe = paddle.static.Executor(paddle.CPUPlace())
        [inference_program, feed_names, fetch_targets] = paddle.static.load_inference_model(
            path_prefix="PP-DocLayout-M_infer/inference",
            executor=exe)
        print(f"[3C] static.load_inference_model OK! feed={feed_names}, fetch={fetch_targets}")
        # Need to convert to jit model... this is tricky
    except Exception as e:
        print(f"[3C] static.load_inference_model failed: {e}")

if model is None:
    print("❌ ALL jit.load approaches failed")
    # Dump paddle version + available APIs for debugging
    print(f"\npaddle version: {paddle.__version__}")
    print(f"core.serialize_pir_program signature: {core.serialize_pir_program.__doc__}")
    sys.exit(1)

# Export ONNX
print(f"\n[4] paddle.onnx.export...")
paddle.onnx.export(
    model, "PP-DocLayout-M",
    input_spec=[
        paddle.static.InputSpec([1,3,640,640], 'float32', 'image'),
        paddle.static.InputSpec([1,2], 'float32', 'scale_factor'),
    ],
    opset_version=13,
)
final_onnx = "PP-DocLayout-M.onnx"
print(f"[4] ✅ {os.path.getsize(final_onnx)/1024/1024:.1f}MB")

# Verify
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession(final_onnx, providers=['CPUExecutionProvider'])
out = sess.run(None, {
    "image": np.random.randn(1,3,640,640).astype(np.float32),
    "scale_factor": np.array([[0.43, 0.30]], dtype=np.float32),
})
print(f"[5] ✅ verify OK! shapes={[o.shape for o in out]}")
print(f"\n🎉 DONE!")
