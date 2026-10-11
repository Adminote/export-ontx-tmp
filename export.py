import paddle2onnx, os, onnxruntime as ort, numpy as np

paddle2onnx.export(
    model_file="PP-DocLayout-M_infer/inference.json",
    params_file="PP-DocLayout-M_infer/inference.pdiparams",
    save_file="PP-DocLayout-M_infer.onnx",
    opset_version=13, model_format="inference")

sz = os.path.getsize("PP-DocLayout-M_infer.onnx")
print(f"Exported: {sz/1024/1024:.1f}MB")

sess = ort.InferenceSession("PP-DocLayout-M_infer.onnx", providers=["CPUExecutionProvider"])
out = sess.run(None, {
    "image": np.random.randn(1,3,640,640).astype(np.float32),
    "scale_factor": np.array([[0.43, 0.30]], dtype=np.float32)})
print(f"OK! shapes={[o.shape for o in out]}")
