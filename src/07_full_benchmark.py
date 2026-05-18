import tensorrt as trt
import numpy as np
import time
import os
import pycuda.autoinit
import pycuda.driver as cuda

def benchmark_engine(engine_path, precision_name, num_runs=100, warmup=20):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)

    input_size = int(np.prod(input_shape) * 4)
    output_size = int(np.prod(output_shape) * 4)
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)

    h_input = np.random.randn(*input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)
    stream = cuda.Stream()

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    times = []
    for _ in range(num_runs):
        t0 = time.time()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        times.append((time.time() - t0) * 1000)

    avg = np.mean(times)
    std = np.std(times)
    fps = 1000 / avg
    print(f"  {precision_name:<10} 延迟: {avg:.2f} ms (±{std:.2f})  FPS: {fps:.1f}")
    return avg, fps

if __name__ == "__main__":
    print("=" * 60)
    print("  YOLOv8n TensorRT 全精度 Benchmark (RTX 4060 8GB)")
    print("=" * 60)

    results = {}
    for prec in ["fp32", "fp16", "int8"]:
        path = f"/workspace/engines/yolov8n_{prec}.engine"
        if os.path.exists(path):
            avg, fps = benchmark_engine(path, prec.upper())
            size = os.path.getsize(path) / 1024 / 1024
            results[prec] = {"latency": avg, "fps": fps, "size": size}

    print("\n" + "=" * 60)
    print("  完整性能对比表")
    print("=" * 60)
    print(f"  {'精度':<10} {'延迟(ms)':<12} {'FPS':<10} {'文件(MB)':<10} {'vs PyTorch'}")
    print(f"  {'-'*55}")
    print(f"  {'PyTorch':<10} {'6.50':<12} {'153.4':<10} {'12.3':<10} {'基线'}")
    for prec, d in results.items():
        speedup = 6.5 / d["latency"]
        print(f"  {prec.upper():<10} {d['latency']:<12.2f} {d['fps']:<10.1f} {d['size']:<10.1f} {speedup:.1f}x")
    print(f"  {'-'*55}")

    if "int8" in results:
        s = 6.5 / results["int8"]["latency"]
        print(f"\n  🏆 INT8 vs PyTorch 最终加速比: {s:.1f}x")
