import tensorrt as trt
import numpy as np
import time
import os

def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def benchmark_engine(engine_path, precision_name, num_runs=100, warmup=20):
    """加载engine,跑推理,测延迟"""
    import pycuda.driver as cuda
    import pycuda.autoinit

    print(f"\n--- {precision_name} ---")
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    # 获取输入输出信息
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    print(f"  输入: {input_name} {list(input_shape)}")
    print(f"  输出: {output_name} {list(output_shape)}")

    # 分配GPU显存
    input_size = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
    output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)

    # 准备假数据(benchmark只测速度,不需要真图片)
    h_input = np.random.randn(*input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)

    stream = cuda.Stream()

    # 设置tensor地址
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # Warmup
    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    # 正式测速
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
    print(f"  平均延迟: {avg:.2f} ms (±{std:.2f})")
    print(f"  FPS: {fps:.1f}")
    return avg, fps

if __name__ == "__main__":
    print("=" * 50)
    print("TensorRT 推理性能 Benchmark")
    print("=" * 50)

    results = {}

    for prec in ["fp32", "fp16"]:
        path = f"/workspace/engines/yolov8n_{prec}.engine"
        if os.path.exists(path):
            avg, fps = benchmark_engine(path, prec.upper())
            results[prec] = {"latency_ms": avg, "fps": fps}

    # 汇总对比表
    print("\n" + "=" * 50)
    print("性能对比汇总")
    print("=" * 50)
    print(f"{'精度':<10} {'延迟(ms)':<15} {'FPS':<10}")
    print("-" * 35)

    # 加上之前的PyTorch基线
    print(f"{'PyTorch':<10} {'6.5':<15} {'153.4':<10}  (基线)")
    for prec, data in results.items():
        tag = ""
        print(f"{prec.upper():<10} {data['latency_ms']:<15.2f} {data['fps']:<10.1f} {tag}")

    print("-" * 35)
    if "fp16" in results:
        speedup = 6.5 / results["fp16"]["latency_ms"]
        print(f"\nFP16 vs PyTorch 加速比: {speedup:.1f}x")
