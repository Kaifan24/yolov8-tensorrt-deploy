import tensorrt as trt
import os
import time

def build_engine(onnx_path, engine_path, precision="fp32"):
    """
    把 ONNX 模型转成 TensorRT engine
    precision: "fp32", "fp16", "int8"
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # 读取 ONNX
    print(f"  读取 ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  解析错误: {parser.get_error(i)}")
            return None

    # 配置 builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB显存给TRT优化用

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  启用 FP16 ✅")
        else:
            print("  GPU不支持FP16!")
            return None

    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # INT8需要校准(calibration),这里先用FP16兜底
            # 后面专门写calibrator再替换
            config.set_flag(trt.BuilderFlag.FP16)  # INT8通常配合FP16
            print("  启用 INT8 + FP16 fallback ✅")
            print("  (暂无calibrator,TRT会用默认range,精度可能偏低)")
        else:
            print("  GPU不支持INT8!")
            return None
    else:
        print("  使用 FP32 (默认精度)")

    # 构建 engine(这一步耗时最长,TRT在做层融合/kernel选择/自动调优)
    print("  正在构建 engine,请等待...")
    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    build_time = time.time() - t0

    if engine_bytes is None:
        print("  构建失败!")
        return None

    # 保存 engine
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    size_mb = os.path.getsize(engine_path) / 1024 / 1024
    print(f"  构建成功 ✅  耗时 {build_time:.1f}s  文件 {size_mb:.1f}MB")
    print(f"  保存到: {engine_path}")
    return engine_path


if __name__ == "__main__":
    onnx_path = "/workspace/yolov8n.onnx"
    os.makedirs("/workspace/engines", exist_ok=True)

    for prec in ["fp32", "fp16", "int8"]:
        print(f"\n{'='*50}")
        print(f"构建 TensorRT engine: {prec.upper()}")
        print(f"{'='*50}")
        engine_path = f"/workspace/engines/yolov8n_{prec}.engine"
        build_engine(onnx_path, engine_path, prec)

    print("\n" + "="*50)
    print("所有 engine 构建完成!")
    print("="*50)
    for f in os.listdir("/workspace/engines"):
        size = os.path.getsize(f"/workspace/engines/{f}") / 1024 / 1024
        print(f"  {f}: {size:.1f} MB")
