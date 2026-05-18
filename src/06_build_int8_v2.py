import tensorrt as trt
import numpy as np
import os
import time
import cv2
import glob

# 必须在tensorrt之前import pycuda,避免CUDA context警告
import pycuda.autoinit
import pycuda.driver as cuda

class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, img_dir, batch_size=1, input_shape=(1, 3, 640, 640), cache_file="calibration.cache"):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.img_files = []
        for ext in ["*.jpg", "*.png"]:
            self.img_files.extend(glob.glob(os.path.join(img_dir, ext)))
        self.img_files = self.img_files[:500]
        print(f"  校准图片数量: {len(self.img_files)}")
        self.index = 0
        size = int(np.prod(input_shape) * 4)
        self.device_input = cuda.mem_alloc(size)

    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, 0).astype(np.float32)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index >= len(self.img_files):
            return None
        img = self.preprocess(self.img_files[self.index])
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(img))
        self.index += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"  校准缓存已保存: {self.cache_file}")


if __name__ == "__main__":
    onnx_path = "/workspace/yolov8n.onnx"
    engine_path = "/workspace/engines/yolov8n_int8.engine"
    calib_dir = "/workspace/calib_images"
    cache_file = "/workspace/engines/int8_calibration.cache"

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    # 关键改动1: 给更多显存(2GB而不是1GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    # 关键改动2: 允许TRT对部分层回退到FP16(而不是强制全INT8)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    calibrator = ImageCalibrator(calib_dir, cache_file=cache_file)
    config.int8_calibrator = calibrator

    print("正在构建 INT8 engine...")
    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    build_time = time.time() - t0

    if engine_bytes:
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        size_mb = os.path.getsize(engine_path) / 1024 / 1024
        print(f"INT8 engine 构建成功 ✅  耗时 {build_time:.1f}s  文件 {size_mb:.1f}MB")
    else:
        # 如果还失败,去掉OBEY_PRECISION_CONSTRAINTS再试
        print("第一次尝试失败,去掉精度约束重试...")
        config2 = builder.create_builder_config()
        config2.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        config2.set_flag(trt.BuilderFlag.INT8)
        config2.set_flag(trt.BuilderFlag.FP16)
        calibrator2 = ImageCalibrator(calib_dir, cache_file=cache_file)
        config2.int8_calibrator = calibrator2

        # 重新创建network(builder用过一次后需要重建)
        network2 = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser2 = trt.OnnxParser(network2, logger)
        with open(onnx_path, "rb") as f:
            parser2.parse(f.read())

        engine_bytes = builder.build_serialized_network(network2, config2)
        build_time = time.time() - t0
        if engine_bytes:
            with open(engine_path, "wb") as f:
                f.write(engine_bytes)
            size_mb = os.path.getsize(engine_path) / 1024 / 1024
            print(f"INT8 engine 构建成功 ✅ (重试)  耗时 {build_time:.1f}s  文件 {size_mb:.1f}MB")
        else:
            print("INT8 构建失败 ❌")
