"""
INT8 重校准: 用 KITTI 真实图片作为校准集
对比之前用 52 张噪声图的版本, 验证"校准数据决定INT8精度"的核心论点
"""
import os
import numpy as np
import cv2
import tensorrt as trt
import time
import random
import pycuda.autoinit
import pycuda.driver as cuda

ONNX_PATH = "/workspace/yolov8n.onnx"            # 等会要把项目1的ONNX拷过来
ENGINE_PATH = "/workspace/engines/yolov8n_int8_v2.engine"  # 新engine,不覆盖旧的
CACHE_PATH = "/workspace/engines/int8_calib_v2.cache"
KITTI_IMG_DIR = "/workspace/data/training/image_2"
NUM_CALIB = 500     # 用500张真实图校准 (vs 之前52张噪声)
INPUT_SIZE = 640


class KITTICalibrator(trt.IInt8EntropyCalibrator2):
    """用真实 KITTI 图片做 INT8 校准, letterbox 预处理保证一致性"""
    def __init__(self, img_dir, num_samples=500, cache_file="calib.cache"):
        super().__init__()
        self.batch_size = 1
        self.input_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)
        self.cache_file = cache_file

        # 从7481张里随机抽 num_samples 张
        all_imgs = sorted(os.listdir(img_dir))
        random.seed(42)   # 固定种子,保证可复现
        self.img_files = random.sample(all_imgs, min(num_samples, len(all_imgs)))
        self.img_files = [os.path.join(img_dir, f) for f in self.img_files]
        print(f"  校准集: {len(self.img_files)} 张真实KITTI图片 (vs 旧版52张噪声图)")

        self.index = 0
        size = int(np.prod(self.input_shape) * 4)
        self.device_input = cuda.mem_alloc(size)

    def preprocess(self, img_path):
        """和评估代码用同一套 letterbox - 校准/推理预处理必须一致"""
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        r = min(INPUT_SIZE / h, INPUT_SIZE / w)
        nh, nw = int(round(h*r)), int(round(w*r))
        img_r = cv2.resize(img, (nw, nh))
        px, py = (INPUT_SIZE-nw)//2, (INPUT_SIZE-nh)//2
        pad = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        pad[py:py+nh, px:px+nw] = img_r
        rgb = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(np.expand_dims(rgb.transpose(2,0,1), 0))

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index >= len(self.img_files):
            return None
        img = self.preprocess(self.img_files[self.index])
        cuda.memcpy_htod(self.device_input, img)
        self.index += 1
        if self.index % 100 == 0:
            print(f"    已喂入 {self.index}/{len(self.img_files)} 张")
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
    if not os.path.exists(ONNX_PATH):
        print(f"ERROR: 找不到 ONNX 文件: {ONNX_PATH}")
        print("请把项目1的 yolov8n.onnx 复制到 /workspace/yolov8n.onnx")
        exit(1)

    print("=" * 60)
    print("  INT8 重校准 (KITTI真实数据)")
    print("=" * 60)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(ONNX_PATH, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    
    calibrator = KITTICalibrator(KITTI_IMG_DIR, NUM_CALIB, CACHE_PATH)
    config.int8_calibrator = calibrator

    print("\n  开始构建 INT8 engine (5-10分钟)...")
    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    elapsed = time.time() - t0

    if engine_bytes:
        with open(ENGINE_PATH, "wb") as f:
            f.write(engine_bytes)
        size_mb = os.path.getsize(ENGINE_PATH) / 1024 / 1024
        print(f"\n  构建成功 ✅  用时 {elapsed:.0f}s  文件 {size_mb:.1f}MB")
        print(f"  保存到: {ENGINE_PATH}")
    else:
        print("\n  构建失败 ❌")
