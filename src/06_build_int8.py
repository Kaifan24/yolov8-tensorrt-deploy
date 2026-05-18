import tensorrt as trt
import numpy as np
import os
import time
import cv2
import glob

class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 校准器: 读取真实图片,喂给TRT做校准
    """
    def __init__(self, img_dir, batch_size=1, input_shape=(1, 3, 640, 640), cache_file="calibration.cache"):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.cache_file = cache_file

        # 收集图片路径
        self.img_files = []
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            self.img_files.extend(glob.glob(os.path.join(img_dir, ext)))
        self.img_files = self.img_files[:500]  # 最多用500张
        print(f"  校准图片数量: {len(self.img_files)}")

        self.index = 0
        self.device_input = None

        # 分配GPU显存
        import pycuda.driver as cuda
        import pycuda.autoinit
        self.cuda = cuda
        size = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
        self.device_input = cuda.mem_alloc(size)

    def preprocess(self, img_path):
        """预处理: 读图 -> resize -> 归一化 -> CHW"""
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(img, 0)  # 加batch维度

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index >= len(self.img_files):
            return None
        img = self.preprocess(self.img_files[self.index])
        self.cuda.memcpy_htod(self.device_input, img.ravel())
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


def build_int8_engine(onnx_path, engine_path, calib_img_dir):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  解析错误: {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # INT8通常配合FP16

    # 设置校准器
    calibrator = ImageCalibrator(
        img_dir=calib_img_dir,
        cache_file="/workspace/engines/int8_calibration.cache"
    )
    config.int8_calibrator = calibrator

    print("  正在构建 INT8 engine (需要校准,会比较慢)...")
    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    build_time = time.time() - t0

    if engine_bytes is None:
        print("  构建失败!")
        return None

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    size_mb = os.path.getsize(engine_path) / 1024 / 1024
    print(f"  INT8 engine 构建成功 ✅  耗时 {build_time:.1f}s  文件 {size_mb:.1f}MB")
    return engine_path


if __name__ == "__main__":
    # 先准备校准图片: 用ultralytics自带的 + 自己生成一些
    calib_dir = "/workspace/calib_images"
    os.makedirs(calib_dir, exist_ok=True)

    # 复制ultralytics自带的示例图
    import ultralytics, shutil
    assets = os.path.join(os.path.dirname(ultralytics.__file__), "assets")
    for f in os.listdir(assets):
        if f.endswith(".jpg"):
            shutil.copy(os.path.join(assets, f), calib_dir)

    # 再生成一些随机图片补充(校准数据越多越好,但这里先用少量跑通)
    print("生成补充校准图片...")
    for i in range(50):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        # 加一些随机矩形模拟目标
        for _ in range(np.random.randint(2, 6)):
            x1, y1 = np.random.randint(0, 500, 2)
            x2, y2 = x1 + np.random.randint(40, 140), y1 + np.random.randint(40, 140)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(img, (x1, y1), (min(x2,639), min(y2,639)), color, -1)
        cv2.imwrite(os.path.join(calib_dir, f"calib_{i:03d}.jpg"), img)

    total = len([f for f in os.listdir(calib_dir) if f.endswith(".jpg")])
    print(f"校准图片准备完成: {total} 张\n")

    print("=" * 50)
    print("构建 TensorRT INT8 engine (带校准)")
    print("=" * 50)
    build_int8_engine(
        onnx_path="/workspace/yolov8n.onnx",
        engine_path="/workspace/engines/yolov8n_int8.engine",
        calib_img_dir=calib_dir,
    )
