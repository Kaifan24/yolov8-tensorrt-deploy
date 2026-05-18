import cv2
import numpy as np
from ultralytics import YOLO
import os

# 从COCO val集或网上下载一张图太依赖外网
# 直接用ultralytics自带的示例图(安装包里自带,不需要联网)
import ultralytics
assets_dir = os.path.join(os.path.dirname(ultralytics.__file__), "assets")
print("ultralytics自带示例图目录:", assets_dir)
print("可用图片:", os.listdir(assets_dir))

# 用自带的bus.jpg(ultralytics包里内置的,不需要联网下载)
img_path = os.path.join(assets_dir, "bus.jpg")

model = YOLO("yolov8n.pt")

results = model.predict(
    source=img_path,
    save=True,
    project="/workspace/outputs",
    name="predict_real",
    exist_ok=True,
)

for r in results:
    print(f"\n检测到 {len(r.boxes)} 个目标:")
    for i, (cls, conf) in enumerate(zip(r.boxes.cls, r.boxes.conf)):
        name = model.names[int(cls)]
        print(f"  [{i+1}] {name}: {conf:.3f}")

# 同时记录PyTorch推理的基线速度(后面和TensorRT对比用)
print("\n--- PyTorch 基线推理速度 ---")
import time
times = []
img = cv2.imread(img_path)
for i in range(50):
    t0 = time.time()
    model.predict(source=img, verbose=False)
    times.append((time.time() - t0) * 1000)

times = times[10:]  # 去掉前10次warmup
print(f"  平均延迟: {np.mean(times):.1f} ms")
print(f"  FPS: {1000 / np.mean(times):.1f}")
print(f"  (基于 {len(times)} 次推理, 去掉前10次warmup)")
