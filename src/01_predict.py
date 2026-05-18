import cv2
import numpy as np
from ultralytics import YOLO

# 用代码自己造一张测试图(不依赖外网),画几个色块+文字
img = np.full((640, 640, 3), 235, dtype=np.uint8)
cv2.rectangle(img, (60, 120), (240, 480), (80, 80, 200), -1)
cv2.rectangle(img, (320, 180), (560, 520), (200, 140, 60), -1)
cv2.circle(img, (430, 110), 55, (60, 180, 120), -1)
cv2.putText(img, "TEST IMAGE", (160, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
cv2.imwrite("/workspace/outputs/test_input.jpg", img)
print("测试图已生成: /workspace/outputs/test_input.jpg")

# 加载官方预训练 YOLOv8n(权重已下好,会直接用本地的)
model = YOLO("yolov8n.pt")

# 对自造图推理。注意:这张图是色块,YOLO大概率检测不到COCO目标,
# 目的只是验证"整条推理链路能跑通",不是验证检测效果
results = model.predict(
    source="/workspace/outputs/test_input.jpg",
    save=True,
    project="/workspace/outputs",
    name="predict_01",
)

for r in results:
    print("链路跑通 ✅  检测到目标数量:", len(r.boxes))

print("\n下一步用真实数据集(KITTI)才会有有意义的检测结果,这步只验证环境与推理链路。")
