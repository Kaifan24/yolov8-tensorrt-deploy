import tensorrt as trt
import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pycuda.autoinit
import pycuda.driver as cuda
from ultralytics import YOLO

def trt_inference(engine_path, img_preprocessed):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    output_shape = engine.get_tensor_shape(output_name)
    input_size = int(np.prod(img_preprocessed.shape) * 4)
    output_size = int(np.prod(output_shape) * 4)
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    h_output = np.empty(output_shape, dtype=np.float32)
    stream = cuda.Stream()
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    cuda.memcpy_htod_async(d_input, img_preprocessed, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output

def preprocess_image(img_path, size=640):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (size, size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_chw = img_float.transpose(2, 0, 1)
    img_batch = np.expand_dims(img_chw, 0)
    return np.ascontiguousarray(img_batch)

def decode_yolo_output(output, conf_threshold=0.25, iou_threshold=0.7):
    pred = output[0].T
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)
    mask = max_scores > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    if len(boxes_xywh) == 0:
        return [], [], []
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    keep = []
    order = max_scores.argsort()[::-1]
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[order[1:], 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[order[1:], 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[order[1:], 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[order[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (boxes_xyxy[i,2]-boxes_xyxy[i,0]) * (boxes_xyxy[i,3]-boxes_xyxy[i,1])
        area_j = (boxes_xyxy[order[1:],2]-boxes_xyxy[order[1:],0]) * (boxes_xyxy[order[1:],3]-boxes_xyxy[order[1:],1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return boxes_xyxy[keep], max_scores[keep], class_ids[keep]

COCO_NAMES = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

print("=" * 60)
print("  精度验证: PyTorch vs TRT-FP32 vs TRT-FP16 vs TRT-INT8")
print("=" * 60)

import ultralytics as ul
img_path = os.path.join(os.path.dirname(ul.__file__), "assets", "bus.jpg")
img_preprocessed = preprocess_image(img_path)

print("\n--- PyTorch ---")
model = YOLO("yolov8n.pt")
pt_results = model.predict(source=img_path, verbose=False, conf=0.25)
pt_boxes = []
for r in pt_results:
    for cls, conf in zip(r.boxes.cls, r.boxes.conf):
        name = model.names[int(cls)]
        pt_boxes.append((name, float(conf)))
        print(f"  {name}: {conf:.3f}")
print(f"  目标总数: {len(pt_boxes)}")

all_results = {"PyTorch": pt_boxes}
for prec in ["fp32", "fp16", "int8"]:
    engine_path = f"/workspace/engines/yolov8n_{prec}.engine"
    if not os.path.exists(engine_path):
        print(f"\n--- TRT-{prec.upper()} --- (跳过)")
        continue
    print(f"\n--- TRT-{prec.upper()} ---")
    output = trt_inference(engine_path, img_preprocessed)
    boxes, scores, class_ids = decode_yolo_output(output)
    trt_boxes = []
    for box, score, cid in zip(boxes, scores, class_ids):
        name = COCO_NAMES[int(cid)]
        trt_boxes.append((name, float(score)))
        print(f"  {name}: {score:.3f}")
    print(f"  目标总数: {len(trt_boxes)}")
    all_results[f"TRT-{prec.upper()}"] = trt_boxes

print("\n" + "=" * 60)
print("  精度对比汇总")
print("=" * 60)
print(f"  {'模型':<15} {'检测数量':<10} {'检测到的类别'}")
print(f"  {'-'*50}")
for name, boxes in all_results.items():
    classes = [b[0] for b in boxes]
    summary = ", ".join(f"{c}({classes.count(c)})" for c in dict.fromkeys(classes))
    print(f"  {name:<15} {len(boxes):<10} {summary}")

print("\n" + "=" * 60)
print("  生成可视化图表")
print("=" * 60)
os.makedirs("/workspace/outputs/charts", exist_ok=True)

labels = ['PyTorch', 'TRT-FP32', 'TRT-FP16', 'TRT-INT8']
latencies = [6.50, 3.15, 2.24, 2.04]
fps_values = [153.4, 317.2, 445.6, 491.3]
sizes = [12.3, 19.5, 9.4, 6.8]
colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, data, ylabel, title in zip(axes,
    [latencies, fps_values, sizes],
    ['Latency (ms)', 'FPS', 'File Size (MB)'],
    ['Inference Latency', 'Throughput (FPS)', 'Model Size']):
    bars = ax.bar(labels, data, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    for bar, val in zip(bars, data):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()*1.02, f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, max(data)*1.3)
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/outputs/charts/benchmark_comparison.png', dpi=150, bbox_inches='tight')
print("  已保存: outputs/charts/benchmark_comparison.png")

fig2, ax = plt.subplots(figsize=(8, 5))
speedups = [1.0, 6.50/3.15, 6.50/2.24, 6.50/2.04]
bars = ax.bar(labels, speedups, color=colors, edgecolor='white', linewidth=1.5)
ax.set_ylabel('Speedup vs PyTorch', fontsize=12)
ax.set_title('Speedup Ratio (PyTorch = 1.0x)', fontsize=14, fontweight='bold')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars, speedups):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.05, f'{val:.1f}x', ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.set_ylim(0, max(speedups)*1.3)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/outputs/charts/speedup_ratio.png', dpi=150, bbox_inches='tight')
print("  已保存: outputs/charts/speedup_ratio.png")
print("\n全部完成!")
