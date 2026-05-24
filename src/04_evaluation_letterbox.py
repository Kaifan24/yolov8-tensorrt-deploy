"""
KITTI 模型评估流水线 (修复版: 加 letterbox 预处理)
对比之前 resize 直拉的版本,验证预处理一致性的重要性
"""
import os
import time
import numpy as np
import cv2
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from collections import defaultdict

IMAGE_DIR = "/workspace/data/training/image_2"
LABEL_DIR = "/workspace/data/training/label_2"
ENGINE_PATH = "/workspace/engines/yolov8n_fp16.engine"
NUM_SAMPLES = 100
INPUT_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH_MATCH = 0.5

COCO_PERSON, COCO_CAR, COCO_TRUCK, COCO_BUS, COCO_BICYCLE = 0, 2, 7, 5, 1

KITTI_TO_COCO = {
    'Car': COCO_CAR, 'Van': COCO_CAR, 'Truck': COCO_TRUCK,
    'Pedestrian': COCO_PERSON, 'Person_sitting': COCO_PERSON,
    'Cyclist': COCO_BICYCLE,
}


class TRTInference:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape) * 4))
        self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape) * 4))
        self.stream = cuda.Stream()
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

    def infer(self, img_in):
        h_output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_htod_async(self.d_input, img_in, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return h_output


def preprocess_letterbox(img_bgr, size=640):
    """
    Letterbox 预处理 - YOLO 官方标准做法
    保持宽高比, 不够的部分用灰色(114)填充
    返回: 输入张量, 缩放系数 r, padding (pad_x, pad_y) - 用来还原坐标
    """
    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)         # 等比缩放系数
    new_h, new_w = int(round(h * r)), int(round(w * r))
    img_resized = cv2.resize(img_bgr, (new_w, new_h))

    # 用 114 灰色 padding 到 size x size
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    img_padded = np.full((size, size, 3), 114, dtype=np.uint8)
    img_padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized

    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_chw = img_float.transpose(2, 0, 1)
    return np.ascontiguousarray(np.expand_dims(img_chw, 0)), r, pad_x, pad_y


def decode_letterbox(output, r, pad_x, pad_y, conf_thresh=0.25, iou_thresh=0.7):
    """解码 + 反letterbox还原到原图坐标"""
    pred = output[0].T
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = max_scores > conf_thresh
    if not mask.any():
        return np.zeros((0,4)), np.zeros(0), np.zeros(0, dtype=int)

    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    # xywh(在640输入坐标系) -> xyxy(原图坐标)
    boxes_xyxy = np.zeros_like(boxes_xywh)
    # 先在 letterbox 坐标系内 xywh -> xyxy
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    # 减去 padding, 再除以 r, 还原到原图
    boxes_xyxy[:, [0,2]] = (boxes_xyxy[:, [0,2]] - pad_x) / r
    boxes_xyxy[:, [1,3]] = (boxes_xyxy[:, [1,3]] - pad_y) / r

    # NMS
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
        w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (boxes_xyxy[i,2]-boxes_xyxy[i,0])*(boxes_xyxy[i,3]-boxes_xyxy[i,1])
        area_j = (boxes_xyxy[order[1:],2]-boxes_xyxy[order[1:],0])*(boxes_xyxy[order[1:],3]-boxes_xyxy[order[1:],1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return boxes_xyxy[keep], max_scores[keep], class_ids[keep]


def load_kitti_gt(label_path):
    gt_boxes = []; gt_orig = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            cls = parts[0]
            if cls not in KITTI_TO_COCO:
                continue
            coco_id = KITTI_TO_COCO[cls]
            x1, y1, x2, y2 = map(float, parts[4:8])
            gt_boxes.append([x1, y1, x2, y2, coco_id])
            gt_orig.append(cls)
    return np.array(gt_boxes), gt_orig


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:,0:1], a[:,1:2], a[:,2:3], a[:,3:4]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)


def evaluate_frame(pred_boxes, pred_scores, pred_classes, gt_boxes, iou_thresh=0.5):
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, np.array([])
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), np.zeros(len(gt_boxes), dtype=bool)
    ious = iou_matrix(pred_boxes, gt_boxes)
    order = np.argsort(-pred_scores)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    pred_matched = np.zeros(len(pred_boxes), dtype=bool)
    for pi in order:
        best_iou = iou_thresh; best_gi = -1
        for gi in range(len(gt_boxes)):
            if gt_matched[gi]: continue
            if int(pred_classes[pi]) != int(gt_boxes[gi, 4]): continue
            if ious[pi, gi] > best_iou:
                best_iou = ious[pi, gi]; best_gi = gi
        if best_gi >= 0:
            gt_matched[best_gi] = True
            pred_matched[pi] = True
    return int(pred_matched.sum()), int((~pred_matched).sum()), int((~gt_matched).sum()), gt_matched


print("=" * 60)
print(f"  Phase 1.5: Letterbox 修复版 (前 {NUM_SAMPLES} 张)")
print("=" * 60)

inferencer = TRTInference(ENGINE_PATH)
img_files = sorted(os.listdir(IMAGE_DIR))[:NUM_SAMPLES]
total_TP = total_FP = total_FN = 0
class_stats = defaultdict(lambda: {'gt': 0, 'hit': 0})

t0 = time.time()
for idx, fname in enumerate(img_files):
    frame_id = fname.replace('.png', '')
    img = cv2.imread(os.path.join(IMAGE_DIR, fname))
    if img is None: continue
    img_in, r, px, py = preprocess_letterbox(img, INPUT_SIZE)
    output = inferencer.infer(img_in)
    pb, ps, pc = decode_letterbox(output, r, px, py, CONF_THRESH)
    gt, gt_orig = load_kitti_gt(os.path.join(LABEL_DIR, frame_id + '.txt'))
    TP, FP, FN, gm = evaluate_frame(pb, ps, pc, gt, IOU_THRESH_MATCH)
    total_TP += TP; total_FP += FP; total_FN += FN
    for i, cls in enumerate(gt_orig):
        class_stats[cls]['gt'] += 1
        if i < len(gm) and gm[i]:
            class_stats[cls]['hit'] += 1
    if (idx+1) % 20 == 0:
        print(f"  已处理 {idx+1}/{len(img_files)}")

elapsed = time.time() - t0
print(f"\n用时 {elapsed:.1f}s")

precision = total_TP / (total_TP + total_FP + 1e-6)
recall = total_TP / (total_TP + total_FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print("\n" + "=" * 60)
print("  Letterbox 版指标 (期待大幅提升,尤其是Pedestrian)")
print("=" * 60)
print(f"  TP: {total_TP}  FP: {total_FP}  FN: {total_FN}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1:        {f1:.3f}")

print("\n  各KITTI类别召回率对比 (修复前 → 修复后):")
print(f"  {'类别':<18} {'GT':<8} {'命中':<8} {'召回率':<10}")
print(f"  {'-'*50}")
prev = {'Car':0.411, 'Van':0.152, 'Truck':0.286, 'Pedestrian':0.043,
        'Person_sitting':0.0, 'Cyclist':0.0}
for cls in sorted(class_stats.keys()):
    s = class_stats[cls]
    rate = s['hit'] / s['gt'] if s['gt'] > 0 else 0
    delta = rate - prev.get(cls, 0)
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "·")
    print(f"  {cls:<18} {s['gt']:<8} {s['hit']:<8} {rate:.1%}  ({prev.get(cls,0):.1%} {arrow})")
