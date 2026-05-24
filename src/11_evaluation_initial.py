"""
KITTI 模型评估流水线 (Phase 1: 小样本验证)
用 TensorRT FP16 engine 在 100 张样本上跑通整个评估流程

核心逻辑:
  对每张图:
    1. TRT推理 -> 预测框列表
    2. 读KITTI真值 -> 真值框列表
    3. 类别映射 (KITTI -> COCO)
    4. IoU匹配 -> 算 TP/FP/FN
  最后输出: 整体 Precision/Recall/mAP, 各类别表现
"""
import os
import time
import numpy as np
import cv2
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from collections import defaultdict

# ============================================================
# 配置
# ============================================================

IMAGE_DIR = "/workspace/data/training/image_2"
LABEL_DIR = "/workspace/data/training/label_2"
ENGINE_PATH = "/workspace/engines/yolov8n_fp16.engine"
NUM_SAMPLES = 100   # Phase 1 只跑 100 张
INPUT_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH_MATCH = 0.5   # 评估时,预测和真值的IoU超过这个值才算"匹配"

# COCO的80类索引
COCO_PERSON = 0
COCO_CAR = 2
COCO_TRUCK = 7
COCO_BUS = 5
COCO_BICYCLE = 1

# KITTI -> COCO 类别映射
# Cyclist 比较特殊: KITTI视为一个整体(人+车),COCO拆成person+bicycle
# 这里映射到COCO的"bicycle",并在后面分析时单独标注这个对齐问题
KITTI_TO_COCO = {
    'Car':           COCO_CAR,
    'Van':           COCO_CAR,         # Van归到car
    'Truck':         COCO_TRUCK,
    'Pedestrian':    COCO_PERSON,
    'Person_sitting':COCO_PERSON,
    'Cyclist':       COCO_BICYCLE,     # 类别不完美对齐,后面分析这点
    # 'Tram': COCO无对应,跳过
}

COCO_NAMES = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']


# ============================================================
# TensorRT engine 封装
# ============================================================

class TRTInference:
    """加载TRT engine并提供推理接口"""
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
        input_size = int(np.prod(self.input_shape) * 4)
        output_size = int(np.prod(self.output_shape) * 4)
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        self.stream = cuda.Stream()
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

    def infer(self, img_preprocessed):
        h_output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_htod_async(self.d_input, img_preprocessed, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return h_output


# ============================================================
# 预处理 / 后处理 (和项目1同款逻辑)
# ============================================================

def preprocess(img_bgr, size=640):
    """KITTI图片 -> 模型输入张量, 同时返回缩放比例供还原坐标"""
    h, w = img_bgr.shape[:2]
    img_resized = cv2.resize(img_bgr, (size, size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_chw = img_float.transpose(2, 0, 1)
    img_batch = np.expand_dims(img_chw, 0)
    # 缩放比例 (原图 / 输入尺寸): 用来把模型坐标还原到原图
    scale_x = w / size
    scale_y = h / size
    return np.ascontiguousarray(img_batch), scale_x, scale_y


def decode_predictions(output, scale_x, scale_y, conf_thresh=0.25, iou_thresh=0.7):
    """
    把 TRT 原始输出 [1, 84, 8400] 解码成 (boxes, scores, class_ids)
    boxes 是 [x1, y1, x2, y2] 格式, 已经还原到原图坐标
    """
    pred = output[0].T  # [8400, 84]
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

    # xywh -> xyxy 同时缩放到原图
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2) * scale_x
    boxes_xyxy[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2) * scale_y
    boxes_xyxy[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2) * scale_x
    boxes_xyxy[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2) * scale_y

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
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return boxes_xyxy[keep], max_scores[keep], class_ids[keep]


# ============================================================
# KITTI 真值读取 + 类别映射
# ============================================================

def load_kitti_gt(label_path):
    """读KITTI标注, 返回COCO类别下的真值框列表"""
    gt_boxes = []     # [[x1,y1,x2,y2,coco_class_id], ...]
    gt_classes_orig = []  # 原始KITTI类别名, 用于细分分析
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            cls = parts[0]
            if cls not in KITTI_TO_COCO:
                continue   # DontCare / Misc / Tram 跳过
            coco_id = KITTI_TO_COCO[cls]
            x1, y1, x2, y2 = map(float, parts[4:8])
            gt_boxes.append([x1, y1, x2, y2, coco_id])
            gt_classes_orig.append(cls)
    return np.array(gt_boxes), gt_classes_orig


# ============================================================
# IoU 计算 + 匹配
# ============================================================

def iou_matrix(boxes_a, boxes_b):
    """
    计算两组框两两之间的IoU
    返回 shape=[len(a), len(b)] 的矩阵
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    a = boxes_a[:, :4]; b = boxes_b[:, :4]
    # broadcast 算两两IoU
    ax1 = a[:, 0:1]; ay1 = a[:, 1:2]; ax2 = a[:, 2:3]; ay2 = a[:, 3:4]
    bx1 = b[:, 0]; by1 = b[:, 1]; bx2 = b[:, 2]; by2 = b[:, 3]
    inter_x1 = np.maximum(ax1, bx1); inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2); inter_y2 = np.minimum(ay2, by2)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def evaluate_frame(pred_boxes, pred_scores, pred_classes, gt_boxes, iou_thresh=0.5):
    """
    单帧评估
    返回: TP数, FP数, FN数, 每个GT是否被命中
    """
    # 预测框结构: [N, 4]
    # GT结构: [M, 5] 最后一列是coco_id
    if len(gt_boxes) == 0:
        # 没有真值时, 所有预测都是FP
        return 0, len(pred_boxes), 0, np.array([])
    if len(pred_boxes) == 0:
        # 没有预测时, 所有真值都漏检
        return 0, 0, len(gt_boxes), np.zeros(len(gt_boxes), dtype=bool)

    # 算IoU矩阵
    ious = iou_matrix(pred_boxes, gt_boxes)  # [P, G]

    # 按预测置信度从高到低排序, 贪心匹配
    order = np.argsort(-pred_scores)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    pred_matched = np.zeros(len(pred_boxes), dtype=bool)

    for pi in order:
        # 找到IoU最高且类别匹配且未被匹配过的GT
        best_iou = iou_thresh
        best_gi = -1
        for gi in range(len(gt_boxes)):
            if gt_matched[gi]:
                continue
            if int(pred_classes[pi]) != int(gt_boxes[gi, 4]):
                continue   # 类别不匹配
            if ious[pi, gi] > best_iou:
                best_iou = ious[pi, gi]
                best_gi = gi
        if best_gi >= 0:
            gt_matched[best_gi] = True
            pred_matched[pi] = True

    TP = int(pred_matched.sum())
    FP = int((~pred_matched).sum())
    FN = int((~gt_matched).sum())
    return TP, FP, FN, gt_matched


# ============================================================
# 主流程: 跑 NUM_SAMPLES 张图
# ============================================================

print("=" * 60)
print(f"  Phase 1: 小样本验证 (前 {NUM_SAMPLES} 张)")
print(f"  Engine: {os.path.basename(ENGINE_PATH)}")
print("=" * 60)

inferencer = TRTInference(ENGINE_PATH)
img_files = sorted(os.listdir(IMAGE_DIR))[:NUM_SAMPLES]

total_TP = 0
total_FP = 0
total_FN = 0
# 按KITTI原始类别统计漏检
class_stats = defaultdict(lambda: {'gt': 0, 'hit': 0})

t0 = time.time()
for idx, fname in enumerate(img_files):
    frame_id = fname.replace('.png', '')
    img_path = os.path.join(IMAGE_DIR, fname)
    label_path = os.path.join(LABEL_DIR, frame_id + '.txt')

    img = cv2.imread(img_path)
    if img is None:
        continue

    # 推理
    img_in, sx, sy = preprocess(img, INPUT_SIZE)
    output = inferencer.infer(img_in)
    pred_boxes, pred_scores, pred_classes = decode_predictions(output, sx, sy, CONF_THRESH)

    # 真值
    gt, gt_orig = load_kitti_gt(label_path)

    # 评估
    TP, FP, FN, gt_matched = evaluate_frame(pred_boxes, pred_scores, pred_classes, gt, IOU_THRESH_MATCH)
    total_TP += TP; total_FP += FP; total_FN += FN

    # 按KITTI原始类别记录
    for i, cls in enumerate(gt_orig):
        class_stats[cls]['gt'] += 1
        if i < len(gt_matched) and gt_matched[i]:
            class_stats[cls]['hit'] += 1

    if (idx+1) % 20 == 0:
        print(f"  已处理 {idx+1}/{len(img_files)}")

elapsed = time.time() - t0
print(f"\n处理完成, 用时 {elapsed:.1f}s, 平均 {elapsed/len(img_files)*1000:.1f}ms/帧")

# ============================================================
# 输出指标
# ============================================================

print("\n" + "=" * 60)
print("  整体指标")
print("=" * 60)
precision = total_TP / (total_TP + total_FP + 1e-6)
recall = total_TP / (total_TP + total_FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print(f"  TP (命中):  {total_TP}")
print(f"  FP (误检):  {total_FP}")
print(f"  FN (漏检):  {total_FN}")
print(f"  Precision (命中/(命中+误检)): {precision:.3f}")
print(f"  Recall    (命中/(命中+漏检)): {recall:.3f}")
print(f"  F1 Score:                     {f1:.3f}")

print("\n" + "=" * 60)
print("  各KITTI类别表现 (按原始KITTI类别细分)")
print("=" * 60)
print(f"  {'类别':<18} {'真值数':<10} {'命中数':<10} {'召回率':<10}")
print(f"  {'-'*50}")
for cls in sorted(class_stats.keys()):
    s = class_stats[cls]
    rate = s['hit'] / s['gt'] if s['gt'] > 0 else 0
    print(f"  {cls:<18} {s['gt']:<10} {s['hit']:<10} {rate:.1%}")

print("\nPhase 1 完成. 数字合理就跑 Phase 2 (全量+三档对比).")
