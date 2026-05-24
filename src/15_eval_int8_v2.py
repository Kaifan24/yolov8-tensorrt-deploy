"""
评估重校准后的 INT8 v2 engine
对比 v1(噪声图校准) vs v2(KITTI真实数据校准)
核心问题: 校准数据真的能决定 INT8 精度吗?
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
ENGINE_PATH = "/workspace/engines/yolov8n_int8_v2.engine"
INPUT_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH_MATCH = 0.5

COCO_PERSON, COCO_CAR, COCO_TRUCK, COCO_BICYCLE = 0, 2, 7, 1
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


def preprocess_letterbox(img, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img_r = cv2.resize(img, (nw, nh))
    px, py = (size - nw) // 2, (size - nh) // 2
    pad = np.full((size, size, 3), 114, dtype=np.uint8)
    pad[py:py+nh, px:px+nw] = img_r
    img_rgb = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)
    img_t = img_rgb.astype(np.float32) / 255.0
    return np.ascontiguousarray(np.expand_dims(img_t.transpose(2,0,1), 0)), r, px, py


def decode(output, r, px, py, conf=0.25, iou_t=0.7):
    pred = output[0].T
    bw = pred[:, :4]; cs = pred[:, 4:]
    sc = cs.max(axis=1); ci = cs.argmax(axis=1)
    m = sc > conf
    if not m.any(): return np.zeros((0,4)), np.zeros(0), np.zeros(0, dtype=int)
    bw, sc, ci = bw[m], sc[m], ci[m]
    bx = np.zeros_like(bw)
    bx[:,0]=bw[:,0]-bw[:,2]/2; bx[:,1]=bw[:,1]-bw[:,3]/2
    bx[:,2]=bw[:,0]+bw[:,2]/2; bx[:,3]=bw[:,1]+bw[:,3]/2
    bx[:,[0,2]]=(bx[:,[0,2]]-px)/r; bx[:,[1,3]]=(bx[:,[1,3]]-py)/r
    keep=[]; order=sc.argsort()[::-1]
    while len(order):
        i=order[0]; keep.append(i)
        if len(order)==1: break
        x1=np.maximum(bx[i,0],bx[order[1:],0]); y1=np.maximum(bx[i,1],bx[order[1:],1])
        x2=np.minimum(bx[i,2],bx[order[1:],2]); y2=np.minimum(bx[i,3],bx[order[1:],3])
        iw=np.maximum(0,x2-x1); ih=np.maximum(0,y2-y1)
        inter=iw*ih
        ai=(bx[i,2]-bx[i,0])*(bx[i,3]-bx[i,1])
        aj=(bx[order[1:],2]-bx[order[1:],0])*(bx[order[1:],3]-bx[order[1:],1])
        iou=inter/(ai+aj-inter+1e-6)
        order=order[np.where(iou<=iou_t)[0]+1]
    return bx[keep], sc[keep], ci[keep]


def load_gt(label_path):
    gts=[]; origs=[]; props=[]
    with open(label_path) as f:
        for line in f:
            p = line.strip().split()
            cls = p[0]
            if cls not in KITTI_TO_COCO: continue
            cid = KITTI_TO_COCO[cls]
            x1,y1,x2,y2 = map(float, p[4:8])
            trunc=float(p[1]); occ=int(p[2]); dist=float(p[13])
            h = y2-y1
            gts.append([x1,y1,x2,y2,cid]); origs.append(cls)
            tag='normal'
            if h<32: tag='small'
            elif occ>=2: tag='occluded'
            elif dist>50: tag='far'
            props.append(tag)
    return np.array(gts), origs, props


def iou_mat(a, b):
    if len(a)==0 or len(b)==0: return np.zeros((len(a),len(b)))
    ax1,ay1,ax2,ay2 = a[:,0:1],a[:,1:2],a[:,2:3],a[:,3:4]
    bx1,by1,bx2,by2 = b[:,0],b[:,1],b[:,2],b[:,3]
    iw=np.clip(np.minimum(ax2,bx2)-np.maximum(ax1,bx1),0,None)
    ih=np.clip(np.minimum(ay2,by2)-np.maximum(ay1,by1),0,None)
    inter=iw*ih
    aa=(ax2-ax1)*(ay2-ay1); ab=(bx2-bx1)*(by2-by1)
    return inter/(aa+ab-inter+1e-6)


def eval_frame(pb,ps,pc,gt,iou_t=0.5):
    if len(gt)==0: return 0,len(pb),0,np.array([])
    if len(pb)==0: return 0,0,len(gt),np.zeros(len(gt),dtype=bool)
    ious=iou_mat(pb,gt); order=np.argsort(-ps)
    gm=np.zeros(len(gt),dtype=bool); pm=np.zeros(len(pb),dtype=bool)
    for pi in order:
        bi=-1; bv=iou_t
        for gi in range(len(gt)):
            if gm[gi]: continue
            if int(pc[pi])!=int(gt[gi,4]): continue
            if ious[pi,gi]>bv: bv=ious[pi,gi]; bi=gi
        if bi>=0: gm[bi]=True; pm[pi]=True
    return int(pm.sum()),int((~pm).sum()),int((~gm).sum()),gm


# ============================================================
# 跑评估
# ============================================================

print("=" * 60)
print("  INT8 v2 评估 (KITTI真实数据校准)")
print("=" * 60)

infer = TRTInference(ENGINE_PATH)
img_files = sorted(os.listdir(IMAGE_DIR))

TP=FP=FN=0
cls_stat = defaultdict(lambda: {'gt':0,'hit':0})
diff_stat = defaultdict(lambda: {'gt':0,'hit':0})

t0 = time.time()
for idx, fname in enumerate(img_files):
    fid = fname.replace('.png','')
    img = cv2.imread(os.path.join(IMAGE_DIR, fname))
    if img is None: continue
    img_in, r, px, py = preprocess_letterbox(img, INPUT_SIZE)
    out = infer.infer(img_in)
    pb, ps, pc = decode(out, r, px, py, CONF_THRESH)
    gt, gt_orig, gt_props = load_gt(os.path.join(LABEL_DIR, fid+'.txt'))
    tp,fp,fn,gm = eval_frame(pb,ps,pc,gt,IOU_THRESH_MATCH)
    TP+=tp; FP+=fp; FN+=fn
    for i, cls in enumerate(gt_orig):
        cls_stat[cls]['gt']+=1
        if i < len(gm) and gm[i]: cls_stat[cls]['hit']+=1
    for i, tag in enumerate(gt_props):
        diff_stat[tag]['gt']+=1
        if i < len(gm) and gm[i]: diff_stat[tag]['hit']+=1
    if (idx+1)%1000==0:
        print(f"  {idx+1}/{len(img_files)}  已用 {time.time()-t0:.0f}s")

elapsed = time.time()-t0
P=TP/(TP+FP+1e-6); R=TP/(TP+FN+1e-6); F1=2*P*R/(P+R+1e-6)

print(f"\n  完成! 用时 {elapsed:.0f}s")
print(f"  TP={TP}  FP={FP}  FN={FN}")
print(f"  Precision={P:.3f}  Recall={R:.3f}  F1={F1:.3f}")

# ============================================================
# 对比 v1 vs v2
# ============================================================

# v1 旧版数据 (52张噪声图校准)
v1 = {
    'P': 0.864, 'R': 0.298, 'F1': 0.443,
    'cls': {
        'Car': 0.337, 'Cyclist': 0.004, 'Pedestrian': 0.342,
        'Person_sitting': 0.063, 'Truck': 0.083, 'Van': 0.112,
    },
    'diff': {'normal': 0.470, 'small': 0.048, 'occluded': 0.132, 'far': 0.000},
}
# FP16 基线
fp16 = {
    'P': 0.559, 'R': 0.644, 'F1': 0.599,
    'cls': {
        'Car': 0.732, 'Cyclist': 0.037, 'Pedestrian': 0.567,
        'Person_sitting': 0.248, 'Truck': 0.236, 'Van': 0.421,
    },
    'diff': {'normal': 0.788, 'small': 0.521, 'occluded': 0.415, 'far': 0.164},
}

print("\n" + "=" * 70)
print("  INT8 v1 (噪声图校准) vs v2 (KITTI真实数据校准) vs FP16基线")
print("=" * 70)
print(f"  {'指标':<18} {'INT8 v1':<12} {'INT8 v2':<12} {'FP16基线':<12} {'v2 vs v1'}")
print(f"  {'-'*70}")
delta_P = P - v1['P']
delta_R = R - v1['R']
delta_F1 = F1 - v1['F1']
print(f"  {'Precision':<18} {v1['P']:<12.3f} {P:<12.3f} {fp16['P']:<12.3f} {delta_P:+.3f}")
print(f"  {'Recall':<18} {v1['R']:<12.3f} {R:<12.3f} {fp16['R']:<12.3f} {delta_R:+.3f}")
print(f"  {'F1':<18} {v1['F1']:<12.3f} {F1:<12.3f} {fp16['F1']:<12.3f} {delta_F1:+.3f}")

print(f"\n  各 KITTI 类别召回率")
print(f"  {'类别':<18} {'v1':<10} {'v2':<10} {'FP16':<10} {'v2 vs v1'}")
classes = sorted(cls_stat.keys())
for cls in classes:
    s = cls_stat[cls]
    r2 = s['hit']/s['gt'] if s['gt']>0 else 0
    r1 = v1['cls'].get(cls, 0)
    rfp16 = fp16['cls'].get(cls, 0)
    print(f"  {cls:<18} {r1:.1%}     {r2:.1%}     {rfp16:.1%}     {r2-r1:+.1%}")

print(f"\n  按难度分桶召回率")
print(f"  {'难度':<12} {'v1':<10} {'v2':<10} {'FP16':<10} {'v2 vs v1'}")
for tag in ['normal','small','occluded','far']:
    s = diff_stat.get(tag, {'gt':0,'hit':0})
    r2 = s['hit']/s['gt'] if s['gt']>0 else 0
    r1 = v1['diff'].get(tag, 0)
    rfp16 = fp16['diff'].get(tag, 0)
    print(f"  {tag:<12} {r1:.1%}     {r2:.1%}     {rfp16:.1%}     {r2-r1:+.1%}")

# 保存
import json
with open('/workspace/outputs/int8_v2_results.json','w') as f:
    json.dump({'TP':TP,'FP':FP,'FN':FN,'P':P,'R':R,'F1':F1,
               'cls':{k:dict(v) for k,v in cls_stat.items()},
               'diff':{k:dict(v) for k,v in diff_stat.items()}}, f, indent=2)
