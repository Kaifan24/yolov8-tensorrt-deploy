"""
KITTI 数据集总览分析
读取全部标注文件,统计:
1. 类别分布(哪些目标最多/最少)
2. 每帧目标数量分布
3. 目标框大小分布(大目标vs小目标)
4. 遮挡/截断程度分布
5. 目标距离分布
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 容器里没有显示器,用这个后端才能保存图片
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# ============================================================
# 第一步: 读取所有标注文件
# ============================================================

label_dir = "/workspace/data/training/label_2"
# os.listdir 列出文件夹里所有文件名
# sorted 排序,确保顺序一致
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
print(f"标注文件数量: {len(label_files)}")

# 用列表存所有目标信息,每个目标是一个字典
all_objects = []
# 用字典存每帧的目标数量
frame_object_counts = {}

for fname in label_files:
    frame_id = fname.replace('.txt', '')  # "000000.txt" -> "000000"
    filepath = os.path.join(label_dir, fname)
    
    # 读取文件的每一行
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    frame_objects = 0  # 这帧里有效目标数量
    
    for line in lines:
        parts = line.strip().split()  # 按空格拆分
        cls = parts[0]  # 第一个字段是类别
        
        # 跳过DontCare和Misc(不关心的标注)
        if cls in ['DontCare', 'Misc']:
            continue
        
        # 解析各字段
        truncation = float(parts[1])   # 截断程度 0~1
        occlusion = int(parts[2])      # 遮挡程度 0/1/2/3
        # parts[3] 是观察角度,跳过
        
        # 框坐标
        x1 = float(parts[4])  # 左上角x
        y1 = float(parts[5])  # 左上角y
        x2 = float(parts[6])  # 右下角x
        y2 = float(parts[7])  # 右下角y
        
        # 计算框的宽和高(像素)
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height  # 框面积(像素²)
        
        # 目标距离(z坐标,第13个字段,单位米)
        distance = float(parts[13])
        
        # 存起来
        all_objects.append({
            'frame': frame_id,
            'class': cls,
            'truncation': truncation,
            'occlusion': occlusion,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'width': box_width,
            'height': box_height,
            'area': box_area,
            'distance': distance,
        })
        
        frame_objects += 1
    
    frame_object_counts[frame_id] = frame_objects

print(f"有效目标总数: {len(all_objects)}")
print(f"平均每帧目标数: {np.mean(list(frame_object_counts.values())):.1f}")

# ============================================================
# 第二步: 类别分布
# ============================================================

print("\n" + "=" * 50)
print("类别分布")
print("=" * 50)

class_counts = Counter([obj['class'] for obj in all_objects])
# most_common() 按数量从多到少排序
for cls, count in class_counts.most_common():
    pct = count / len(all_objects) * 100
    print(f"  {cls:<15} {count:>6}  ({pct:.1f}%)")

# 画图
fig, ax = plt.subplots(figsize=(10, 5))
classes = [c for c, _ in class_counts.most_common()]
counts = [n for _, n in class_counts.most_common()]
colors_map = plt.cm.Set3(np.linspace(0, 1, len(classes)))
bars = ax.barh(classes[::-1], counts[::-1], color=colors_map)
ax.set_xlabel('Number of Instances', fontsize=12)
ax.set_title('KITTI Object Class Distribution', fontsize=14, fontweight='bold')
for bar, val in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
            f'{val}', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/outputs/charts/class_distribution.png', dpi=150)
print("  图表已保存: class_distribution.png")

# ============================================================
# 第三步: 每帧目标数量分布
# ============================================================

print("\n" + "=" * 50)
print("每帧目标数量分布")
print("=" * 50)

counts_list = list(frame_object_counts.values())
print(f"  最少: {min(counts_list)} 个目标")
print(f"  最多: {max(counts_list)} 个目标")
print(f"  中位数: {np.median(counts_list):.0f}")
print(f"  空帧(0个目标): {counts_list.count(0)} 帧")

# 找到目标最多的5帧
sorted_frames = sorted(frame_object_counts.items(), key=lambda x: x[1], reverse=True)
print(f"\n  目标最密集的5帧:")
for fid, cnt in sorted_frames[:5]:
    print(f"    帧 {fid}: {cnt} 个目标")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(counts_list, bins=range(0, max(counts_list)+2), edgecolor='white', color='#2196F3', alpha=0.8)
ax.set_xlabel('Number of Objects per Frame', fontsize=12)
ax.set_ylabel('Number of Frames', fontsize=12)
ax.set_title('Distribution of Objects per Frame', fontsize=14, fontweight='bold')
ax.axvline(np.mean(counts_list), color='red', linestyle='--', label=f'Mean: {np.mean(counts_list):.1f}')
ax.legend()
plt.tight_layout()
plt.savefig('/workspace/outputs/charts/objects_per_frame.png', dpi=150)
print("  图表已保存: objects_per_frame.png")

# ============================================================
# 第四步: 目标框大小分布(找小目标——模型最容易漏检的)
# ============================================================

print("\n" + "=" * 50)
print("目标框大小分布")
print("=" * 50)

areas = [obj['area'] for obj in all_objects]
heights = [obj['height'] for obj in all_objects]

# 按COCO标准分类: 小目标<32², 中目标<96², 大目标>=96²
small = sum(1 for h in heights if h < 32)
medium = sum(1 for h in heights if 32 <= h < 96)
large = sum(1 for h in heights if h >= 96)
total = len(heights)

print(f"  小目标 (高度<32px):  {small:>5}  ({small/total*100:.1f}%)")
print(f"  中目标 (32-96px):    {medium:>5}  ({medium/total*100:.1f}%)")
print(f"  大目标 (高度>=96px): {large:>5}  ({large/total*100:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 5))
sizes = ['Small\n(<32px)', 'Medium\n(32-96px)', 'Large\n(>=96px)']
vals = [small, medium, large]
colors_size = ['#F44336', '#FF9800', '#4CAF50']
bars = ax.bar(sizes, vals, color=colors_size, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+50,
            f'{val}\n({val/total*100:.1f}%)', ha='center', fontweight='bold')
ax.set_ylabel('Number of Objects', fontsize=12)
ax.set_title('Object Size Distribution (by height)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/outputs/charts/size_distribution.png', dpi=150)
print("  图表已保存: size_distribution.png")

# ============================================================
# 第五步: 遮挡和截断分布(这些是模型容易出错的"难例")
# ============================================================

print("\n" + "=" * 50)
print("遮挡/截断分布")
print("=" * 50)

occ_counts = Counter([obj['occlusion'] for obj in all_objects])
occ_labels = {0: '完全可见', 1: '部分遮挡', 2: '大面积遮挡', 3: '未知'}
for occ, count in sorted(occ_counts.items()):
    pct = count / len(all_objects) * 100
    print(f"  遮挡={occ} ({occ_labels.get(occ, '?')}): {count:>5}  ({pct:.1f}%)")

trunc_severe = sum(1 for obj in all_objects if obj['truncation'] > 0.5)
print(f"\n  严重截断(>0.5): {trunc_severe}  ({trunc_severe/len(all_objects)*100:.1f}%)")

# ============================================================
# 第六步: 距离分布
# ============================================================

print("\n" + "=" * 50)
print("目标距离分布")
print("=" * 50)

distances = [obj['distance'] for obj in all_objects if obj['distance'] > 0]
print(f"  最近: {min(distances):.1f}m")
print(f"  最远: {max(distances):.1f}m")
print(f"  平均: {np.mean(distances):.1f}m")
print(f"  远距离目标(>50m): {sum(1 for d in distances if d > 50)}  ({sum(1 for d in distances if d > 50)/len(distances)*100:.1f}%)")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(distances, bins=50, edgecolor='white', color='#9C27B0', alpha=0.8)
ax.set_xlabel('Distance (m)', fontsize=12)
ax.set_ylabel('Number of Objects', fontsize=12)
ax.set_title('Object Distance Distribution', fontsize=14, fontweight='bold')
ax.axvline(50, color='red', linestyle='--', label='50m threshold')
ax.legend()
plt.tight_layout()
plt.savefig('/workspace/outputs/charts/distance_distribution.png', dpi=150)
print("  图表已保存: distance_distribution.png")

# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 50)
print("分析总结")
print("=" * 50)
print(f"  总帧数: {len(label_files)}")
print(f"  有效目标总数: {len(all_objects)}")
print(f"  类别数: {len(class_counts)}")
print(f"  最多的类别: {class_counts.most_common(1)[0][0]} ({class_counts.most_common(1)[0][1]})")
print(f"  最少的类别: {class_counts.most_common()[-1][0]} ({class_counts.most_common()[-1][1]})")
print(f"  小目标占比: {small/total*100:.1f}%")
print(f"  严重遮挡(>=2)占比: {(occ_counts.get(2,0)+occ_counts.get(3,0))/total*100:.1f}%")
print(f"  远距离(>50m)占比: {sum(1 for d in distances if d>50)/len(distances)*100:.1f}%")
print(f"\n  潜在难例特征: 小目标 + 严重遮挡 + 远距离")
print(f"  下一步: 02_hard_case_mining.py 自动挖掘这些难例")
