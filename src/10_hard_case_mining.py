"""
KITTI 长尾 Corner Case 自动挖掘
定义"难例"的5个维度,给每帧打分,自动找出最难的帧
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# ============================================================
# 第一步: 读取全部标注(和01脚本一样的逻辑)
# ============================================================

label_dir = "/workspace/data/training/label_2"
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

# 按帧组织数据: frame_data[帧号] = [该帧所有目标的信息列表]
frame_data = {}

for fname in label_files:
    frame_id = fname.replace('.txt', '')
    filepath = os.path.join(label_dir, fname)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    objects = []
    for line in lines:
        parts = line.strip().split()
        cls = parts[0]
        if cls in ['DontCare', 'Misc']:
            continue
        
        objects.append({
            'class': cls,
            'truncation': float(parts[1]),
            'occlusion': int(parts[2]),
            'x1': float(parts[4]), 'y1': float(parts[5]),
            'x2': float(parts[6]), 'y2': float(parts[7]),
            'height': float(parts[7]) - float(parts[5]),
            'distance': float(parts[13]),
        })
    
    frame_data[frame_id] = objects

print(f"总帧数: {len(frame_data)}")

# ============================================================
# 第二步: 定义"难度评分"规则
# ============================================================

# 每帧根据5个维度打分,分数越高越"难"
# 这就是数据挖掘的核心逻辑——把"什么是难例"形式化成可计算的规则

def score_frame(objects):
    """
    给一帧打难度分,返回总分和各维度得分
    """
    if len(objects) == 0:
        return 0, {}
    
    scores = {}
    
    # 维度1: 稀有类别(不是Car/Van的目标越多越难)
    rare_classes = ['Cyclist', 'Tram', 'Person_sitting']
    rare_count = sum(1 for obj in objects if obj['class'] in rare_classes)
    scores['rare_class'] = rare_count * 3  # 权重3,因为稀有类别很重要

    # 维度2: 小目标(高度<32像素)
    small_count = sum(1 for obj in objects if obj['height'] < 32)
    scores['small_obj'] = small_count * 2  # 权重2
    
    # 维度3: 严重遮挡(遮挡>=2)
    occluded_count = sum(1 for obj in objects if obj['occlusion'] >= 2)
    scores['occlusion'] = occluded_count * 2
    
    # 维度4: 严重截断(>0.5)
    truncated_count = sum(1 for obj in objects if obj['truncation'] > 0.5)
    scores['truncation'] = truncated_count * 1
    
    # 维度5: 远距离目标(>50米)
    far_count = sum(1 for obj in objects if obj['distance'] > 50)
    scores['far_distance'] = far_count * 2
    
    # 维度6: 目标密集(目标多的场景更复杂)
    if len(objects) >= 15:
        scores['crowded'] = 5
    elif len(objects) >= 10:
        scores['crowded'] = 2
    else:
        scores['crowded'] = 0
    
    total = sum(scores.values())
    return total, scores

# ============================================================
# 第三步: 给每帧打分
# ============================================================

print("\n" + "=" * 50)
print("给每帧打难度分...")
print("=" * 50)

frame_scores = {}
for frame_id, objects in frame_data.items():
    total, detail = score_frame(objects)
    frame_scores[frame_id] = {
        'total': total,
        'detail': detail,
        'num_objects': len(objects),
    }

# 按总分排序
sorted_frames = sorted(frame_scores.items(), key=lambda x: x[1]['total'], reverse=True)

# 打印最难的10帧
print("\n最难的10帧(难度分最高):")
print(f"  {'帧号':<10} {'总分':<8} {'目标数':<8} {'难度来源'}")
print(f"  {'-'*65}")
for frame_id, info in sorted_frames[:10]:
    # 找出得分>0的维度
    sources = [f"{k}={v}" for k, v in info['detail'].items() if v > 0]
    print(f"  {frame_id:<10} {info['total']:<8} {info['num_objects']:<8} {', '.join(sources)}")

# 打印最简单的5帧(作为对比)
print("\n最简单的5帧:")
for frame_id, info in sorted_frames[-5:]:
    print(f"  {frame_id:<10} 总分={info['total']:<8} 目标数={info['num_objects']}")

# ============================================================
# 第四步: 统计难例分布
# ============================================================

print("\n" + "=" * 50)
print("难例统计")
print("=" * 50)

all_scores = [info['total'] for info in frame_scores.values()]

# 定义难度等级
easy = sum(1 for s in all_scores if s == 0)
medium = sum(1 for s in all_scores if 1 <= s <= 5)
hard = sum(1 for s in all_scores if 6 <= s <= 15)
very_hard = sum(1 for s in all_scores if s > 15)

total_frames = len(all_scores)
print(f"  简单 (分数=0):      {easy:>5}帧  ({easy/total_frames*100:.1f}%)")
print(f"  中等 (分数1-5):     {medium:>5}帧  ({medium/total_frames*100:.1f}%)")
print(f"  困难 (分数6-15):    {hard:>5}帧  ({hard/total_frames*100:.1f}%)")
print(f"  极难 (分数>15):     {very_hard:>5}帧  ({very_hard/total_frames*100:.1f}%)")

# ============================================================
# 第五步: 各难度维度贡献分析
# ============================================================

print("\n" + "=" * 50)
print("各难度维度贡献(哪个因素导致最多难例)")
print("=" * 50)

dimension_totals = defaultdict(int)
for info in frame_scores.values():
    for dim, val in info['detail'].items():
        if val > 0:
            dimension_totals[dim] += 1  # 有多少帧在这个维度上得分>0

for dim, count in sorted(dimension_totals.items(), key=lambda x: x[1], reverse=True):
    dim_labels = {
        'rare_class': '稀有类别',
        'small_obj': '小目标',
        'occlusion': '严重遮挡',
        'truncation': '严重截断',
        'far_distance': '远距离',
        'crowded': '密集场景',
    }
    print(f"  {dim_labels.get(dim, dim):<12} 影响 {count:>5} 帧  ({count/total_frames*100:.1f}%)")

# ============================================================
# 第六步: 可视化
# ============================================================

print("\n" + "=" * 50)
print("生成可视化图表")
print("=" * 50)

# 图1: 难度分数分布直方图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(all_scores, bins=30, edgecolor='white', color='#E91E63', alpha=0.8)
ax.set_xlabel('Difficulty Score', fontsize=12)
ax.set_ylabel('Number of Frames', fontsize=12)
ax.set_title('Frame Difficulty Score Distribution', fontsize=14, fontweight='bold')
ax.axvline(np.mean(all_scores), color='blue', linestyle='--', 
           label=f'Mean: {np.mean(all_scores):.1f}')
ax.axvline(np.percentile(all_scores, 90), color='red', linestyle='--',
           label=f'90th percentile: {np.percentile(all_scores, 90):.0f}')
ax.legend()

# 图2: 难度等级饼图
ax = axes[1]
level_labels = ['Easy (0)', f'Medium (1-5)\n{medium}', f'Hard (6-15)\n{hard}', f'Very Hard (>15)\n{very_hard}']
level_vals = [easy, medium, hard, very_hard]
level_colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
wedges, texts, autotexts = ax.pie(level_vals, labels=['Easy','Medium','Hard','Very Hard'],
                                   colors=level_colors, autopct='%1.1f%%',
                                   startangle=90, textprops={'fontsize': 11})
ax.set_title('Frame Difficulty Level Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('/workspace/outputs/charts/difficulty_analysis.png', dpi=150)
print("  已保存: difficulty_analysis.png")

# 图3: 各维度贡献柱状图
fig, ax = plt.subplots(figsize=(10, 5))
dim_names = ['Small\nObjects', 'Heavy\nOcclusion', 'Far\nDistance', 'Rare\nClasses', 'Severe\nTruncation', 'Crowded\nScene']
dim_keys = ['small_obj', 'occlusion', 'far_distance', 'rare_class', 'truncation', 'crowded']
dim_vals = [dimension_totals.get(k, 0) for k in dim_keys]
dim_colors = ['#F44336', '#FF9800', '#9C27B0', '#2196F3', '#607D8B', '#009688']
bars = ax.bar(dim_names, dim_vals, color=dim_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, dim_vals):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+30,
            f'{val}', ha='center', fontweight='bold')
ax.set_ylabel('Number of Affected Frames', fontsize=12)
ax.set_title('Difficulty Dimension Contribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/outputs/charts/dimension_contribution.png', dpi=150)
print("  已保存: dimension_contribution.png")

# ============================================================
# 第七步: 导出难例帧列表(供算法工程师使用)
# ============================================================

# 把最难的帧导出成文件,这就是"数据挖掘的产出物"
hard_frames_file = "/workspace/outputs/hard_frames_top200.txt"
with open(hard_frames_file, 'w') as f:
    f.write("# KITTI Hard Frame List (Top 200 by difficulty score)\n")
    f.write("# Format: frame_id, score, num_objects, difficulty_sources\n")
    for frame_id, info in sorted_frames[:200]:
        sources = "|".join(f"{k}={v}" for k, v in info['detail'].items() if v > 0)
        f.write(f"{frame_id},{info['total']},{info['num_objects']},{sources}\n")

print(f"\n  难例帧列表已导出: hard_frames_top200.txt (前200帧)")
print(f"  这个文件可以直接交给算法工程师,让他们重点关注这些难帧")

print("\n" + "=" * 50)
print("全部完成!")
print("=" * 50)
