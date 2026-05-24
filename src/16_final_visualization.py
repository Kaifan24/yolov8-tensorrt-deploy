"""生成最终对比图表 (纯英文,避免字体问题)"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("/workspace/outputs/charts", exist_ok=True)

labels = ['FP32', 'FP16', 'INT8 v1\n(noise calib)', 'INT8 v2\n(KITTI calib)']
precision = [0.559, 0.559, 0.864, 0.902]
recall = [0.645, 0.644, 0.298, 0.050]
f1 = [0.599, 0.599, 0.443, 0.095]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

# 图1: 三指标对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, data, ylabel, title in zip(
    axes, [precision, recall, f1],
    ['Precision', 'Recall', 'F1 Score'],
    ['Precision (higher = fewer false positives)',
     'Recall (higher = fewer missed detections)',
     'F1 (overall accuracy)']):
    bars = ax.bar(labels, data, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, data):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('YOLOv8n on KITTI 7481 frames: FP32 vs FP16 vs INT8',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/workspace/outputs/charts/accuracy_comparison.png', dpi=150, bbox_inches='tight')
print("  saved: accuracy_comparison.png")

# 图2: 难度分桶
difficulty = ['Normal', 'Small\n(<32px)', 'Occluded\n(level>=2)', 'Far\n(>50m)']
fp32_d = [0.788, 0.521, 0.415, 0.175]
fp16_d = [0.788, 0.521, 0.415, 0.164]
int8v1_d = [0.470, 0.048, 0.132, 0.000]
int8v2_d = [0.083, 0.000, 0.020, 0.000]

x = np.arange(len(difficulty)); width = 0.2
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 1.5*width, fp32_d, width, label='FP32', color='#2196F3')
ax.bar(x - 0.5*width, fp16_d, width, label='FP16', color='#4CAF50')
ax.bar(x + 0.5*width, int8v1_d, width, label='INT8 v1 (noise)', color='#FF9800')
ax.bar(x + 1.5*width, int8v2_d, width, label='INT8 v2 (KITTI)', color='#F44336')
ax.set_xticks(x); ax.set_xticklabels(difficulty)
ax.set_ylabel('Recall', fontsize=12)
ax.set_title('Recall by Difficulty Bucket: Model failure concentrates on hard scenes',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right'); ax.set_ylim(0, 1.0); ax.grid(axis='y', alpha=0.3)
for i, vals in enumerate(zip(fp32_d, fp16_d, int8v1_d, int8v2_d)):
    for j, v in enumerate(vals):
        ax.text(i + (j-1.5)*width, v+0.015, f'{v:.0%}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('/workspace/outputs/charts/recall_by_difficulty.png', dpi=150, bbox_inches='tight')
print("  saved: recall_by_difficulty.png")

# 图3: 类别召回率
classes = ['Car', 'Pedestrian', 'Van', 'Truck', 'Person_sitting', 'Cyclist']
fp32_c = [0.732, 0.567, 0.422, 0.239, 0.248, 0.037]
fp16_c = [0.732, 0.567, 0.421, 0.236, 0.248, 0.037]
int8v1_c = [0.337, 0.342, 0.112, 0.083, 0.063, 0.004]
int8v2_c = [0.059, 0.043, 0.018, 0.001, 0.000, 0.000]

x = np.arange(len(classes))
fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x - 1.5*width, fp32_c, width, label='FP32', color='#2196F3')
ax.bar(x - 0.5*width, fp16_c, width, label='FP16', color='#4CAF50')
ax.bar(x + 0.5*width, int8v1_c, width, label='INT8 v1', color='#FF9800')
ax.bar(x + 1.5*width, int8v2_c, width, label='INT8 v2', color='#F44336')
ax.set_xticks(x); ax.set_xticklabels(classes)
ax.set_ylabel('Recall', fontsize=12)
ax.set_title('Recall by KITTI Class', fontsize=13, fontweight='bold')
ax.legend(); ax.set_ylim(0, 1.0); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/outputs/charts/recall_by_class.png', dpi=150, bbox_inches='tight')
print("  saved: recall_by_class.png")

# 图4: FP16 free lunch
fig, ax = plt.subplots(figsize=(11, 6))
metrics = ['Latency\n(ms)', 'Precision', 'Recall', 'F1']
fp32_v = [87.7, 0.559, 0.645, 0.599]
fp16_v = [71.4, 0.559, 0.644, 0.599]
fp16_norm = [v/f for v, f in zip(fp16_v, fp32_v)]
delta = [(f - 1.0) * 100 for f in fp16_norm]
x = np.arange(len(metrics))
bars = ax.bar(x, delta, color=['#4CAF50' if d>=0 else '#F44336' for d in delta],
              edgecolor='white', linewidth=1.5)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel('FP16 vs FP32 (%)', fontsize=12)
ax.set_title('FP16 = Free Lunch: 18% faster, ~0% accuracy loss',
             fontsize=13, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)
for bar, val in zip(bars, delta):
    y = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, y + (0.5 if y >= 0 else -1.5),
            f'{val:+.1f}%', ha='center', fontweight='bold')
ax.grid(axis='y', alpha=0.3); ax.set_ylim(-25, 5)
plt.tight_layout()
plt.savefig('/workspace/outputs/charts/fp16_free_lunch.png', dpi=150, bbox_inches='tight')
print("  saved: fp16_free_lunch.png")

print("\n全部4张图重新生成完毕")
