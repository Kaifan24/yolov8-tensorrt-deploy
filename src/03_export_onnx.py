from ultralytics import YOLO
import onnx
import os

print("=" * 50)
print("Step 1: 加载 PyTorch 模型")
print("=" * 50)
model = YOLO("yolov8n.pt")
print(f"模型类型: {type(model.model)}")
print(f"参数量: {sum(p.numel() for p in model.model.parameters()):,}")

print("\n" + "=" * 50)
print("Step 2: 导出 ONNX")
print("=" * 50)
# ultralytics 内置了导出功能,一行搞定
# opset=17 是ONNX的版本,17对TensorRT 10.x兼容最好
# simplify=True 会自动简化计算图(去掉冗余节点)
model.export(
    format="onnx",
    opset=17,
    simplify=True,
    dynamic=False,       # 固定输入尺寸(部署时通常固定,更快)
    imgsz=640,           # 输入图片尺寸
)

# 导出后文件在 yolov8n.onnx(和.pt同目录)
onnx_path = "yolov8n.onnx"
print(f"\nONNX 文件已生成: {onnx_path}")
print(f"文件大小: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")

print("\n" + "=" * 50)
print("Step 3: 验证 ONNX 模型")
print("=" * 50)
# 用 onnx 库检查导出的模型是否合法
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX 模型验证通过 ✅")

# 打印输入输出信息(面试常问:你的模型输入输出shape是什么)
for inp in onnx_model.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  输入: {inp.name}, shape: {shape}")
for out in onnx_model.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  输出: {out.name}, shape: {shape}")

# 把onnx文件复制到workspace让它持久化
import shutil
dst = "/workspace/yolov8n.onnx"
shutil.copy(onnx_path, dst)
print(f"\n已复制到: {dst}")
print("下一步: ONNX → TensorRT (FP32/FP16/INT8)")
