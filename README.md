# YOLOv8 模型部署优化: PyTorch 到 TensorRT

在消费级GPU RTX 4060 8GB 上完成YOLOv8n目标检测模型 PyTorch到ONNX到TensorRT 全流程部署
系统对比 FP32 / FP16 / INT8 三档精度下的推理延迟/吞吐/精度损失

## 项目目标

* 打通模型部署链路: PyTorch到ONNX到TensorRT
* 量化优化: FP32 / FP16 / INT8 性能-精度对比
* 在KITTI数据集上验证部署后检测效果
* 全流程容器化可复现

## 环境

* 镜像: nvcr.io/nvidia/pytorch:24.10-py3
* GPU: NVIDIA RTX 4060 8GB
* TensorRT 10.5.0 / PyTorch 2.5 / Ultralytics 8.4.51

## 进度

* \[x] 环境搭建与推理链路打通
* \[x] PyTorch → ONNX 导出
* \[x] TensorRT FP32/FP16/INT8 三档构建
* \[x] INT8 PTQ校准(含校准器实现)
* \[x] 性能benchmark(延迟/FPS/文件大小对比)
* \[x] 精度验证(检测结果一致性对比)
* \[x] 可视化图表(延迟/FPS/加速比)
* \[ ] 踩坑记录完善与技术博客

## Benchmark 结果

|精度|延迟(ms)|FPS|文件大小(MB)|加速比|
|-|-|-|-|-|
|PyTorch|6.50|153.4|12.3|基线|
|TRT-FP32|3.15|317.2|19.5|2.1x|
|TRT-FP16|2.24|445.6|9.4|2.9x|
|TRT-INT8|2.04|491.3|6.8|3.2x|

## 踩坑记录

详见 docs/troubleshooting.md

## 复现

docker run --gpus all -it --rm -v 项目路径:/workspace --shm-size=8g nvcr.io/nvidia/pytorch:24.10-py3
pip install -r requirements.txt
python src/01\_predict.py

