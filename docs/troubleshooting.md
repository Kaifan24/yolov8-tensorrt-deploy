# 部署环境搭建踩坑记录 
 
## 1. NGC镜像四层依赖地狱 
- ultralytics安装连锁拉入numpy2+opencv4.13,撞镜像基线numpy1.x 
- 降numpy后opencv又强依赖numpy2,形成循环 
- 锁版本后仍报DictValue: 根因是Python优先加载镜像预装的系统级cv2 
- 解法: 隔离系统级cv2+改用headless版+锁版本 
 
## 2. 容器网络与代理作用域 
- 宿主机VPN正常但容器内SSL中断: 容器独立网络命名空间不走宿主TUN隧道 
- 工程结论: 部署环境假设网络受限,关键资源提前固化
