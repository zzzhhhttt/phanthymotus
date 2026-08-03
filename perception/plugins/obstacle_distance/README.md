# obstacle_distance — 正前方最近障碍物距离

独立于 `plugins/ocr.py` / `plugins/ppocr_adapter.py`，代码不共享、不互相
import，可以单独修改、单独测试，不影响 OCR 插件。

对应榜单任务：给一张 png/jpg 图片，输出机器人/车头正前方最近障碍物的
距离（米）：

```json
{"pred_distance": 2.43}
```

## 设计

**模型**：[Depth Anything V2 Small（metric depth 版本）](https://huggingface.co/depth-anything)，
ViT-S backbone，24.8M 参数。原始 fp32 权重单文件 99MB，**超过榜单"模型
30M 以下"的限制**（这里的 30M 指模型文件大小，不是参数量）——导出成
ONNX 后用 onnxruntime 做 INT8 动态量化，压到单文件 **~26MB**，在预算内。
量化后跟 fp32 基线相比，在真实图片上平均绝对深度误差约 0.02m（相对
误差约 1%），可接受。导出/量化过程见 `export_onnx.py`。

有两个针对性微调 checkpoint（已经导出量化好，直接放在仓库里
`perception/models/depth_anything_v2/`）：

| 文件 | 训练数据 | 量程 | 大小 | 用于 |
|---|---|---|---|---|
| `depth_anything_v2_metric_indoor_vits_int8.onnx` | Hypersim | ~0-20m | ~26MB | 室内机器人场景 |
| `depth_anything_v2_metric_outdoor_vits_int8.onnx` | VKITTI2 | ~0-80m | ~26MB | 无人车场景 |

**场景域判定（indoor/outdoor 选哪个 checkpoint）**：直接用图片格式。
榜单输入说明里写了"输入为单张 png（等效焦距 29mm）或 jpg（等效焦距
33mm）图像"——室内机器人数据集和无人车数据集用的是不同相机，采集时
格式就没混用过，所以文件格式本身就是场景域的标签，不需要另外训练一个
分类器。见 `domain.py`（判定用 magic bytes，比后缀名更可靠；也支持调
用方直接传 `domain=` 覆盖）。

**ROI + 最近距离**：对深度图取 ROI，取 ROI 内深度分布的 P1 分位数（不是
严格 min()，对单像素噪声更鲁棒）作为"最近障碍物距离"。见 `roi.py`：

- **indoor**：col 1/3~2/3、row 0~5/8 —— 跟榜单文档"6. 距离计算逻辑
  说明 · (一) 室内机器人场景"给出的 640x480 ROI（col 213~426, row
  0~300）完全一致（213/640≈1/3，300/480=5/8），用比例表示所以能适配
  任意分辨率输入。
- **outdoor**：col 1/3~2/3、row 1/4~7/8（近似，见下面 Limitations）。

## 用法

### 单图推理（对应榜单输入输出契约）

```bash
cd perception
python3 -m plugins.obstacle_distance.predict /path/to/image.png
# -> {"pred_distance": 1.7172}
```

或当库用：

```python
from plugins.obstacle_distance import predict_distance
result = predict_distance(image_bytes, filename="frame.jpg")
```

`predict_distance` / `predict.py` 只依赖 onnxruntime + Pillow + numpy，
**不依赖 rclpy、不依赖 torch/transformers**，本地开发机（没装 ROS2）也
能直接跑，方便离开容器单独调试模型效果。

### 本地复现榜单评测指标（提交前自测）

```bash
python3 -m plugins.obstacle_distance.eval manifest.csv
```

`manifest.csv`（表头必须含 `image,gt_distance`，`domain` 可选）：

```csv
image,gt_distance,domain
/data/indoor/0001.png,1.72,indoor
/data/outdoor/0002.jpg,9.07,outdoor
```

输出 Precision@1m / Recall@1m / F1@1m / RMSE / FPS / 失败率，口径对应
榜单文档《4. 评测标准》。

### 接入 Perception Stack（ROS2/MCP，真正喂给 Core 层）

`config.yaml` 里 `plugins.obstacle_distance.enabled: true` 即可加载
（这个 key 只是 config.yaml/main.py 内部的分组名，可以跟 MCP 工具名不
一样）。`main.py` 已经接好了加载逻辑（跟 `ocr` 平级、互不影响）。

**MCP 工具名是 `obstacle`，不是 `obstacle_distance`**——榜单评测框架固定
用环境变量 `OBSTACLE_PLUGIN=obstacle` 去调用，工具名对不上会导致每个
case 直接报 `Unknown tool: obstacle`、全部评测失败（踩过这个坑，
2026-08-03 的 judge flow 日志里能看到）。用法：

```
obstacle(action=start, input_topic=/benchmark/camera/image/xxx)
# 之后持续把 {"pred_distance": ..., "timestamp": ...} 发布到
# /benchmark/camera/image/xxx/obstacle
```

## 模型权重

量化好的 onnx 文件（indoor/outdoor 各 ~26MB）直接放在仓库里：

```
perception/models/depth_anything_v2/depth_anything_v2_metric_indoor_vits_int8.onnx
perception/models/depth_anything_v2/depth_anything_v2_metric_outdoor_vits_int8.onnx
```

`DepthEstimator` 默认就从这个路径加载，本地开发机/Jetson 都不需要联网、
不需要额外下载步骤。`Dockerfile.jetson` 也是直接 `COPY` 这个目录进镜像
（跟 OCR 模型 `COPY perception/models/ppocr/...` 是同一个模式）。

如果要用别的路径（比如挂载 JuiceFS 卷而不是打进镜像本体），设置
`OBSTACLE_DISTANCE_MODEL_DIR` 指向对应目录即可，`config.yaml` 里
`plugins.obstacle_distance.model_dir` 也能配。

需要重新生成这两个 onnx 文件（比如换量化策略）时才用得到
`export_onnx.py`，那是一次性开发脚本，依赖 torch/transformers/onnx，
跟运行时（onnxruntime-only）完全分开，不会污染部署镜像的依赖。

**为什么用 CPU 而不是 GPU 跑**：INT8 动态量化产出的 ConvInteger /
MatMulInteger 算子在 `CUDAExecutionProvider` 上支持不稳定（本地实测
`QInt8` 权重在 CPU EP 上直接报"找不到 ConvInteger 实现"，换成 `QUInt8`
权重才能跑；GPU EP 没有本地 Jetson 设备可验证，风险更高，没有强行走）。
CPU 单帧实测 ~150-500ms（含首次加载模型的开销），远在榜单 3 秒的 FPS
预算内，这也顺带让"GPU 占用 <10%"这条约束不战而胜。如果之后想榨干
Jetson GPU，需要换成静态 QDQ 量化 + TensorRT EP，那条路需要标定数据集
和真实 Jetson 设备验证，这次没做，见 `export_onnx.py` 里的说明。

## Limitations（诚实说明近似之处）

榜单对无人车场景的"真值"定义是：**3D OBB 最近面**到**车头保险杠**
（自车坐标系 x=3.412m）的**水平距离**，且只统计特定动态/可移动物体
类别（车辆/行人/锥桶/护栏等），排除静态杆柱、建筑、地面。这需要完整
的 3D 目标检测 + 类别过滤 + 相机-车身标定，单目深度图本身给不出这些
信息。

本实现对 outdoor 场景用的是"ROI 内最近深度"近似整个"最近障碍物表面
距离"，代价：

1. 无法区分"计入障碍物的类别"和"不计入的静态结构"（比如路灯杆子、
   护栏立柱刚好在 ROI 中心且比真正的车/人更近，会被误判成更近的
   "最近障碍物"，拉低 precision）。
2. 是相机光心的直线/深度距离，不是严格的车头保险杠水平距离（两者在
   物体基本正对前方、俯仰角小的典型行车场景下相差不大，但不是精确
   等价，贡献一部分系统性 RMSE 偏差）。

如果后续要往这个方向提升精度，比较务实的路径是复用仓库里已有的
`plugins/vop.py`（YOLOv8-World 开放词表检测，已经支持 COCO 里的
car/truck/bus/motorcycle/bicycle/person 等相关类别）：检测框跟 ROI
求交、按类别白名单过滤，只在过滤后的像素里取最近深度，能显著降低
"误把静态物体当最近障碍物"的 false positive。这次先不做是因为
YOLOv8-World + CLIP 文本编码器（分别 ~24MB / ~150MB 级别的权重文件）
跟深度模型叠在一起，很容易把 30MB 的模型体积预算打穿，需要先确认
榜单的 30M 限制是算单个模型文件还是整个推理链路的总大小，再决定要不
要往这个方向扩展。

室内场景的 ROI/百分位逻辑是跟榜单文档逐条对齐的，没有这个近似问题。
