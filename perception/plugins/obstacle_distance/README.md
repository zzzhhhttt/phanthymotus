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

**去噪 + ROI + 最近距离**：先对整张深度图做一次中值滤波（`roi.py` 里
`_denoise_depth_map()`，size=5），去掉单目深度模型在物体边缘/深度不
连续处常见的"孤立噪声像素"——ROI 面积小的时候，百分位数（尤其 P1）
很容易被一两个噪声点直接带偏，中值滤波是这类问题的标准解法，对真实
物体表面（本身是连续一片）几乎没有副作用。滤波之后再取 ROI 内深度
分布的 P1 分位数（不是严格 min()，对滤波后仍存在的分布尾部更鲁棒）
作为"最近障碍物距离"。2026-08-05 加的这一层，动机是室内实测 F1@1m
只有约 0.5（同类提交里有人到了 0.9），怀疑噪声鲁棒性不够是其中一个
可优化点；没有 GT 数据验证过实际提升多少，效果需要跑一次真实评测确认。
见 `roi.py`：

**地面平面剔除**：中值滤波+百分位数之后，再加一层可选的 RANSAC 地面
平面剔除（`roi.py` 里 `_ground_removal_mask()`，`predict.py` 里传
`focal_length_px` 触发）。动机：ROI 已经通过行范围排除了大部分地面
（indoor/outdoor 都是 row 0~5/8），但如果房间比较空旷、相机角度偏低，
ROI 内残留的一小片地面仍可能被误判成"最近障碍物"——这一层是给这种
残留情况加的第二层保护，不是替代 ROI。

实现上把 ROI 内每个像素用针孔相机模型反投影成 3D 点（需要焦距估计：
indoor 用榜单给的"等效焦距 29mm"按标准换算公式估算，outdoor 复用已有
的 nuScenes CAM_FRONT 标定值），RANSAC 拟合平面后，**只有法向量接近
"竖直"方向的平面才判定为地面**——这一步是必须的，因为 ROI 里如果背景
是一整面墙（榜单室内示例图那种场景），法向量是接近"水平"（正对镜头）
的，如果不做这个区分，会把真实的墙面障碍物错当成地面整个剔除掉。
拟合不出足够置信的平面、或者平面更像墙不像地面，一律不剔除——宁可
漏判残留地面，也不能误删真实障碍物。

RANSAC 的候选平面只从 ROI 下方 40% 的区域采样（物理上地面更可能出现
在画面下方），但收集内点时用整个 ROI——这是实测出来的一个坑：如果
纯随机采样，画面里如果同时有一大片背景墙和一小片地面，两个都是完美
平面，随机采样很容易先撞上面积更大的墙，RANSAC 会拟合出墙这个平面
（虽然会被法向量检查正确拒绝，不会误删），但也就此错过了真正的地面，
白白抽样一轮、等于没做剔除——这个问题是写了一个模拟地面+背景墙+障碍物
的合成场景单元测试跑出来才发现的，加了采样偏置之后重新测试通过。

- **indoor**：col 1/3~2/3、row 0~5/8 —— 跟榜单文档"6. 距离计算逻辑
  说明 · (一) 室内机器人场景"给出的 640x480 ROI（col 213~426, row
  0~300）完全一致（213/640≈1/3，300/480=5/8），用比例表示所以能适配
  任意分辨率输入。
- **outdoor**：跟 indoor 完全一样，col 1/3~2/3、row 0~5/8。榜单文档没有
  给无人车场景定义图像空间 ROI（它的真值基于 3D OBB 标注，不是图像裁剪
  +百分位数），所以不自己发明一个宽/窄的框，直接复用榜单唯一明确给出
  过的比例。曾经试过一版更宽的裁剪（col 1/12~11/12、row 1/8~1）想解决
  F1@1m 精确等于 0.0000 的问题，但这个改动没有 GT 数据验证过、且引入
  了新风险（row 加宽到接近画面底部容易框进 GT 排除的路面），已经改回
  跟 indoor 一致，见 `roi.py` 里 `outdoor_roi()` 的详细说明。

**outdoor 距离换算（纵深 -> 水平面 2D 距离）**：榜单对 outdoor 的定义是
"车头保险杠到最近障碍物表面的**水平面（x-y 平面）**欧氏距离"，不是单纯
的相机纵深。榜单给的行人示例图能直接看出这个差异——行人跟车头"纵深
方向基本对齐"，2.498m 这个距离主要来自**横向**偏移（车头偏右一点），
如果只用纵深会把这类 case 算成完全不同（大概率大得多）的错误值，而且
没有任何常数能矫正这种错误（横向偏移和纵深是两个独立变量，不是固定
比例关系）。

`predict.py` 里对 outdoor 用针孔相机模型把纵深换算成水平面 2D 距离：
1. `roi.py` 的 `nearest_obstacle_pixel()` 除了返回最近距离，还返回那个
   像素在图像里的列位置
2. 用这个列位置 + 深度值 + 相机内参（焦距 fx、主点 cx），反推这个点
   相对相机光轴的横向偏移 X（标准针孔投影公式 `X = (col - cx) * Z / fx`）
3. 保险杠偏移矫正（1.7m）应用在纵深分量上，然后 `sqrt(X² + Z_矫正²)`
   作为最终水平面距离

相机内参用的是 nuScenes CAM_FRONT 公开数据里的典型标定值（fx≈1266.4，
1600px 宽下），不是这批评测数据集实测标定出来的精确值，运行时按实际
图片宽度等比缩放。indoor 场景不做这个换算——榜单对 indoor 的定义就是
纯纵深（"沿相机光轴方向的深度"），直接用深度值即可。

2026-08-03/04 实测：先后试过换 ROI（窄/宽）、加纯纵深的常数矫正，
F1@1m 都精确等于 0.0000、完全没有变化——这个"始终不变"的现象不太像
"参数没调准"，更符合"有一部分case在测量一个错误的物理量（纵深而非
水平距离），常数矫正对这类 case 无效"这个假设，这也是加这个针孔换算
的直接动机。这个方法本身仍然是近似（没有真实的 3D 目标检测和 OBB
朝向信息，只能假设最近点就是障碍物朝向本车的那个面），效果还没有用
真实评测结果验证过。

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
2. 是相机光心的直线/深度距离，不是严格的车头保险杠水平距离。这两者
   不只是"数值上略有误差"——2026-08-03 实测两次提交（分别用窄 ROI 和
   宽 ROI）F1@1m 都精确等于 0.0000，换 ROI 完全没用，说明问题不是
   "漏检"而是**参考点系统性偏移**：这批数据是 nuScenes（文件名带
   `CAM_FRONT`），nuScenes 自车坐标系原点在后轴中心，`CAM_FRONT`
   标定平移量约 x≈1.70m，比保险杠(x=3.412m)靠后约 1.7m——保险杠比
   相机更靠前、离前方障碍物更近，相机测出来的距离会系统性地比真值大
   出这个偏移量，导致真实距保险杠 <1m 的 case 相机测出来变成 >1m，
   永远判不成 Positive。已在 `predict.py` 里对 outdoor 结果减去
   `_OUTDOOR_CAMERA_TO_BUMPER_OFFSET_M`（默认 1.7m，根据 nuScenes 公开
   标定数据估的近似值，不是这批评测集实测标定出来的精确值）做矫正，
   效果还没有用真实评测结果验证过，如果矫正后 F1 还是不对，第一个该
   调的就是这个常数。

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
