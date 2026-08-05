# obstacle — 正前方最近障碍物距离

`plugins/obstacle.py` 单文件插件（TOOLS name / MCP 工具名固定为
`obstacle`，跟评测框架 `OBSTACLE_PLUGIN=obstacle` 的约定一致）。`provider=local`
时用 `LocalDistanceAdapter`：Depth Anything V2 Small（metric depth，ONNX，
int8 动态量化，室内/室外各一个 checkpoint，各 ~26MB）做真实单目深度推理，
替换原来返回 `random.uniform` 的占位实现。`ROS2` 订阅/发布逻辑、`TOOLS`
schema、`dispatch()` 的 action 分支均未改动；`_build_distance_adapter()`
里 `provider == 'local'` 分支的调用方式（`LocalDistanceAdapter(cfg.get('model_path'))`）
也未改动，只重写了 `LocalDistanceAdapter` 内部实现。

## 场景域判定（indoor / outdoor）

用图片容器格式的 magic bytes：PNG → indoor，JPEG → outdoor（对应榜单
输入说明"png 等效焦距 29mm=室内 / jpg 等效焦距 33mm=室外"，室内机器人
和无人车数据集采集时格式从未混用过，格式本身就是场景标签，不需要单独
训练分类器）。magic bytes 判不出来时退回文件名后缀，两者都判不出来时
默认 `indoor` 并记录日志——室内模型量程小（~0-20m），误用在室外图像上
对近距离读数的影响小于反过来，是更保守的兜底选择。

独立成模块级函数 `_detect_scene_domain(image_bytes, filename=None)`，
不需要加载模型就能单测。

## ROI 裁剪

`_compute_roi_bounds(height, width)`：列方向中间 1/3（`[W/3, 2W/3)`），
行方向上方 5/8（`[0, 5H/8)`），跟榜单文档给出的室内 640×480 参考 ROI
（col 213~426, row 0~300）完全一致（213/640≈1/3，300/480=5/8），用比例
表示，按实际输入分辨率换算。outdoor 复用同一比例——榜单文档没有给
outdoor 定义图像空间 ROI（其真值基于 3D 标注框而非图像裁剪+百分位数），
不自行发明一个宽/窄框。

预处理把图像整体 resize 成模型固定输入尺寸 518×518（不保持长宽比），
因为是各轴独立拉伸，原图和 518×518 深度图上的行/列比例边界是等价的，
所以 ROI 直接在深度图（518×518）尺寸上按比例计算，不需要额外坐标换算。

## 稳健取值（百分位数，不是 min）

`_robust_nearest_distance(depth_roi, percentile=1.0)`：先用 5×5 中值
滤波去掉单目深度模型在物体边缘/深度不连续处常见的孤立噪声像素（对真实
连续物体表面几乎没有副作用），再取 P1 分位数（不是严格 `min()`）作为
"最近障碍物距离"——单个噪声点极易把 `min()` 直接带偏，分位数对滤波后
仍存在的分布尾部更鲁棒。ROI 为空或没有任何有限正值时返回 `None`，由
调用方决定 fallback。

## 推理失败与超时兜底

- 图像解码失败、模型未加载、ROI 内没有有效像素等任何异常路径，均
  `try/except` 捕获并返回常量 fallback `{"pred_distance": 10.0}`，记录
  `log.error`/`log.warning`，绝不让异常穿透到 `_inference_worker` 的
  `except Exception` 导致整帧丢弃且无输出。
- 单帧推理跑在专用 `ThreadPoolExecutor` 里，`estimate()` 用
  `future.result(timeout=2.7)` 加硬超时保护：`onnxruntime.run()` 是一次
  同步 C 调用，无法从外部中途打断，用线程池 + timeout 是唯一能保证接口
  在预算内返回结果的办法（超时后台线程会继续跑完，结果被丢弃，不会
  无限堆积——线程池大小固定为 2，多帧同时超时时旧的排队请求仍受限）。
  2.7s 是 3s 硬约束减去给 ROS2 序列化/发布的安全边际。

## 性能

- 模型只在 `LocalDistanceAdapter.__init__` 里加载一次（两个 onnxruntime
  session 常驻实例属性），不会每次 `estimate()` 重新加载。
- Provider 优先 `CUDAExecutionProvider`，不可用时（本地实测：量化模型的
  `ConvInteger`/`MatMulInteger` 算子在部分 CUDA EP 版本上支持不稳定，或
  环境本身缺 CUDA 运行库）自动回退 `CPUExecutionProvider`，加载完成后
  用 `session.get_providers()` 记录实际生效的 provider，不假设配置的
  provider 一定生效。CPU 推理时"GPU 占用 < 10%"这条约束天然满足。
  可用 `OBSTACLE_LOCAL_DEVICE=cpu|cuda|auto` 环境变量强制指定。
- 加载时检查模型文件大小，超过 30MB 预算会 `log.warning`（当前两个
  int8 量化模型各 ~26MB，在预算内）。
- 本地实测（这台开发机，CPU provider）单帧端到端（含场景判定+预处理+
  推理+ROI/百分位后处理）约 300-400ms，在 3 秒预算内有明显余量。
  **已知局限**：这只是开发机的实测数据，不代表 Jetson Orin 真实硬件下
  的表现；`export_onnx.py`（历史开发脚本，见 `plugins/obstacle_distance/`
  遗留目录）之前记录过真实评测环境下 indoor/outdoor 单帧延迟分别约
  20s/40s、远超预算，本地小图测试量不出这个问题，怀疑瓶颈在 Jetson 实际
  算力/onnxruntime CPU 推理/ROS2-Docker 开销，而非这次改写的这几个函数
  本身——上线前应该在真实 Jetson 硬件上单独 profile 确认。

## 2026-08-05 精度调优尝试记录（没有 GT 数据，凭第一性原理推理）

用户反馈上一版模型（`plugins/obstacle_distance/` 遗留包，跟这次 `plugins/obstacle.py`
用的是同一个模型+同一套核心逻辑）在真实榜单上效果不好，要求"调硬一点"。
本地没有 GT 数据能验证任何改动，只能靠推理 + 合成数据自测，记录下来
避免以后重复踩同样的坑：

**outdoor（F1@1m 卡死在 0.0000）**：翻了 `plugins/obstacle_distance/`
的 git 历史，发现能想到的两类改动都已经用真实评测数据试过、且都没有
让 F1@1m 从 0.0000 变化过：
1. ROI 形状——原始版本（col 1/3~2/3, row 1/4~7/8）、极宽版本（col
   1/12~11/12, row 1/8~1）、跟 indoor 完全一致的版本（当前这版用的），
   三种形状实测 F1@1m 全部精确等于 0.0000。
2. 数值矫正——纯保险杠偏移常数、偏移+针孔相机横向投影，两种矫正实测
   F1@1m 依旧是 0.0000，后者还让 RMSE 从 4.42 涨到 9.45（负优化）。

三种 ROI 形状 + 两种数值矫正都不能让这个指标偏离 0，说明问题大概率不是
"ROI 框歪了"或"参考点没对齐"这类流水线层面能修的问题，更可能是模型
本身（在合成数据集 VKITTI2 上训练）在真实照片上的尺度/量程本身就有
系统性偏差，是模型能力上限，不是后处理能调出来的。没有再造第四版 ROI
猜测——已知的三种形状已经排除了"换个框"这个方向，继续猜大概率是浪费
时间。这次没有改动 outdoor 的距离计算逻辑，维持当前 RMSE 最优（4.42）
的纯深度值版本。

**indoor（F1@1m ~0.5，同类提交 ~0.9）**：这个方向历史上从来没有做过
专门实验（只有一版对 indoor/outdoor 都通用、且从未验证过收益的 RANSAC
地面剔除），是唯一还有空间的方向。尝试过一个改动：把 `_robust_nearest_distance`
从"取 P1 这一个百分位点"改成"取 [P1, P1+4] 这一段区间的均值"，指望
进一步压制个别残留像素的影响。写了合成数据测试后发现这个方向是错的：
占 ROI 面积 1%~5% 之间的真实小尺寸障碍物（门框边缘、家具细腿这个量级）
会被区间里混进来的背景像素拉平，读数从正确的"近"变成错误的"远"——
这不是"更保守"，是净负向，用合成数据就能证伪，所以没有采用，保留原始
单点百分位数实现（见 `_robust_nearest_distance` 函数上方注释）。

**结论**：在没有真实 GT/评测反馈的情况下，能想到的"流水线调参"方向
大部分要么已经被历史数据证伪，要么自测证伪。继续调整现在这份代码风险
是"看起来做了努力但实际是噪声"。下一步真正有效的路径是（任选其一）：
拿到哪怕几个真实 case 的 GT 或评测详情（`detailed_cases.json` 之类），
或者接受当前实现重新提交一次拿到新的 F1/RMSE 反馈再迭代，或者重新
评估是否要换模型（这次按要求保留了 Depth-Anything-V2，没有做这个）。

## 已知局限（诚实说明近似之处）

- outdoor 场景直接用 ROI 内深度分位数近似"最近障碍物距离"，没有做
  "车头保险杠水平面距离"换算（榜单对无人车场景的真值定义是 3D 标注框
  到车头保险杠的水平距离，且排除静态杆柱/地面等类别；单目深度图本身
  给不出目标类别信息，换算依赖的相机-车身标定/焦距估计如果不准，误差
  会按最近像素离图像中心的距离成比例放大——历史遗留的 `obstacle_distance`
  包里实测过一版换算，RMSE 从 4.42 涨到 9.45，是负优化，故这次不引入）。
- indoor/outdoor 用完全相同的 ROI 比例，outdoor 没有独立验证过这个比例
  是否最优，只是复用了榜单文档唯一明确给出过的室内比例。
- 没有做地面平面剔除（RANSAC 等）：历史实现加过这一层，但没有 GT 数据
  验证过实际收益，且当前更优先解决的是延迟预算问题，这次先不引入
  额外复杂度，符合"没有实质收益就不过度设计"的原则。
- 场景域判定完全依赖图片容器格式（PNG/JPEG），如果调用方传入的图片被
  转码过（比如 JPEG 转 PNG 保存），判定会不准——目前没有更可靠的备选
  信号（EXIF 焦距字段本质是同一个假设的另一种表现形式，不算独立验证）。

## 本地验证

```bash
cd perception
python3 test_obstacle.py /path/to/image.png /path/to/image2.jpg
# -> {"image": "...", "domain": "indoor", "pred_distance": 1.7125, "elapsed_ms": 377.5}
```

`test_obstacle.py` 不依赖 ROS2（rclpy 缺失时会自动打桩，只是为了绕过
`plugins/obstacle.py` 顶部的模块级 rclpy import，`LocalDistanceAdapter`
本身完全不用 rclpy），方便在没装 ROS2 的开发机上跑。仓库里目前没有带
已知 GT 的示例图片文件，提交榜单前建议自己找几张已知距离的图片跑一遍
做 sanity check（本地用 `test_nameplate.png` 冒烟测试，indoor 域输出
1.7125m，量级合理，但那张图不是榜单示例图，不能当 GT 用）。

## 配置

`perception/config.yaml`：

```yaml
plugins:
  obstacle:
    enabled: true
    provider: local
    model_path: ""   # 留空用仓库自带 perception/models/depth_anything_v2/
```

`provider=local` 分支的构造调用方式固定不变，运行时设备/百分位数用环境
变量调，不走 configSchema：`OBSTACLE_LOCAL_MODEL_DIR`、
`OBSTACLE_LOCAL_DEVICE`（`auto`|`cpu`|`cuda`）、`OBSTACLE_LOCAL_PERCENTILE`。

## 关于 `plugins/obstacle_distance/` 目录

这次改动前，仓库里已经有一个独立的 `plugins/obstacle_distance/` 包
（`main.py` 之前 `from plugins.obstacle_distance import ObstacleDistancePlugin`
加载它），实现思路更完整（额外做了 RANSAC 地面剔除、outdoor 换算实验等），
`README.md` 记录了大量真实评测迭代过程。这次按要求把逻辑整合进单文件
`plugins/obstacle.py`（并把 `main.py`/`config.yaml` 的加载入口切过来），
`obstacle_distance/` 目录本身没有删除，暂时保留供参考，但已不再被
`main.py` 加载。是否需要清理，需要确认后再处理。
