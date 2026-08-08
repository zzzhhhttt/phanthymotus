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
- 2026-08-08 起固定只用 `CPUExecutionProvider`，`_resolve_providers()` 里
  CUDA 相关代码整个删掉了，不再有 `OBSTACLE_LOCAL_DEVICE` 这个环境变量
  （见下面对应章节的详细原因：真实评测框架会并发拉起 10 个 obstacle 容器，
  且都带 `--runtime nvidia`，GPU 是真实可用的，没必要留一条能被打开的
  GPU 显存分配路径）。CPU 推理时"GPU 占用 < 10%"这条约束天然满足。
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

## 2026-08-08 收到新的真实榜单反馈，重新加回地面平面剔除

新一次提交的榜单明细：indoor RMSE=0.9203（TN=42/TP=2），outdoor
RMSE=7.2102（TN=12/TP=2），总 Score=23.30、F1=0.233。

重点看 outdoor：7.2102 比历史记录里 outdoor 实测最优的 4.4231 差了
不少。翻旧账确认这两次的路径差异——4.4231 那次的状态是"纯深度值（无
偏移换算）+ RANSAC 地面剔除都在"（`obstacle_distance/predict.py` 旧
注释、commit a72a2ce 明确记录），而这次提交跑的是 consolidate 之后的
`plugins/obstacle.py`（commit 5bfa6c9），distance 计算逻辑没变（同样是
纯深度值），但 RANSAC 地面剔除在 consolidate 时被去掉了（当时的理由是
"没有 GT 数据验证过收益，优先级排在延迟预算之后"，见上面"已知局限"）。
两个版本之间唯一有实质差异的就是这一层——不是凭空猜测一个新方向，是
"去掉这层之后指标变差了"这个具体证据支撑的假设。

已经把 `_ground_removal_mask`/`_ransac_plane`/`_backproject`（原
`obstacle_distance/roi.py` 的实现，逻辑不变，RANSAC 迭代次数沿用
a72a2ce 调过的 50 次）重新加回 `plugins/obstacle.py`，接到
`_robust_nearest_distance` 里、取分位数之前。保守策略不变：拟合不出
足够置信的平面、或者平面更像墙不像地面，一律不剔除，最差情况下退化成
"没做这层"，不会比这次 7.2102 更差。

坐标系上跟旧版本不完全一样：`plugins/obstacle.py` 故意不把深度图 resize
回原图尺寸（consolidate 时的简化，见 `_preprocess`），地面剔除直接在
固定 518×518 坐标系里算，等效焦距按 `518 * (等效焦距mm / 36mm)` 换算
成这个固定尺寸下的常量（indoor≈417.3px，outdoor 复用 nuScenes CAM_FRONT
标定值等比缩放≈410.1px），不依赖原图宽度。

验证：本地跑 `test_obstacle.py` 对 `test_nameplate.png`，输出跟改动前
完全一致的 `pred_distance=1.7125`（这张图是近景合影，ROI 内没有地面
平面，符合"无地面场景应该是 no-op"的预期）；另外写了合成场景（真实平面
地板 + 独立于地板之外的近处方块障碍物）验证：地板像素正确识别剔除
（≥99%），障碍物像素完全保留（0% 被误删），障碍物本身的最近距离读数
不受影响。也复现了原实现文档里记录过的已知局限——如果 ROI 内有一块比
地板面积更大的纯背景平面（比如正对的墙），RANSAC 可能优先拟合出那个
平面，虽然会被法向量校验正确拒绝（不会误删），但这一轮就没能检测到
真正的地板，属于沿用而非新增的局限。

这次没有改动 outdoor 的 ROI 形状或距离数值换算——那两个方向已经用真实
评测数据反复试过（见上面"outdoor（F1@1m 卡死在 0.0000）"），继续动它们
没有新证据支撑。indoor TP=2、outdoor TP=2 这两个数字本身暂时无法进一步
诊断：榜单反馈里只给了 TN/TP，没有 FP/FN，不知道这批测试集里真正 <1m
的正样本总数有多少——如果正样本本来就稀少，TP=2 可能只是反映了低患病率
而不是召回率差，需要 FN 或者 case 级别明细才能分清楚，不能再靠"看起来
合理"的推理继续调，这一点跟本文档一贯的原则一致。

## 2026-08-08 修复：action=config 每个 case 都无条件重建 adapter，OOM 根因

提交上面这版改动后，第一次真实评测在 case 2/36（container
`phanthymotus-perception-obstacle-1`）就以退出码 137 异常终止。排查
发现是一个跟这次地面剔除改动无关、更早就存在的 bug：`dispatch()` 里
`action=config` 且不带 `instance_id` 的分支（全局 config），每次被
调用都无条件执行 `self._adapter = _build_distance_adapter(...)`——
`provider=local` 时这意味着每次都重新加载两个 onnx 模型、new 一个
onnxruntime session、new 一个 `ThreadPoolExecutor`，而旧的 adapter
（旧 session、旧线程池）既没有 shutdown 也没有显式释放，直接被引用
覆盖丢弃。评测日志显示 `MCP 配置障碍物检测(tool=obstacle)` 这行每个
case 都打一次，说明评测框架确实是每个 case 都调一次 `action=config`——
这正是 `plugins/obstacle_distance/plugin.py`（consolidate 前的旧包）
docstring 里专门写注释警告过的坑："config 只有内容真的变化时才重建…
避免评测框架每个 case 都调一次 start/stop/config 时，重复加载模型、
重复创建销毁 ROS2 资源导致的内存上涨（OCR 插件曾经因为这个被 OOM
杀掉）"——旧包里这层"只有真的变化才重建"的判断，consolidate 成单文件
`plugins/obstacle.py`（commit 5bfa6c9）时丢掉了。

修复：`action=config` 全局分支现在会对比 `provider/url/key/model` 跟
上一次的值，完全没变就跳过重建，直接复用现有 `self._adapter`；真的
需要重建时，对旧 adapter（如果是 `LocalDistanceAdapter`）调用新加的
`close()` 方法 shutdown 掉它的 `ThreadPoolExecutor`，避免线程池累积。

验证：本地起一个 `ObstacleDistancePlugin`（provider=local），连续调用
5 次 `action=config`（内容跟上次完全一样）——修改前每次都应该触发一次
模型重新加载（onnxruntime session 创建日志/CUDA fallback 警告各出现
5 次），修改后 5 次连续调用 `self._adapter` 对象 id 完全不变，只有在
后面真的换了 `model` 参数时才重建了一次，跟预期一致。

这次没有触碰 `instance_id` 场景（带 instance_id 的 config 只是记录
per-instance 配置 + 拆掉对应节点，不涉及重建全局 adapter，本来就没有
这个问题）。

## 2026-08-08 又一次真实评测：config 泄漏修好了（撑到 case 5），但每个
## case 都是 pred_distance=10.0（超时兜底）+ 最终仍然 137

推上一条修复后重新提交，这次撑到了 case 5/36 才异常退出（比之前 case 2
好，说明 config 重建那个泄漏确实是真问题、也确实修对了方向），但案例
级别日志显示前 4 个全部完成的 case，`pred_distance` 无一例外都是
`10.0`——也就是 `LocalDistanceAdapter.estimate()` 每一帧都撞到了内部
2.7s 硬超时，从来没有真正跑完过一次推理、返回过真实距离。这解释了
本文档最早（2026-08-05）就记录过、当时没查出根因的那个谜团："本地
1600x900 合成数据实测中值滤波+RANSAC 全部加起来也就 0.6 秒左右，跟
真实环境 20-40 秒的差距解释不了"——这次的 case 全是 outdoor（jpg，
nuScenes CAM_FRONT），跟历史记录里 outdoor(40s) 比 indoor(20s) 更慢
这个方向也对得上。

**怀疑的根因，跟这几个函数本身的计算量无关**：`_load_models()` 建
onnxruntime session 时没有传 `SessionOptions`，`intra_op_num_threads`
默认是 0（= onnxruntime 自动探测）。自动探测在容器里是已知的坑——按
宿主机能看到的 CPU 核数建线程池（这台开发机 36 核，探测结果拿这台机器
测不出问题），不一定感知 Docker/K8s 的 cgroup CPU 配额。真实评测容器
如果配额远小于宿主机核数（大概率），onnxruntime 会建一个远超实际配额
的线程池，线程互相抢占被 CFS 节流，实测这类问题能把推理拖慢一个数量级
以上，量级上能解释 20-40s；线程本身的栈内存/调度开销同时推高内存占用，
也可能是反复 OOM(137) 的另一个诱因（不止是上一节修的那个 config 泄漏，
两个问题可能都在起作用）。numpy/scipy 底层的 BLAS 线程池（OpenBLAS/MKL）
是同一类问题的另一半，会影响中值滤波和这次加回来的 RANSAC 矩阵运算。

**做的改动**（都是本地测不出收益、只能按最佳实践先做、需要下一次真实
评测反馈验证的改动，跟本文档一贯的风格一致地记录下来）：
1. `_load_models()` 现在显式传 `SessionOptions`，`intra_op_num_threads`
   固定为 1（`OBSTACLE_LOCAL_INTRA_OP_THREADS` 环境变量可调，2026-08-08
   进一步从 2 收紧到 1，先按最保守的设置排除过度订阅这个可能性，确认
   评测容器实际 CPU 配额更宽裕之后再考虑调大），
   `inter_op_num_threads=1`（模型是单条推理图，没有需要并行调度的独立
   子图），不再依赖 onnxruntime 的自动探测。
2. 文件最顶上（`import numpy` 之前）用 `os.environ.setdefault` 把
   `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` 同步设成 1——
   这几个必须在 numpy 第一次被 import 之前设置才有效，只有在这个进程里
   `plugins.obstacle` 是第一个 import numpy 的模块时才生效（比如
   config.yaml 只开了 `obstacle` 这一个插件的场景）；如果 `main.py`
   先加载了别的也用 numpy 的插件，这里设置就晚了，需要在容器启动层面
   （Dockerfile ENV 或启动脚本）设同名变量才能保证生效。

**没有做的事**：没有因为怀疑 RANSAC/中值滤波拖慢了推理就把这次加回来的
地面剔除撤掉——线程过度订阅这个假设能解释"变慢一个数量级"这个量级，
纯 numpy 矩阵运算本身（50 次 RANSAC 迭代，几万个点的向量化点积）在
任何正常线程配置下都不应该到秒级，档次上对不上，不像"过度订阅"这个
解释吻合，所以判断它不是主因，先不动。

**下一步如果这次改动还是没解决**：说明瓶颈可能确实纯粹是这颗真实评测
CPU 太弱（比如 Jetson 实测算力），或者是 ROS2/Docker 本身的开销，
需要能登上那台真实机器（或者至少拿到那次运行的 `docker stats`/CPU
配额信息）才能进一步定位，本地这台开发机器已经明确测不出这个问题
（36 核、没有 cgroup 限制）。

## 2026-08-08 暂时去掉 2.7s 超时兜底，先看真实预测值对不对

上面两轮排查（config 泄漏、线程过度订阅）都是在没有看到过一个真实
`pred_distance` 的情况下做的——目前为止真实评测里 100% 的完成 case
返回的都是 `10.0`，没有任何直接证据能说明模型本身、ROI 逻辑、地面剔除
这些是不是准的，一直是在盲调超时/内存这类"跑不起来"的问题，跟"跑起来
之后准不准"是两个独立的问题，需要先分开验证。

应要求把 `estimate()` 里 `future.result(timeout=self._TIMEOUT_SECONDS)`
的 2.7s 超时去掉了，一开始改成完全不设上限，但很快发现这样跑不通：
`_ObstacleNode.stop()` 等 worker 线程退出只给 3 秒（`join(timeout=3.0)`），
`_inference_worker` 的循环只在每次 `estimate()` 调用返回之后才检查
`_stop_event`，如果一帧推理真的跑很久，3 秒等不到线程真正退出，`stop()`
会直接放弃、把 node 标记删掉，但那个 worker 线程其实还在后台继续跑，
变成"僵尸"——下一个 case 又起一个新 worker，几个 case 下来越堆越多，
现实中很快就又是一次 137（比"没有这次改动"更快出现）。

改成折中方案：`estimate()` 用一个宽松但仍然有限的等待上限
（`LocalDistanceAdapter._DEBUG_MAX_WAIT_SECONDS`，默认 60s，可用
`OBSTACLE_LOCAL_DEBUG_MAX_WAIT` 环境变量调，比历史记录里最差的约 40s
留了余量），`_ObstacleNode.stop()` 的 join 超时也同步跟着调大
（`getattr(adapter, '_DEBUG_MAX_WAIT_SECONDS', 3.0) + 3.0`），保证两边
一致——绝大多数情况下还是能等到真实推理结果，同时 worker 线程最终一定
会退出，不会无限堆积（模型加载失败/图像解码失败/ROI 内没有有效深度这些
真正的错误路径不受影响，那些没有真实值可给，仍然走兜底，这跟"时间"
无关）。

**这是临时状态，不是最终方案**：榜单本身有 3 秒硬约束，放宽到 60s 只是
为了先确认模型/后处理逻辑本身对不对——等看到几个真实预测值、确认量级
合理之后，两处的时间预算都必须收紧回真正的约束（哪怕最后调出来的真实
推理时间稳定在 3 秒以内、不需要这么宽松的保护了，也应该有意识地重新
评估一次，不能放着不管）。

## 2026-08-08 obstacle 插件默认不再尝试 CUDA，改成默认强制 CPU

`_resolve_providers()` 原来的默认值是 `OBSTACLE_LOCAL_DEVICE=auto`，
会优先尝试 `CUDAExecutionProvider`，创建失败才回退 CPU。这台开发机没装
CUDA，"auto" 优先尝试 CUDA 总是直接失败、本地测不出问题；但真实评测
硬件大概率是 Jetson（GPU/CPU 是同一块物理内存，不是独立显存），如果
CUDA 可用，"auto" 会成功建出 CUDA session，这部分占用会直接算在系统
可用内存里——是这几轮反复排查 137 之外还没检查过的一个内存来源。

当时先把默认值改成了 `OBSTACLE_LOCAL_DEVICE=cpu`，但 CUDA 分支的代码
还留着（理论上还能被环境变量重新打开）。CPU 推理本来就在预算内（本地
实测几百毫秒到一点几秒，取决于线程数设置），干脆不用 GPU，也顺带彻底
不用管"GPU 占用"这条评测约束。同一批改动里，OCR 插件（`config.yaml`
里 `ocr.device: cpu`）本来就是显式配置成 CPU，没有这个问题，没有改动。
`config.yaml` 里除了 `ocr`/`obstacle` 之外的插件（asr/tts/htmsg/vop）
本来就都是 `enabled: false`，`main.py` 只按需加载，也没有额外要关的。

**2026-08-08 后续更新**：这次真实评测日志证实了评测框架的运行方式——
会同时拉起 10 个 `phanthymotus-perception-obstacle-N`（N=0~9）容器
并发跑评测，每个都带 `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all`，
GPU 对这些容器是真实可见、可用的（不是本地开发机那种"反正没装CUDA"的
情况），10 个容器还共享同一台宿主机的资源。应要求把 `_resolve_providers()`
里 CUDA 相关代码整个删掉了，不再是"默认 cpu 但留一条 cuda 分支"，而是
硬编码只返回 `["CPUExecutionProvider"]`，`OBSTACLE_LOCAL_DEVICE` 这个
环境变量不再存在。`_load_models()` 里原来"provider 失败就换 CPU 重试"
那段也顺带简化掉了——`providers` 现在永远是 CPU，重试用同样的 provider
列表没有意义，加载失败直接让异常抛出去，比吞掉异常继续跑一个模型没
加载成功的实例更容易定位问题。

## 2026-08-08 真正找到了：ROS2 node/publisher 从来没有被 destroy 过

上面几轮改动（config 泄漏、线程过度订阅、CUDA 显存、去掉超时）推上去
之后，这次评测终于连续拿到了 4 个真实预测值（1.2853 / 9.0671 / 6.0394 /
1.8124），每次都在放宽后的等待上限内正常返回（17~21s），第 5 个 case
才异常退出——不是卡死在某一次推理里，是**成功跑完几个 case 之后才崩**，
典型的"每个 case 攒一点、攒够了才炸"的内存泄漏特征，而不是"推理一直
在等"。

翻代码找到了真正的资源泄漏点，之前一直没查到这里：`_ObstacleNode.stop()`
只 `destroy_subscription()` 了订阅，`self._pub`（`__init__` 里创建的
publisher）和整个 `Node` 本身**从来没有显式 destroy 过**；
`ObstacleDistancePlugin.dispatch()` 的 `action=stop`/`action=config`
分支也只是 `self._executor.remove_node(node)` + `del self._nodes[key]`，
只是把 Python 层的引用丢掉、把 node 从 executor 的 spin 列表里摘掉，
但 node 在 rmw/DDS 层绑定的资源（participant、还活着的 publisher、
发现数据等）不会因为 Python 引用计数归零就自动释放——rclpy 明确要求
显式调用 `destroy_node()` 才会真正释放这些。

这个评测框架的调用方式是**每个 case 都 start 一个新 node、case 结束
都 stop 掉**（`start()` 里 `if node_key not in self._nodes` 这个复用
判断在这套调用模式下从来没有真正生效过，因为 stop 每次都会把 node 从
`self._nodes` 里删掉，下一次 start 必然重新建一个）——也就是说改之前
每个 case 都在泄漏一份 node+publisher 级别的 DDS 资源，只是单份泄漏量
比之前修的模型/线程泄漏小得多，攒够 4~5 个 case 的量级才会被 OOM 杀掉，
表现上很容易被误判成"推理慢/卡住"（这次能明确排除是因为这次终于看到了
真实预测值、且每次都在预算内完成）。这也是最早 `plugin.py`（consolidate
前的旧包）注释里就点名过的"重复创建销毁 ROS2 资源导致内存上涨"那类
问题——之前几轮只修了"重复加载模型"这一半（config 泄漏那次），没有
补上"重复创建销毁 node"这一半，这次补上。

**改动**：`dispatch()` 里 `action=stop`（两个分支）和 `action=config`
（instance_id 分支）在 `self._executor.remove_node(node)` 之后，都加了
一次 `node.destroy_node()`，让 node 真正被丢弃前完整释放它的 publisher
和 DDS 层资源。

**这次没能在本地验证**：`_ObstacleNode` 继承的是真正的 `rclpy.node.Node`，
这台开发机没装 rclpy（`test_obstacle.py` 为了绕开这个专门 stub 了一版
`Node = object`），`destroy_node()` 这类真正的 ROS2 生命周期行为没法在
本地跑起来验证，只做了语法检查，效果需要下一次真实评测确认。

## 已知局限（诚实说明近似之处）

- outdoor 场景直接用 ROI 内深度分位数近似"最近障碍物距离"，没有做
  "车头保险杠水平面距离"换算（榜单对无人车场景的真值定义是 3D 标注框
  到车头保险杠的水平距离，且排除静态杆柱/地面等类别；单目深度图本身
  给不出目标类别信息，换算依赖的相机-车身标定/焦距估计如果不准，误差
  会按最近像素离图像中心的距离成比例放大——历史遗留的 `obstacle_distance`
  包里实测过一版换算，RMSE 从 4.42 涨到 9.45，是负优化，故这次不引入）。
- indoor/outdoor 用完全相同的 ROI 比例，outdoor 没有独立验证过这个比例
  是否最优，只是复用了榜单文档唯一明确给出过的室内比例。
- 地面平面剔除（RANSAC）2026-08-08 已重新加回（见上面对应章节），仍然是
  保守策略：ROI 内如果有一块比地板更大的纯背景平面（比如正对的墙），
  RANSAC 可能优先拟合出那个平面，被法向量校验拒绝后这一轮就不再剔除
  地板——不会误删障碍物，但也不保证每次都能测到地板，是已知且接受的
  局限，不是这次要解决的目标。
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

`provider=local` 分支的构造调用方式固定不变，运行时参数用环境变量调，
不走 configSchema：`OBSTACLE_LOCAL_MODEL_DIR`、`OBSTACLE_LOCAL_PERCENTILE`、
`OBSTACLE_LOCAL_INTRA_OP_THREADS`（默认1）、`OBSTACLE_LOCAL_DEBUG_MAX_WAIT`
（默认60秒，debug 阶段用，见上面章节）。推理设备固定 CPU，没有对应的
环境变量（2026-08-08 起 `_resolve_providers()` 不再支持切 GPU）。

## 关于 `plugins/obstacle_distance/` 目录

这次改动前，仓库里已经有一个独立的 `plugins/obstacle_distance/` 包
（`main.py` 之前 `from plugins.obstacle_distance import ObstacleDistancePlugin`
加载它），实现思路更完整（额外做了 RANSAC 地面剔除、outdoor 换算实验等），
`README.md` 记录了大量真实评测迭代过程。这次按要求把逻辑整合进单文件
`plugins/obstacle.py`（并把 `main.py`/`config.yaml` 的加载入口切过来），
`obstacle_distance/` 目录本身没有删除，暂时保留供参考，但已不再被
`main.py` 加载。是否需要清理，需要确认后再处理。
