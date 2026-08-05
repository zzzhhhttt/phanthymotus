#!/usr/bin/env python3
"""
plugins/obstacle_distance/roi.py — ROI 提取与最近障碍物距离计算。

严格对应榜单文档《6. 距离计算逻辑说明 · (一) 室内机器人场景》里给出的 ROI：

    图像尺寸 640x480 时：
        列范围（水平）：213 ~ 426   == 宽度的 1/3 ~ 2/3（正前方中心视野）
        行范围（垂直）：0   ~ 300   == 高度的 0   ~ 5/8（排除下方地面）

这里用比例（1/3~2/3、0~5/8）而不是写死像素值，这样非 640x480 的输入
（比如榜单声明的等效焦距对应的原始拍摄分辨率）也能按同样的相对区域取 ROI。

无人车场景的"真值"是 3D OBB 最近面到车头保险杠的水平距离，需要完整的
3D 检测 + 类别过滤 + 自车标定，单张图片 + 单目深度模型没有这个信息。
这里对无人车场景用同一套"ROI + 百分位数"逻辑做近似（前向中心区域、
排除天空的一段裁切），本质是把"ROI 内最近深度"当成"最近障碍物表面
距离"的代理指标。见 README.md 的 Limitations 一节。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RoiBox:
    row_start: int
    row_end: int   # exclusive
    col_start: int
    col_end: int   # exclusive


def indoor_roi(width: int, height: int) -> RoiBox:
    """室内机器人 ROI：col 1/3~2/3，row 0~5/8（榜单文档给定的比例）。"""
    return RoiBox(
        row_start=0,
        row_end=max(1, round(height * 5 / 8)),
        col_start=round(width / 3),
        col_end=round(width * 2 / 3),
    )


def outdoor_roi(width: int, height: int) -> RoiBox:
    """无人车 ROI：跟 indoor_roi 完全一样的比例（col 1/3~2/3、row 0~5/8）。

    榜单文档只给了室内机器人场景精确的像素级 ROI 定义（6. 距离计算逻辑
    说明·(一)），无人车场景没有给出对应的图像空间 ROI（它的真值是基于
    3D OBB 标注算出来的，不是基于图像裁剪+百分位数）。既然榜单没有给
    outdoor 专门定义 ROI，就不自己发明一个更宽/更窄的裁剪框，直接复用
    榜单唯一明确给出过的那个比例——这样至少行为跟文档"对得上"，不引入
    额外的、没有依据的假设。

    2026-08-03 曾经改过一版更宽的裁剪（col 1/12~11/12、row 1/8~1）想
    解决 F1@1m 连续两次精确等于 0 的问题，理由是"窄框可能把真正最近的
    障碍物裁在框外"，但这个改动：(1) 没有 case 级别 GT 数据验证过，
    是猜测；(2) 引入了新风险——row 加宽到接近画面底部，很容易把 GT 里
    明确排除的路面（`flat.*` 类别）当成"最近障碍物"框进来，可能比原来
    更差。改回跟 indoor 一致的比例，不再自行加宽。
    """
    return indoor_roi(width, height)


def _denoise_depth_map(depth_map: np.ndarray, size: int = 5) -> np.ndarray:
    """中值滤波去掉深度图里的孤立噪声点，再算百分位数。

    单目深度模型在物体边缘/深度不连续处经常出现"孤立异常值"（个别像素
    被预测得比周围明显更近或更远）。P1 这种低百分位数虽然比 min() 稳，
    但 ROI 面积小的时候 1% 可能只对应一两个像素，一个噪声点就能直接
    决定最终距离——中值滤波是标准做法，用每个像素邻域的中位数替换它，
    直接消掉这种孤立噪声，同时基本不影响真实的物体表面（表面本身是
    连续的一片，中值滤波后数值几乎不变）。

    在整张深度图上滤波（不是只滤 ROI 内的小块），避免裁出小块之后边缘
    因为缺少邻域上下文导致滤波效果变差。size=5 是标准选择，太小起不到
    去噪效果，太大会把真正的物体边缘也磨掉。
    """
    from scipy.ndimage import median_filter
    return median_filter(depth_map, size=size)


def _backproject(rows: np.ndarray, cols: np.ndarray, depths: np.ndarray,
                  fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """像素 (row, col) + 深度 -> 相机坐标系下的 3D 点。

    针孔相机模型，约定 X 右正、Y 下正（行方向）、Z 前正（深度方向），
    跟图像坐标系自然对应（不需要额外的坐标轴翻转）。
    """
    x = (cols - cx) * depths / fx
    y = (rows - cy) * depths / fy
    return np.stack([x, y, depths], axis=-1)


def _ransac_plane(sample_points: np.ndarray, score_points: Optional[np.ndarray] = None,
                   iterations: int = 150, inlier_thresh_m: float = 0.05, seed: int = 0):
    """RANSAC 拟合一个平面，返回 (normal, inlier_mask, inlier_fraction)，
    点数太少或者拟不出稳定平面时返回 None。

    平面方程：normal · p + d = 0（normal 已归一化）。

    sample_points 是用来"抽 3 个点试拟合"的候选池，score_points 是用来
    "数这个候选平面到底有多少内点"的评分池——两者可以不同（见
    _ground_removal_mask：只从画面下方抽样候选平面，避免大片背景墙这种
    更大的平面把随机采样"抢走"，但抽出候选平面之后，还是在整个 ROI 范围
    评分/收集内点，这样即使地面延伸到画面上方，也能被同一个平面覆盖到）。
    不传 score_points 时默认等于 sample_points。
    """
    n = sample_points.shape[0]
    if n < 20:
        return None
    if score_points is None:
        score_points = sample_points
    if score_points.shape[0] < 20:
        return None

    rng = np.random.default_rng(seed)
    best_count = 0
    best_inliers = None
    best_normal = None

    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = sample_points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:  # 三点共线，选不出平面，跳过这次采样
            continue
        normal = normal / norm
        d = -np.dot(normal, p0)
        dist = np.abs(score_points @ normal + d)
        inliers = dist < inlier_thresh_m
        count = int(np.sum(inliers))
        if count > best_count:
            best_count, best_inliers, best_normal = count, inliers, normal

    if best_inliers is None:
        return None
    return best_normal, best_inliers, best_count / score_points.shape[0]


def _ground_removal_mask(
    patch: np.ndarray,
    row_offset: int,
    col_offset: int,
    focal_length_px: float,
    min_inlier_fraction: float = 0.25,
    normal_vertical_ratio: float = 1.5,
) -> np.ndarray:
    """在 ROI patch 内用 RANSAC 拟合地面平面，返回"判定为地面、应剔除"的
    布尔 mask（跟 patch 同形状）。

    地面 vs 墙面怎么区分：地面法向量接近"竖直"方向（相机坐标系里的 Y
    轴，也就是图像的行方向），墙面法向量接近"水平"方向（正对/侧对相机，
    X/Z 分量更大）——单纯"RANSAC 拟合出一个平面就当地面剔除"是不对的，
    ROI 里如果背景是一整面墙（榜单示例图那种场景），会被误判成"地面"
    整个铲掉，把真实障碍物也删了。这里要求拟合出的平面法向量的 Y 分量
    明显大于 X、Z 分量才判定为地面。

    保守策略：拟合不出足够置信的平面（inlier 比例太低），或者平面看起来
    更像墙不像地面，一律不剔除任何像素（返回全 False）——宁可漏判一点
    地面，也不能误删真实障碍物，这个错误的代价更高。
    """
    valid = np.isfinite(patch) & (patch > 0)
    ground = np.zeros_like(patch, dtype=bool)
    if not np.any(valid):
        return ground

    rows_local, cols_local = np.nonzero(valid)
    depths = patch[rows_local, cols_local].astype(np.float64)
    rows_global = (rows_local + row_offset).astype(np.float64)
    cols_global = (cols_local + col_offset).astype(np.float64)

    h, w = patch.shape[:2]
    cx = col_offset + w / 2.0  # 主点用整张图的中心估计，这里近似用 patch 中心 + 偏移
    cy = row_offset + h / 2.0
    points = _backproject(rows_global, cols_global, depths,
                           focal_length_px, focal_length_px, cx, cy)

    # 只从 patch 下方 40% 的行里抽 3 点候选平面（地面物理上更可能出现在
    # 画面下方），但评分/收集内点还是用全部有效点。原因：如果 ROI 内
    # 同时有一整面大背景墙（榜单示例图那种场景）和一小片地面，两者都是
    # 完美平面，纯随机采样很容易被面积更大的墙"抢走"——RANSAC 会拟合出
    # 墙这个平面，因为它的法向量不够竖直会被下面的检查正确拒绝，但也
    # 就此错过了真正的地面，白白抽样一轮，等于没做地面剔除。限定采样
    # 范围到画面下方，让候选平面更有机会命中地面。
    bottom_threshold = row_offset + h * 0.6
    bottom_mask = rows_global >= bottom_threshold
    sample_points = points[bottom_mask] if np.count_nonzero(bottom_mask) >= 20 else points

    result = _ransac_plane(sample_points, score_points=points)
    if result is None:
        return ground
    normal, inliers, inlier_fraction = result
    if inlier_fraction < min_inlier_fraction:
        return ground
    if abs(normal[1]) < normal_vertical_ratio * max(abs(normal[0]), abs(normal[2]), 1e-6):
        # 法向量不够"竖直"，更像墙面而不是地面，不剔除
        return ground

    ground[rows_local[inliers], cols_local[inliers]] = True
    return ground


def nearest_obstacle_distance(
    depth_map: np.ndarray,
    roi: RoiBox,
    percentile: float = 1.0,
    focal_length_px: Optional[float] = None,
) -> float:
    """ROI 内深度值的 P{percentile} 分位数，作为"最近障碍物表面距离"。

    用低百分位数（默认 P1）代替严格的 min()，是为了对单像素噪声/深度图
    个别异常低值更鲁棒——跟榜单示例图里"P1 百分位数对应的最近像素点"的
    描述一致。滤波（见 _denoise_depth_map）在百分位数之前做，两层防护
    叠加：中值滤波先去掉孤立噪声点，百分位数再对滤波后仍存在的分布
    尾部更鲁棒。

    Args:
        depth_map: HxW，单位米，值越大表示越远。
        roi: 感兴趣区域（像素索引，含 row_start/col_start，不含 row_end/col_end）。
        percentile: 0~100，取 ROI 内深度分布的第几百分位作为最近距离。
        focal_length_px: 传了就额外做一次 RANSAC 地面平面剔除（见
            _ground_removal_mask），不传（默认 None）就跳过这一步，保持
            旧行为——ROI 本身（indoor/outdoor 都用 col 1/3~2/3、row 0~5/8）
            已经排除了大部分地面，这一步是给"ROI 内仍残留少量地面"这种
            情况加的第二层保护，不是替代 ROI。

    Returns:
        最近障碍物距离（米）。ROI 内没有有效深度时返回 float('nan')。
    """
    depth_map = _denoise_depth_map(depth_map)
    h, w = depth_map.shape[:2]
    r0, r1 = max(0, roi.row_start), min(h, roi.row_end)
    c0, c1 = max(0, roi.col_start), min(w, roi.col_end)
    if r1 <= r0 or c1 <= c0:
        return float("nan")

    patch = depth_map[r0:r1, c0:c1]
    valid_mask = np.isfinite(patch) & (patch > 0)
    if focal_length_px is not None and focal_length_px > 0:
        ground = _ground_removal_mask(patch, r0, c0, focal_length_px)
        valid_mask = valid_mask & ~ground

    valid = patch[valid_mask]
    if valid.size == 0:
        return float("nan")

    return float(np.percentile(valid, percentile))


def nearest_obstacle_pixel(
    depth_map: np.ndarray,
    roi: RoiBox,
    percentile: float = 1.0,
    focal_length_px: Optional[float] = None,
) -> tuple[float, int, int] | None:
    """跟 nearest_obstacle_distance 一样算最近距离，但同时返回那个像素的
    (row, col)——outdoor 场景要用这个像素在图像里的列位置，配合针孔相机
    模型换算横向偏移（见 predict.py 里的说明：outdoor 的 GT 是水平面
    2D 距离，不是纯深度，偏心的障碍物需要横向偏移信息才能算对）。

    实现上：percentile 本身不直接对应某个具体像素（是插值出来的统计量），
    这里退而求其次，找 ROI 内深度值离这个百分位数最近的那个像素，用它
    的位置近似"最近点在图像里的位置"。

    跟 nearest_obstacle_distance 一样，先做中值滤波去掉孤立噪声点（见
    _denoise_depth_map），避免噪声点被误当成"最近像素"、连带把它的
    列位置也带偏（这个函数返回的列位置还要参与 outdoor 的横向偏移换算，
    位置本身不准的话，横向距离会跟着错）。focal_length_px 传了的话，
    额外做一次地面平面剔除，说明见 nearest_obstacle_distance。

    Returns:
        (distance_m, row, col)，ROI 内没有有效深度时返回 None。
    """
    depth_map = _denoise_depth_map(depth_map)
    h, w = depth_map.shape[:2]
    r0, r1 = max(0, roi.row_start), min(h, roi.row_end)
    c0, c1 = max(0, roi.col_start), min(w, roi.col_end)
    if r1 <= r0 or c1 <= c0:
        return None

    patch = depth_map[r0:r1, c0:c1]
    mask = np.isfinite(patch) & (patch > 0)
    if not np.any(mask):
        return None

    if focal_length_px is not None and focal_length_px > 0:
        ground = _ground_removal_mask(patch, r0, c0, focal_length_px)
        mask_after_ground = mask & ~ground
        # 极端情况下地面剔除把 ROI 内全部有效像素都判成了地面（比如
        # RANSAC 误判），这时候宁可退回"不剔除地面"的结果，也不要直接
        # 判失败——一个可能包含地面污染的距离，比完全没有结果要有用。
        if np.any(mask_after_ground):
            mask = mask_after_ground

    target = np.percentile(patch[mask], percentile)
    diff = np.where(mask, np.abs(patch - target), np.inf)
    local_row, local_col = np.unravel_index(np.argmin(diff), diff.shape)

    return float(target), int(r0 + local_row), int(c0 + local_col)


def scale_roi_to_size(canonical: RoiBox, canonical_size: tuple[int, int], target_size: tuple[int, int]) -> RoiBox:
    """把在 canonical_size=(W,H) 下定义的 ROI 等比缩放到 target_size=(W,H)。

    depth_model 的输出分辨率跟原图分辨率可能不一致（模型有自己的推理
    尺寸），推理完再把深度图 resize 回原图尺寸即可复用同一个 ROI —— 这个
    函数是备用工具，正常路径里用不到（因为我们总是先把 depth map resize
    回原图尺寸再算 ROI），保留是为了不强制要求调用方这么做。
    """
    cw, ch = canonical_size
    tw, th = target_size
    sx, sy = tw / cw, th / ch
    return RoiBox(
        row_start=round(canonical.row_start * sy),
        row_end=round(canonical.row_end * sy),
        col_start=round(canonical.col_start * sx),
        col_end=round(canonical.col_end * sx),
    )
