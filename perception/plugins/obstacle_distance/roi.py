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
    """无人车 ROI 近似：col 1/3~2/3（前向中心车道），row 1/4~7/8
    （裁掉上方 1/4 天空/远景，下方留到 7/8 因为车头本身较低、近距离
    障碍物在画面里位置更靠下，跟室内机器人 0~5/8 的假设不一样）。"""
    return RoiBox(
        row_start=round(height / 4),
        row_end=max(1, round(height * 7 / 8)),
        col_start=round(width / 3),
        col_end=round(width * 2 / 3),
    )


def nearest_obstacle_distance(
    depth_map: np.ndarray,
    roi: RoiBox,
    percentile: float = 1.0,
) -> float:
    """ROI 内深度值的 P{percentile} 分位数，作为"最近障碍物表面距离"。

    用低百分位数（默认 P1）代替严格的 min()，是为了对单像素噪声/深度图
    个别异常低值更鲁棒——跟榜单示例图里"P1 百分位数对应的最近像素点"的
    描述一致。

    Args:
        depth_map: HxW，单位米，值越大表示越远。
        roi: 感兴趣区域（像素索引，含 row_start/col_start，不含 row_end/col_end）。
        percentile: 0~100，取 ROI 内深度分布的第几百分位作为最近距离。

    Returns:
        最近障碍物距离（米）。ROI 内没有有效深度时返回 float('nan')。
    """
    h, w = depth_map.shape[:2]
    r0, r1 = max(0, roi.row_start), min(h, roi.row_end)
    c0, c1 = max(0, roi.col_start), min(w, roi.col_end)
    if r1 <= r0 or c1 <= c0:
        return float("nan")

    patch = depth_map[r0:r1, c0:c1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return float("nan")

    return float(np.percentile(valid, percentile))


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
