#!/usr/bin/env python3
"""
plugins/obstacle_distance/predict.py — 单图推理入口，对应榜单输入输出契约。

榜单契约（见项目 docs / 需求描述）：
    输入：单张 png 或 jpg 图像
    输出：{"pred_distance": <float, 米>}  —— 正前方最近障碍物距离

用法：
    python3 predict.py /path/to/image.png
    # -> {"pred_distance": 1.7172}

也可以当库用：
    from plugins.obstacle_distance.predict import predict_distance
    result = predict_distance(image_bytes, filename="frame_0001.jpg")
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from .depth_model import DepthEstimator, get_default_estimator, load_image
from .domain import Domain, detect_domain
from .roi import indoor_roi, nearest_obstacle_distance, outdoor_roi

_ROUND_NDIGITS = 4

# 无人车场景的 GT 参考点是车头保险杠（自车坐标系 x=3.412m，见榜单文档
# "距离计算逻辑说明·(二)"），不是相机光心。我们测的是相机到障碍物表面的
# 直线距离，两个参考点不是同一个点——这批数据是 nuScenes（文件名带
# CAM_FRONT），nuScenes 自车坐标系原点在后轴中心，CAM_FRONT 标定平移量
# 大约 x≈1.70m（nuScenes 官方标定的常见值），比保险杠(x=3.412m)靠后约
# 1.7m。保险杠比相机更靠前、离前方障碍物更近，所以相机测出来的距离会
# 系统性地比"保险杠到障碍物"的真值大出这个偏移量——对 F1@1m 这种阈值类
# 指标是致命的：真实距保险杠 <1m 的 case，相机测出来可能是 <1m+1.7m
# ≈2.7m+，永远判不成 Positive，不管 ROI 多准都没用。
#
# 2026-08-03 实测：两次提交（窄 ROI、宽 ROI）F1@1m 都精确等于 0.0000，
# 换 ROI 没用，符合"参考点系统性偏移"这个假设（而不是"漏检"）。这里
# 减去这个偏移量做近似矫正。这个 1.7m 是根据 nuScenes 公开标定数据估的
# 一个近似值，不是这批评测数据集实测标定出来的精确值，如果矫正后 F1
# 还是不对（比如变成 0 附近但不是 0，或者矫枉过正变太小），需要根据
# 实际提交结果调整这个常数——先跑一次看效果，比空想更准。
_OUTDOOR_CAMERA_TO_BUMPER_OFFSET_M = 1.7


def predict_distance(
    image_bytes: bytes,
    filename: Optional[str] = None,
    domain: Optional[Domain] = None,
    percentile: float = 1.0,
    estimator: Optional[DepthEstimator] = None,
) -> dict:
    """核心推理函数：图片 bytes -> {"pred_distance": float}。

    Args:
        image_bytes: 原始图片文件内容（png 或 jpg）。
        filename: 可选，仅在 image_bytes 的 magic bytes 无法判断格式时
            用来兜底判定 domain（见 domain.detect_domain）。
        domain: 显式指定 "indoor"/"outdoor"，跳过自动判定（比如调用方
            已经知道是哪个 topic/相机）。
        percentile: ROI 内取第几百分位深度作为最近距离，默认 P1（对应
            榜单示例图"P1 百分位数对应的最近像素点"）。
        estimator: 复用已加载好模型的 DepthEstimator 实例；不传则用
            进程内默认单例（避免重复加载模型）。

    Returns:
        {"pred_distance": float} —— 出错或 ROI 内没有有效深度时返回
        {"pred_distance": None, "error": "..."}，调用方（比如 eval.py）
        据此统计"推理失败率"。
    """
    try:
        image = load_image(image_bytes)
    except Exception as e:  # noqa: BLE001 — 榜单会喂各种畸形图片，兜底成失败样本而不是崩进程
        return {"pred_distance": None, "error": f"decode_failed: {e}"}

    resolved_domain = domain or detect_domain(image_bytes, filename)
    est = estimator or get_default_estimator()

    try:
        depth = est.estimate(image, resolved_domain)
    except Exception as e:  # noqa: BLE001
        return {"pred_distance": None, "error": f"inference_failed: {e}"}

    w, h = image.size
    roi = indoor_roi(w, h) if resolved_domain == "indoor" else outdoor_roi(w, h)
    dist = nearest_obstacle_distance(depth, roi, percentile=percentile)

    if dist != dist:  # NaN check without importing math
        return {"pred_distance": None, "error": "empty_roi_or_no_valid_depth"}

    if resolved_domain == "outdoor":
        # 相机光心距离 -> 近似换算成保险杠参考点距离，见模块顶部
        # _OUTDOOR_CAMERA_TO_BUMPER_OFFSET_M 的详细说明
        dist = max(0.0, dist - _OUTDOOR_CAMERA_TO_BUMPER_OFFSET_M)

    return {"pred_distance": round(dist, _ROUND_NDIGITS)}


def _main() -> int:
    parser = argparse.ArgumentParser(description="正前方最近障碍物距离推理")
    parser.add_argument("image", type=str, help="输入图片路径（png/jpg）")
    parser.add_argument("--domain", choices=["indoor", "outdoor"], default=None,
                         help="强制指定场景域，默认根据图片格式自动判定")
    parser.add_argument("--percentile", type=float, default=1.0,
                         help="ROI 内取第几百分位深度作为最近距离，默认 1")
    parser.add_argument("--timing", action="store_true", help="额外打印推理耗时")
    args = parser.parse_args()

    path = Path(args.image)
    image_bytes = path.read_bytes()

    t0 = time.monotonic()
    result = predict_distance(image_bytes, filename=path.name, domain=args.domain,
                               percentile=args.percentile)
    elapsed = time.monotonic() - t0

    print(json.dumps(result, ensure_ascii=False))
    if args.timing:
        print(f"# elapsed={elapsed*1000:.1f}ms", file=sys.stderr)

    return 0 if result.get("pred_distance") is not None else 1


if __name__ == "__main__":
    raise SystemExit(_main())
