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
