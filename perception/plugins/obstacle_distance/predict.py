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
from .roi import indoor_roi, nearest_obstacle_pixel, outdoor_roi

_ROUND_NDIGITS = 4

# ── Outdoor 距离计算：曾经做过换算，2026-08-05 撤回了 ────────────────────
#
# 榜单文档对 outdoor 的定义是"车头保险杠到障碍物的水平面2D距离"，跟单目
# 深度模型直接给出的"相机光轴方向的纵深 Z"不是同一个量，理论上需要换算
# （保险杠参考点偏移 + 针孔相机横向投影）。之前确实按这个思路加过换算
# （保险杠偏移 1.7m + 用 nuScenes 公开标定值 fx≈1266.4 反算横向偏移，
# sqrt(X²+Z²) 合成水平距离），但 2026-08-05 真实评测结果证明这是负
# 优化：outdoor RMSE 从最简单版本的 4.4231 涨到加了这两层换算之后的
# 9.4513（几乎翻倍），F1@1m 依旧是 0.0000，没有任何正面效果。怀疑是
# 横向距离换算依赖的焦距估计值（没有真实标定数据验证过，见 README
# Limitations）不准，误差按"最近像素离图像中心的距离"成比例放大到了
# 每个 case 上，越换算越错。
#
# 现在改回跟 indoor 完全一样的纯深度值——这是目前 outdoor 实测 RMSE
# 最低的一版，也是唯一有真实数据支撑"没有让结果变差"的方法。今后如果
# 想再引入类似换算，应该先想办法拿到哪怕几个真实 GT 样本验证一下换算
# 方向和参数对不对，而不是叠加多个没有 GT 数据验证过的假设。

# ── 地面平面剔除用的焦距估计 ────────────────────────────────────────────
#
# roi.py 的地面平面剔除（RANSAC 拟合 + 法向量校验）需要相机内参（焦距）
# 才能把像素+深度反投影成 3D 点——注意这个焦距只用来算"哪些像素是地面"，
# 不再用来做上面撤回的那套距离换算，两者是独立的用途。
# indoor 用榜单文档给的"等效焦距 29mm"按标准 35mm 等效换算公式估算：
#     fx_pixels ≈ image_width_px × 等效焦距mm / 36mm
# （36mm 是全画幅传感器的参考宽度，这是"35mm 等效焦距"这个概念本身的
# 定义基准，不是额外假设）。outdoor 用 nuScenes CAM_FRONT 公开标定的
# 常见值（1600px 宽下 fx≈1266.4，这个值本身没有真实核实过，见 README
# Limitations，但地面剔除只需要焦距量级大致对，不像距离换算那样对精确
# 数值敏感——法向量校验本身就是个粗粒度的方向判断，焦距差个 10~20%
# 不会明显改变"这个平面朝上还是朝前"这个结论）。
_INDOOR_FOCAL_LENGTH_EQUIV_MM = 29.0
_FULL_FRAME_SENSOR_WIDTH_MM = 36.0
_NUSCENES_CAM_FRONT_FX_AT_1600W = 1266.4  # nuScenes CAM_FRONT 典型焦距（像素，1600px 宽下）
_NUSCENES_CAM_FRONT_REF_WIDTH = 1600


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

    if resolved_domain == "indoor":
        roi = indoor_roi(w, h)
        fx = w * _INDOOR_FOCAL_LENGTH_EQUIV_MM / _FULL_FRAME_SENSOR_WIDTH_MM
    else:
        roi = outdoor_roi(w, h)
        fx = _NUSCENES_CAM_FRONT_FX_AT_1600W * (w / _NUSCENES_CAM_FRONT_REF_WIDTH)

    # focal_length_px 传进去之后，nearest_obstacle_pixel 会额外做一次
    # RANSAC 地面平面剔除（indoor/outdoor 都做——outdoor 场景 GT 同样
    # 明确排除路面 flat.*，跟 indoor 排除地面是同一个意图），详见 roi.py
    # 里 _ground_removal_mask 的说明
    pixel = nearest_obstacle_pixel(depth, roi, percentile=percentile, focal_length_px=fx)

    if pixel is None:
        return {"pred_distance": None, "error": "empty_roi_or_no_valid_depth"}

    depth_z, _row, _col = pixel

    # outdoor 曾经在这里做过"保险杠偏移矫正 + 针孔相机横向2D距离换算"
    # （見 git 历史），2026-08-05 real 评测结果证明这是负优化：加了这两层
    # 换算之后 RMSE 从 4.4231 涨到 9.4513（近乎翻倍），F1@1m 依旧是
    # 0.0000，没有任何正面效果。怀疑是横向距离换算依赖的焦距估计值
    # （没有真实标定数据验证过）不准，误差按"最近像素离图像中心的距离"
    # 成比例放大到了每个 case 上。改回跟 indoor 完全一样的纯深度值——
    # 这是目前 outdoor 实测 RMSE 最低的一版，也是唯一有真实数据支撑
    # "没有让结果变差"的方法。不再叠加没有 GT 数据验证过的额外假设，
    # 见 README.md Limitations 一节。
    dist = depth_z

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
