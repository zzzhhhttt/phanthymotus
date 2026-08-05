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

# ── Outdoor 距离矫正说明 ─────────────────────────────────────────────────
#
# 榜单文档对 outdoor 的定义是："从自车车头前保险杠到最近障碍物表面的
# 水平面（x-y 平面）内的欧氏距离"——这跟单目深度模型直接给出的"沿相机
# 光轴方向的深度（Z）"是两个不同的量：
#
#   1. 参考点不同：GT 量的是"保险杠"（ego 坐标系 x=3.412m），我们的
#      深度模型天然量的是"相机光心"。这批数据是 nuScenes（文件名带
#      CAM_FRONT），nuScenes 自车坐标系原点在后轴中心，CAM_FRONT 标定
#      平移量常见值约 x≈1.70m，比保险杠靠后约 1.7m。
#   2. 距离类型不同：GT 是水平面 2D 欧氏距离（沿相机光轴的纵深 Z，加上
#      垂直于光轴的横向偏移 X，两者做欧氏距离），不是单纯的纵深 Z。
#      这一点从榜单给的行人示例图能直接看出来：行人跟车头"纵深方向
#      基本对齐"（lx 在 OBB 内部），真正拉开 2.498m 这个距离的是
#      "车头偏右"这个横向分量（ly 方向大幅超出）——如果只用纵深 Z，
#      这类case会被算成远得多的错误值,且没有任何常数能矫正回来
#      （因为横向偏移跟纵深根本是两个独立变量，不是固定比例关系）。
#
# 2026-08-03/04 实测：换了两次 ROI（窄/宽）、加了纯纵深的常数矫正，
# F1@1m 都精确等于 0.0000，没有任何变化——这跟"只是参考点没对齐"的
# 假设不符（如果只是常数偏移，矫正后应该能看到 F1 从 0 变成非 0），
# 更符合"有一部分case的最近障碍物是从侧面接近的，纵深值本身就是在测
# 错误的物理量"这个假设。
#
# 用针孔相机模型把纵深换算成水平面 2D 距离：ROI 内最近点所在的像素列
# 位置 + 深度值 + 相机内参，能反推出这个点相对相机光轴的横向偏移 X，
# 再用 sqrt(X² + Z²) 近似水平面欧氏距离，比单纯用 Z 更贴近榜单定义。
#
# 相机内参用的是 nuScenes CAM_FRONT 公开数据里的常见标定值（原始
# 1600x900 图像下 fx≈1266.4，主点 cx 近似取图像中心），不是这批评测
# 数据集实测标定出来的精确值——跟 1.7m 那个偏移量一样，是能拿到的最好
# 估计，不是确认过的精确值。运行时按实际图片宽度等比缩放 fx。
_OUTDOOR_CAMERA_TO_BUMPER_OFFSET_M = 1.7
_NUSCENES_CAM_FRONT_FX_AT_1600W = 1266.4  # nuScenes CAM_FRONT 典型焦距（像素，1600px 宽下）
_NUSCENES_CAM_FRONT_REF_WIDTH = 1600

# ── 地面平面剔除用的焦距估计 ────────────────────────────────────────────
#
# roi.py 的地面平面剔除（RANSAC 拟合 + 法向量校验）需要相机内参（焦距）
# 才能把像素+深度反投影成 3D 点。outdoor 复用上面 nuScenes 的真实标定值
# （更准），indoor 没有对应的公开标定数据，用榜单文档给的"等效焦距 29mm"
# 按标准 35mm 等效换算公式估算：
#     fx_pixels ≈ image_width_px × 等效焦距mm / 36mm
# （36mm 是全画幅传感器的参考宽度，这是"35mm 等效焦距"这个概念本身的
# 定义基准，不是额外假设）。这是能拿到的最好估计，不是这批评测数据集
# 实测标定出来的精确值。
_INDOOR_FOCAL_LENGTH_EQUIV_MM = 29.0
_FULL_FRAME_SENSOR_WIDTH_MM = 36.0


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

    depth_z, _row, col = pixel

    if resolved_domain == "indoor":
        # indoor 榜单定义就是纯纵深（"沿相机光轴方向的深度"），不做
        # 横向换算，直接用深度值
        dist = depth_z
    else:
        # outdoor：纵深 -> 保险杠参考点 + 水平面 2D 距离，见模块顶部说明
        z_from_bumper = max(0.0, depth_z - _OUTDOOR_CAMERA_TO_BUMPER_OFFSET_M)
        cx = w / 2.0
        lateral_x = (col - cx) * depth_z / fx if fx > 0 else 0.0
        dist = (lateral_x ** 2 + z_from_bumper ** 2) ** 0.5

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
