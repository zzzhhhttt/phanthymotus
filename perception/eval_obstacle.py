#!/usr/bin/env python3
"""
eval_obstacle.py — 本地复现 obstacle 赛道评测指标，方便提交前自测。

指向的是当前生效的 `plugins/obstacle.py`（LocalDistanceAdapter），不是已经
不再被 main.py 加载的 `plugins/obstacle_distance/` 旧包——旧包里的
`eval.py` 阈值还是 F1@1m，是 2026-08-09 改成 2m 之前的口径，仅作历史参考，
不要拿旧包的数字跟这里比较。

复现的指标：
    - Precision@2m / Recall@2m / F1@2m（主排序指标，阈值 2 米，
      2026-08-09 确认从 1m 改为 2m）
    - RMSE（米）
    - FPS（每秒处理图片数）
    - 推理失败率

用法：
    python3 eval_obstacle.py manifest.csv
    python3 eval_obstacle.py manifest.jsonl

manifest.csv 格式（表头必须包含 image,gt_distance，domain 可选，不填就按
图片格式自动判定 png=indoor/jpg=outdoor，见 plugins/obstacle.py 里
_detect_scene_domain）：
    image,gt_distance,domain
    /data/indoor/0001.png,1.72,indoor
    /data/outdoor/0002.jpg,9.07,outdoor

manifest.jsonl 格式（每行一个 JSON 对象）：
    {"image": "/data/indoor/0001.png", "gt_distance": 1.72, "domain": "indoor"}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

_THRESHOLD_M = 2.0  # F1@2m，2026-08-09 确认（此前是 F1@1m，见模块 docstring）


def _load_local_adapter_class():
    """跟 test_obstacle.py 里的同名函数一样：在没装 rclpy 的开发机上也能跑，
    因为 plugins/obstacle.py 顶部为了 ROS2 Node 类需要 import rclpy，
    这里只用得到不依赖 ROS2 的 LocalDistanceAdapter，缺了就 stub 一下。
    """
    try:
        import rclpy  # noqa: F401
        from plugins.obstacle import LocalDistanceAdapter
        return LocalDistanceAdapter
    except ImportError:
        import types
        import sys as _sys

        class _EnumStub:
            RELIABLE = BEST_EFFORT = KEEP_LAST = KEEP_ALL = VOLATILE = TRANSIENT_LOCAL = 0

        class _QoSProfileStub:
            def __init__(self, *args, **kwargs):
                pass

        rclpy_stub = types.ModuleType("rclpy")
        rclpy_node_stub = types.ModuleType("rclpy.node")
        rclpy_node_stub.Node = object
        rclpy_qos_stub = types.ModuleType("rclpy.qos")
        rclpy_qos_stub.QoSProfile = _QoSProfileStub
        for _name in ("ReliabilityPolicy", "HistoryPolicy", "DurabilityPolicy"):
            setattr(rclpy_qos_stub, _name, _EnumStub)
        sensor_msgs_stub = types.ModuleType("sensor_msgs")
        sensor_msgs_msg_stub = types.ModuleType("sensor_msgs.msg")
        sensor_msgs_msg_stub.CompressedImage = object
        std_msgs_stub = types.ModuleType("std_msgs")
        std_msgs_msg_stub = types.ModuleType("std_msgs.msg")
        std_msgs_msg_stub.String = object

        for mod_name, mod in (
            ("rclpy", rclpy_stub),
            ("rclpy.node", rclpy_node_stub),
            ("rclpy.qos", rclpy_qos_stub),
            ("sensor_msgs", sensor_msgs_stub),
            ("sensor_msgs.msg", sensor_msgs_msg_stub),
            ("std_msgs", std_msgs_stub),
            ("std_msgs.msg", std_msgs_msg_stub),
        ):
            _sys.modules.setdefault(mod_name, mod)

        from plugins.obstacle import LocalDistanceAdapter
        return LocalDistanceAdapter


@dataclass
class _Sample:
    image: str
    gt_distance: float
    domain: Optional[str] = None


@dataclass
class _CaseResult:
    sample: _Sample
    pred_distance: Optional[float]
    elapsed_s: float


def _load_manifest(path: Path) -> list[_Sample]:
    samples = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(_Sample(image=row["image"], gt_distance=float(row["gt_distance"]),
                                    domain=row.get("domain")))
    else:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                samples.append(_Sample(image=row["image"], gt_distance=float(row["gt_distance"]),
                                        domain=(row.get("domain") or None)))
    return samples


def run_eval(manifest_path: Path, model_path: Optional[str] = None) -> dict:
    samples = _load_manifest(manifest_path)
    if not samples:
        raise ValueError(f"manifest is empty: {manifest_path}")

    LocalDistanceAdapter = _load_local_adapter_class()
    adapter = LocalDistanceAdapter(model_path)
    results: list[_CaseResult] = []

    for s in samples:
        image_bytes = Path(s.image).read_bytes()
        t0 = time.monotonic()
        out = adapter.estimate(image_bytes)
        elapsed = time.monotonic() - t0
        results.append(_CaseResult(sample=s, pred_distance=out.get("pred_distance"), elapsed_s=elapsed))

    return _summarize(results)


def _summarize(results: list[_CaseResult]) -> dict:
    n = len(results)
    # LocalDistanceAdapter.estimate() 出错/超时会返回常量兜底值而不是 None
    # （见 plugins/obstacle.py _FALLBACK_DISTANCE），这里没有一个专门的
    # "推理失败"信号可用，failure_rate 恒为 0——如果要统计真实失败率，
    # 需要 estimate() 把失败原因带出来，这次没有改这部分接口。
    tp = fp = fn = tn = 0
    sq_errors = []
    for r in results:
        gt_pos = r.sample.gt_distance < _THRESHOLD_M
        pred_pos = r.pred_distance < _THRESHOLD_M
        if gt_pos and pred_pos:
            tp += 1
        elif pred_pos and not gt_pos:
            fp += 1
        elif gt_pos and not pred_pos:
            fn += 1
        else:
            tn += 1
        sq_errors.append((r.pred_distance - r.sample.gt_distance) ** 2)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    rmse = (sum(sq_errors) / len(sq_errors)) ** 0.5 if sq_errors else float("nan")

    total_elapsed = sum(r.elapsed_s for r in results)
    fps = n / total_elapsed if total_elapsed > 0 else float("nan")

    return {
        "n_samples": n,
        "precision_at_2m": round(precision, 4),
        "recall_at_2m": round(recall, 4),
        "f1_at_2m": round(f1, 4),
        "rmse_m": round(rmse, 4) if rmse == rmse else None,
        "fps": round(fps, 2) if fps == fps else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description="本地复现 obstacle 赛道评测指标（F1@2m/RMSE/FPS）")
    parser.add_argument("manifest", type=str, help="manifest.csv 或 manifest.jsonl 路径")
    parser.add_argument("--model-path", default=None, help="覆盖默认模型目录")
    args = parser.parse_args()

    report = run_eval(Path(args.manifest), model_path=args.model_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
