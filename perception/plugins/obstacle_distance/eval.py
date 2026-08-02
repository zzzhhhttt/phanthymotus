#!/usr/bin/env python3
"""
plugins/obstacle_distance/eval.py — 本地复现榜单评测指标，方便提交前自测。

复现的指标（对应榜单文档《4. 评测标准》）：
    - Precision@1m / Recall@1m / F1@1m（主排序指标，阈值 1 米）
    - RMSE（米）
    - FPS（每秒处理图片数）
    - 推理失败率

用法：
    python3 eval.py manifest.csv
    python3 eval.py manifest.jsonl --domain indoor

manifest.csv 格式（表头必须包含 image,gt_distance，domain 可选）：
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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .depth_model import get_default_estimator
from .predict import predict_distance

_THRESHOLD_M = 1.0


@dataclass
class _Sample:
    image: str
    gt_distance: float
    domain: Optional[str] = None


@dataclass
class _CaseResult:
    sample: _Sample
    pred_distance: Optional[float]
    error: Optional[str]
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


def run_eval(manifest_path: Path, domain_override: Optional[str] = None,
             percentile: float = 1.0) -> dict:
    samples = _load_manifest(manifest_path)
    if not samples:
        raise ValueError(f"manifest is empty: {manifest_path}")

    estimator = get_default_estimator()
    results: list[_CaseResult] = []

    for s in samples:
        image_bytes = Path(s.image).read_bytes()
        t0 = time.monotonic()
        out = predict_distance(
            image_bytes, filename=s.image,
            domain=domain_override or s.domain,
            percentile=percentile, estimator=estimator,
        )
        elapsed = time.monotonic() - t0
        results.append(_CaseResult(
            sample=s, pred_distance=out.get("pred_distance"),
            error=out.get("error"), elapsed_s=elapsed,
        ))

    return _summarize(results)


def _summarize(results: list[_CaseResult]) -> dict:
    n = len(results)
    failed = [r for r in results if r.pred_distance is None]
    ok = [r for r in results if r.pred_distance is not None]

    tp = fp = fn = tn = 0
    sq_errors = []
    for r in ok:
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
    failure_rate = len(failed) / n if n else 0.0

    return {
        "n_samples": n,
        "n_failed": len(failed),
        "failure_rate": round(failure_rate, 4),
        "precision_at_1m": round(precision, 4),
        "recall_at_1m": round(recall, 4),
        "f1_at_1m": round(f1, 4),
        "rmse_m": round(rmse, 4) if rmse == rmse else None,
        "fps": round(fps, 2) if fps == fps else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "failed_images": [r.sample.image for r in failed],
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description="本地复现榜单评测指标")
    parser.add_argument("manifest", type=str, help="manifest.csv 或 manifest.jsonl 路径")
    parser.add_argument("--domain", choices=["indoor", "outdoor"], default=None,
                         help="强制指定所有样本的场景域，默认按每条样本的 domain 字段/图片格式自动判定")
    parser.add_argument("--percentile", type=float, default=1.0)
    args = parser.parse_args()

    report = run_eval(Path(args.manifest), domain_override=args.domain, percentile=args.percentile)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
