#!/usr/bin/env python3
"""
plugins/obstacle_distance/export_onnx.py — 一次性开发脚本：把 HuggingFace
上的 Depth Anything V2 Small metric checkpoint 导出成 ONNX 再做 INT8 动态
量化，产出的文件已经放在 perception/models/depth_anything_v2/ 下，正常
使用（predict.py / eval.py / plugin.py）不需要跑这个脚本、也不需要装
torch/transformers。

只有在需要重新生成模型文件时才用得到（比如切换到更大的 backbone、换一个
量化策略）。依赖比运行时重得多：
    pip install torch transformers torchvision onnx onnxruntime

用法：
    python3 export_onnx.py indoor
    python3 export_onnx.py outdoor
    python3 export_onnx.py all      # 两个都导出

产出：
    perception/models/depth_anything_v2/depth_anything_v2_metric_{domain}_vits_int8.onnx

选型说明（为什么是这几步，不是别的）：
    1. fp32 safetensors 原始权重 24.8M 参数 = 99MB，超过榜单"模型 30M
       以下"（文件大小）的限制 3 倍多，必须压缩。
    2. FP16 只能减半到 ~50MB，仍然超标，INT8（理论 4x）才够。
    3. onnxruntime 的 INT8 量化有两条路：
       - quantize_dynamic 权重量化：产出 ConvInteger/MatMulInteger 算子。
         实测 QInt8 权重在 CPUExecutionProvider 上会报
         "Could not find an implementation for ConvInteger"；换成
         QUInt8 权重可以正常跑，量化后 ~27MB，跟 fp32 基线相比在真实
         图片上平均绝对误差约 0.02m（相对误差约 1%），可接受。
       - 静态 QDQ 量化 + TensorRT EP：更适合 GPU/TensorRT 部署，但需要
         一批有代表性的标定图片，且没有真实 Jetson 设备可以本地验证
         TensorRT EP 对 QDQ 图的实际支持情况，风险更高，这次没走这条路。
    4. 量化后跑 CPU（不是 GPU）：见 depth_model.py 模块 docstring，
       ConvInteger/MatMulInteger 在 CUDAExecutionProvider 上支持不稳定，
       CPU 单帧 ~150ms 远在榜单 3 秒预算内，且这也是仓库里 OCR 插件
       （ppocr_adapter）已经验证过的同一个取舍。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "depth_anything_v2"

_HF_REPO = {
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
}

_INPUT_SIZE = 518


def _export_one(domain: str) -> None:
    import torch
    from transformers import AutoModelForDepthEstimation
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    repo = _HF_REPO[domain]
    fp32_path = _MODEL_DIR / f"_tmp_{domain}_fp32.onnx"
    pre_path = _MODEL_DIR / f"_tmp_{domain}_fp32_pre.onnx"
    out_path = _MODEL_DIR / f"depth_anything_v2_metric_{domain}_vits_int8.onnx"

    print(f"[{domain}] loading {repo} ...")
    model = AutoModelForDepthEstimation.from_pretrained(repo).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{domain}] params: {n_params/1e6:.1f}M")

    dummy = torch.randn(1, 3, _INPUT_SIZE, _INPUT_SIZE)
    print(f"[{domain}] exporting fp32 onnx ...")
    torch.onnx.export(
        model, (dummy,), str(fp32_path),
        input_names=["pixel_values"], output_names=["predicted_depth"],
        opset_version=17, dynamo=False,
    )

    print(f"[{domain}] pre-processing (shape inference) ...")
    quant_pre_process(str(fp32_path), str(pre_path))

    print(f"[{domain}] quantizing to INT8 (QUInt8 weights) ...")
    quantize_dynamic(str(pre_path), str(out_path), weight_type=QuantType.QUInt8)

    fp32_path.unlink(missing_ok=True)
    pre_path.unlink(missing_ok=True)

    size_mb = out_path.stat().st_size / 1e6
    print(f"[{domain}] done -> {out_path} ({size_mb:.1f} MB)")
    if size_mb > 30:
        print(f"[{domain}] WARNING: {size_mb:.1f}MB exceeds the 30MB budget!", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 + 量化 Depth Anything V2 Small metric 模型")
    parser.add_argument("domain", choices=["indoor", "outdoor", "all"])
    args = parser.parse_args()

    domains = ["indoor", "outdoor"] if args.domain == "all" else [args.domain]
    for d in domains:
        _export_one(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
