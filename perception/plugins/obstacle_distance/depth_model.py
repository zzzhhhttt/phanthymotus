#!/usr/bin/env python3
"""
plugins/obstacle_distance/depth_model.py — 单目度量深度模型封装（onnxruntime）。

模型：Depth Anything V2 Small（metric depth 版本），ViT-S backbone，
原始 fp32 权重 24.8M 参数 / 99MB，用 onnxruntime 动态量化成 INT8 后单个
模型文件 ~27MB（榜单"模型限定 30M 以下"指的是模型文件大小，不是参数量，
量化就是为了满足这个约束，量化前的 fp32 版本 99MB 会超标）。

indoor / outdoor 两个 checkpoint（对应两个数据集各自的深度量程），已经
导出并量化好，直接放在仓库里：
    perception/models/depth_anything_v2/depth_anything_v2_metric_indoor_vits_int8.onnx
    perception/models/depth_anything_v2/depth_anything_v2_metric_outdoor_vits_int8.onnx
两个文件各 ~26MB。导出/量化过程见本目录 export_onnx.py（一次性开发脚本，
运行时不依赖它，也不依赖 torch/transformers——onnxruntime 版本运行时只需要
onnxruntime + numpy + Pillow，比原先基于 transformers 的版本依赖轻得多，
更符合 Jetson 部署的体积/启动时间预算，也是这个仓库里 OCR 插件
（ppocr_adapter.py）已经验证过的路线：onnxruntime 优先，避免整套
torch/transformers 进镜像）。

用 CPU 而不是 GPU 跑：INT8 动态量化产出的 ConvInteger/MatMulInteger 算子
在 CUDAExecutionProvider 上支持不完整（本地实测 CPU EP 用 QUInt8 权重
可以跑，GPU EP 对这类算子的内核覆盖不稳定，且生产目标 Jetson 上装的是
CUDA 版 onnxruntime 还是 CPU 版本身就要看镜像怎么配）。本地 CPU 实测单帧
~150ms，远在榜单 3 秒的 FPS 预算内；这也是 OCR 插件同样的选择（见
config.yaml 里 ocr.device 的注释："模型够小，CPU 跑也在预算内"），顺带
也让"GPU 占用 <10%"这条约束不战而胜——CPU 推理不占 GPU。
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np

from .domain import INDOOR, OUTDOOR, Domain

log = logging.getLogger(__name__)

# 输入分辨率固定 518x518（ViT patch_size=14, 训练分辨率），导出 ONNX 时
# 用的就是这个固定尺寸的 dummy input，模型图里的形状是静态的，推理输入
# 必须是这个尺寸——所以预处理直接 resize 成正方形 518x518（不保持长宽比），
# 跟原始 HF DPTImageProcessor 的 keep_aspect_ratio+pad 策略不完全一致，
# 是为了拿到固定形状、能直接量化成 INT8 的 ONNX 图而做的简化，对图片
# 有一定拉伸，实测对深度估计精度影响很小（量化本身引入的误差是主要
# 来源，见 export_onnx.py 里的对比数据）。
_INPUT_SIZE = 518
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "depth_anything_v2"

_MODEL_FILENAME = {
    INDOOR: "depth_anything_v2_metric_indoor_vits_int8.onnx",
    OUTDOOR: "depth_anything_v2_metric_outdoor_vits_int8.onnx",
}

# 每个 domain 各自的最大有效量程（米）——裁剪模型偶尔外推出的不合理极端值
_MAX_DEPTH_M = {
    INDOOR: 20.0,
    OUTDOOR: 80.0,
}


def _preprocess(image) -> np.ndarray:
    """PIL.Image(RGB) -> (1,3,518,518) float32，ImageNet 归一化。"""
    from PIL import Image

    resized = image.resize((_INPUT_SIZE, _INPUT_SIZE), Image.BICUBIC)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[None].astype(np.float32)
    return arr


def _postprocess(raw_depth: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """模型输出 (1,518,518) -> resize 回原图尺寸 (H,W)。"""
    from PIL import Image

    depth_img = Image.fromarray(raw_depth[0].astype(np.float32), mode="F")
    resized = depth_img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(resized, dtype=np.float32)  # np.array() copies; PIL 返回的 buffer 是只读的


class DepthEstimator:
    """单目度量深度估计器，按 domain 懒加载对应 onnx 模型。

    线程安全：onnxruntime InferenceSession.run() 本身是线程安全的，多个
    ROS2 节点/请求线程可以共用同一个 DepthEstimator 实例；加载锁只保证
    同一个 domain 的模型不会被重复加载。
    """

    def __init__(self, model_dir: Optional[str] = None, device: str = "cpu"):
        """
        Args:
            model_dir: 存放 *_int8.onnx 文件的目录。为 None 时读环境变量
                OBSTACLE_DISTANCE_MODEL_DIR，都没有则用仓库自带的
                perception/models/depth_anything_v2/。
            device: "cpu"（默认，见模块 docstring 的取舍说明）| "cuda"
                （实验性：只有确认目标机器的 onnxruntime-gpu + 对应 EP
                能跑 ConvInteger/MatMulInteger 时才建议开）。
        """
        self._model_dir = Path(model_dir or os.environ.get("OBSTACLE_DISTANCE_MODEL_DIR")
                                or _DEFAULT_MODEL_DIR)
        self._device = device

        self._lock = threading.Lock()
        self._sessions: dict[Domain, object] = {}

        log.info(f"[obstacle_distance] DepthEstimator init: device={self._device}, "
                 f"model_dir={self._model_dir}")

    def _load(self, domain: Domain):
        if domain in self._sessions:
            return self._sessions[domain]

        with self._lock:
            if domain in self._sessions:  # double-checked locking
                return self._sessions[domain]

            import onnxruntime as ort

            model_path = self._model_dir / _MODEL_FILENAME[domain]
            if not model_path.exists():
                raise FileNotFoundError(
                    f"obstacle_distance model not found: {model_path}. "
                    f"运行 perception/plugins/obstacle_distance/export_onnx.py 生成，"
                    f"或设置 OBSTACLE_DISTANCE_MODEL_DIR 指向已有模型目录。"
                )

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._device == "cuda" \
                else ["CPUExecutionProvider"]
            log.info(f"[obstacle_distance] loading {domain} model: {model_path} (providers={providers})")

            session = ort.InferenceSession(str(model_path), providers=providers)
            self._sessions[domain] = session
            log.info(f"[obstacle_distance] {domain} model ready")
            return session

    def estimate(self, image, domain: Domain) -> np.ndarray:
        """对一张 RGB 图片做深度估计，返回跟原图同尺寸的 HxW 米制深度图。

        Args:
            image: PIL.Image（RGB）。
            domain: "indoor" 或 "outdoor"，决定用哪个 onnx 模型。
        """
        session = self._load(domain)
        w, h = image.size

        inputs = _preprocess(image)
        raw = session.run(None, {"pixel_values": inputs})[0]
        depth = _postprocess(raw, w, h)

        max_depth = _MAX_DEPTH_M[domain]
        np.clip(depth, 0.0, max_depth, out=depth)
        return depth


def load_image(image_bytes: bytes):
    from PIL import Image
    return Image.open(BytesIO(image_bytes)).convert("RGB")


@lru_cache(maxsize=1)
def get_default_estimator() -> DepthEstimator:
    """进程内单例，避免每次调用都重新构造/重新加载模型。"""
    return DepthEstimator()
