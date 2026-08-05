#!/usr/bin/env python3
"""
plugins/obstacle.py — ObstacleDistancePlugin: obstacle distance estimation.

Subscribes to image/jpeg topics, estimates obstacle distance from camera,
publishes distance results to ROS2 topic.
Supports multi-instance (one instance per input topic).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeoutError
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

log = logging.getLogger(__name__)

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    durability=DurabilityPolicy.VOLATILE,
)

_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "obstacle",
        "type": "processor",
        "multiInstance": True,
        "description": "Obstacle Distance Estimation — estimate distance to obstacles from camera feed",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 image topic to subscribe (e.g. /hostname/camera/rgb, required for action=start)"
                },
            },
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "enum": ["openai", "qwen", "local"], "description": "Distance estimation provider", "scope": "shared"},
                "url":      {"type": "string", "description": "API URL (optional)", "scope": "shared"},
                "key":      {"type": "string", "description": "API Key", "format": "password", "scope": "shared"},
                "model":    {"type": "string", "description": "Model name", "scope": "instance"},
            },
            "required": ["provider"]
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "obstacle distance estimation result"}],
    }
]


# ── Distance Estimation Adapters ──────────────────────────────────────────────

class DistanceAdapter(ABC):
    """障碍物距离估计适配器抽象基类"""

    @abstractmethod
    def estimate(self, image_bytes: bytes) -> dict:
        """估计图片中障碍物的距离，返回包含 pred_distance 的字典"""
        ...


class OpenAIVisionDistanceAdapter(DistanceAdapter):
    """OpenAI Vision API 距离估计"""

    _SYSTEM_PROMPT = (
        "You are an obstacle distance estimation system for a robot camera.\n\n"
        "Your task is to analyze the provided image and estimate the distance "
        "to the nearest obstacle in meters.\n\n"
        "Output format: Return a JSON object with:\n"
        '- "pred_distance": estimated distance in meters (float)\n'
        '- "confidence": confidence score 0-1 (float)\n'
        '- "reasoning": brief explanation of your estimation\n\n'
        "Rules:\n"
        "1. Distance should be in meters.\n"
        "2. If no obstacle is visible, return a large value (e.g., 10.0).\n"
        "3. Be precise — typical indoor distances range from 0.3m to 5m.\n"
        "4. Output ONLY the JSON object, nothing else.\n\n"
        'Example: {"pred_distance": 1.25, "confidence": 0.85, "reasoning": "clear wall visible"}'
    )

    def __init__(self, url: str, key: str, model: str):
        self.base_url = url.rstrip('/') if url else "https://api.openai.com/v1"
        self.key = key
        self.model = model or "gpt-4o-mini"

    def estimate(self, image_bytes: bytes) -> dict:
        import requests
        import base64

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        image_format = "jpeg"
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            image_format = "png"
        elif image_bytes[:2] == b'BM':
            image_format = "bmp"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            image_format = "webp"

        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_b64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Estimate the distance to the nearest obstacle in this image."
                    }
                ]
            }
        ]

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 512,
            },
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._parse_result(content)

    @staticmethod
    def _parse_result(content: str) -> dict:
        """解析模型返回的 JSON 结果"""
        content = content.strip()
        if content.startswith("{"):
            try:
                parsed = json.loads(content)
                return {
                    "pred_distance": float(parsed.get("pred_distance", 10.0)),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "reasoning": parsed.get("reasoning", ""),
                }
            except (json.JSONDecodeError, ValueError):
                pass
        # 尝试从 markdown 代码块中提取
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                return {
                    "pred_distance": float(parsed.get("pred_distance", 10.0)),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "reasoning": parsed.get("reasoning", ""),
                }
            except (json.JSONDecodeError, ValueError):
                pass
        # 兜底：尝试提取数字
        import re
        numbers = re.findall(r"\d+\.?\d*", content)
        if numbers:
            try:
                return {"pred_distance": float(numbers[0]), "confidence": 0.5, "reasoning": content[:200]}
            except ValueError:
                pass
        log.warning(f"[obstacle] failed to parse distance result, returning default: {content[:200]!r}")
        return {"pred_distance": 10.0, "confidence": 0.0, "reasoning": "parse failed"}


class QwenVLDistanceAdapter(DistanceAdapter):
    """Qwen-VL 距离估计"""

    _SYSTEM_PROMPT = (
        "你是一个机器人摄像头障碍物距离估计系统。\n\n"
        "任务：分析提供的图片，估计最近障碍物的距离（单位：米）。\n\n"
        "输出格式：返回 JSON 对象，包含：\n"
        '- "pred_distance": 估计距离（米，浮点数）\n'
        '- "confidence": 置信度 0-1（浮点数）\n'
        '- "reasoning": 简要说明\n\n'
        "规则：\n"
        "1. 距离单位为米。\n"
        "2. 如果没有可见障碍物，返回较大值（如 10.0）。\n"
        "3. 室内典型距离范围：0.3m 到 5m。\n"
        "4. 只输出 JSON 对象，不要其他内容。\n\n"
        '示例：{"pred_distance": 1.25, "confidence": 0.85, "reasoning": "清晰可见的墙壁"}'
    )

    def __init__(self, url: str, key: str, model: str):
        self.base_url = url.rstrip('/') if url else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.key = key
        self.model = model or "qwen-vl-max"

    def estimate(self, image_bytes: bytes) -> dict:
        import requests
        import base64

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        image_format = "jpeg"
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            image_format = "png"

        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/{image_format};base64,{image_b64}"
                    },
                    {
                        "type": "text",
                        "text": "估计这张图片中最近障碍物的距离。"
                    }
                ]
            }
        ]

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 512,
            },
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return OpenAIVisionDistanceAdapter._parse_result(content)


# ── Local (ONNX) distance estimation — scene-domain sub-logic ────────────────
#
# 榜单输入说明：png（等效焦距 29mm）图像来自室内机器人数据集，jpg（等效
# 焦距 33mm）图像来自无人车数据集，两者采集时格式从未混用过，所以图片
# 容器格式本身就是场景域的标签，不需要单独训练/跑一个分类器。用文件
# magic bytes 判定（比只看扩展名可靠——扩展名可能被改写，内容不会）。
# EXIF 焦距字段本质上是同一个签名的另一种表现形式，不作为独立判据用，
# 避免两个信号打架时不知道信任哪个。
#
# 独立成模块级函数，方便单测：不需要构造 LocalDistanceAdapter（也就不需要
# 加载 onnx 模型）就能验证判定逻辑本身。
def _detect_scene_domain(image_bytes: bytes, filename: Optional[str] = None) -> str:
    """返回 'indoor' 或 'outdoor'。

    Indoor: PNG。Outdoor: JPEG。判不出来时默认 indoor 并记录日志——室内
    模型的度量深度量程较小（~0-20m），即便误用在室外图像上，对"最近
    障碍物"这个近距离场景的可用性影响也比反过来（用室外模型判室内近距
    离）更小，是两个都不确定时更保守的选择。
    """
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "indoor"
    if image_bytes[:2] == b'\xff\xd8':
        return "outdoor"

    if filename:
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext == "png":
            return "indoor"
        if ext in ("jpg", "jpeg"):
            return "outdoor"

    log.warning("[obstacle][local] cannot determine scene domain from image bytes/filename, "
                "defaulting to 'indoor'")
    return "indoor"


# ── Local (ONNX) distance estimation — ROI sub-logic ──────────────────────────
#
# 榜单文档室内场景 ROI（640x480 参考分辨率）：col 213~426（宽度中间
# 1/3），row 0~300（上方 5/8，剔除下方 3/8 地面区域）。213/640≈1/3，
# 300/480=5/8，用比例表示，按实际输入分辨率（H, W）重新换算，不写死
# 像素值。榜单文档没有单独给出无人车场景的图像空间 ROI 定义（它的真值
# 基于 3D 标注框，不是图像裁剪+百分位数），这里对 outdoor 复用同一个
# 比例——不新发明一个宽/窄框，避免引入没有依据支撑的假设。
def _compute_roi_bounds(height: int, width: int) -> tuple[int, int, int, int]:
    """返回 (row_start, row_end, col_start, col_end)，均为像素索引（含 start，不含 end）。"""
    col_start = int(round(width * (1.0 / 3.0)))
    col_end = int(round(width * (2.0 / 3.0)))
    row_start = 0
    row_end = int(round(height * (5.0 / 8.0)))
    return row_start, row_end, col_start, col_end


# ── Local (ONNX) distance estimation — 稳健取值 sub-logic ─────────────────────
#
# 不直接取 ROI 内的 min()：单目深度模型在物体边缘/深度不连续处容易出现
# 孤立噪声像素，min() 对这类噪声极其敏感，一个坏点就能把整帧的距离读数
# 拉飞。先做中值滤波去掉孤立噪声像素（对真实物体表面这种连续区域几乎
# 没有副作用），再取 P1 分位数（而不是 P0=min）作为"最近障碍物距离"，
# 对滤波后仍存在的分布尾部更鲁棒。
#
# 曾经尝试过把单一 P1 点值换成 [P1, P1+4] 百分位区间的均值，指望进一步
# 压制个别残留像素的影响——但自己写了个合成测试验证后发现这个方向是
# 错的：真实的小尺寸障碍物（占 ROI 面积 1%~5% 之间，比如门框边缘、家具
# 细腿这类量级）会被这段区间里混进来的背景像素拉平，读数从正确的"近"
# 被拉回"远"，反而丢失了纯 P1 点估计能正确检出的案例。换句话说，区间
# 均值不是"单调更保守"，它在这个尺寸区间上是净负向的——没有 GT 数据也
# 能用合成数据证伪，所以没有采用，退回原始单点百分位数。
def _robust_nearest_distance(depth_roi: np.ndarray, percentile: float = 1.0) -> Optional[float]:
    """输入 ROI 内的深度值（2D array，单位米），返回稳健的最近距离估计。

    ROI 为空、或滤波后没有任何有限正值时返回 None（由调用方决定 fallback）。
    """
    if depth_roi.size == 0:
        return None

    try:
        from scipy.ndimage import median_filter
        filtered = median_filter(depth_roi, size=5)
    except ImportError:
        # scipy 不可用时退化为不滤波，仍然用分位数而不是 min() 兜底噪声敏感问题
        log.warning("[obstacle][local] scipy not available, skipping median-filter denoise")
        filtered = depth_roi

    valid = filtered[np.isfinite(filtered) & (filtered > 0)]
    if valid.size == 0:
        return None

    return float(np.percentile(valid, percentile))


class LocalDistanceAdapter(DistanceAdapter):
    """本地距离估计：基于 Depth Anything V2（metric, ONNX, int8 量化）的单目度量深度推理。

    室内/室外分别用各自的量程校准 checkpoint（见 `_detect_scene_domain`），
    两个 onnxruntime session 在 __init__ 里各加载一次、常驻内存，estimate()
    不重复加载模型。
    """

    _INPUT_SIZE = 518  # ViT patch_size=14 训练分辨率，导出 onnx 时用的固定尺寸
    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    _MODEL_FILENAME = {
        "indoor": "depth_anything_v2_metric_indoor_vits_int8.onnx",
        "outdoor": "depth_anything_v2_metric_outdoor_vits_int8.onnx",
    }
    _MODEL_SIZE_BUDGET_MB = 30.0

    _FALLBACK_DISTANCE = 10.0
    # 单帧总处理时间硬性预算（场景判别 + 预处理 + 推理 + ROI/后处理），
    # 留出安全边际给 ROS2 序列化/发布开销，硬约束是 3 秒。
    _TIMEOUT_SECONDS = 2.7
    _DEFAULT_PERCENTILE = 1.0

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: 可选，覆盖默认模型目录（需包含
                depth_anything_v2_metric_{indoor,outdoor}_vits_int8.onnx 两个文件）。
                传入单个文件路径时会退化为使用其所在目录。不传则默认指向
                perception/models/depth_anything_v2/（仓库自带，已量化好）。
                通过 OBSTACLE_LOCAL_MODEL_DIR 环境变量也可以覆盖，优先级低于
                显式传参。
        """
        env_dir = os.environ.get("OBSTACLE_LOCAL_MODEL_DIR")
        chosen = model_path or env_dir
        if chosen:
            p = Path(chosen)
            self.model_dir = p if p.is_dir() else p.parent
        else:
            # perception/plugins/obstacle.py -> perception/models/depth_anything_v2
            self.model_dir = Path(__file__).resolve().parent.parent / "models" / "depth_anything_v2"

        self._percentile = float(os.environ.get("OBSTACLE_LOCAL_PERCENTILE", self._DEFAULT_PERCENTILE))
        self._sessions: dict[str, object] = {}
        self._input_names: dict[str, str] = {}
        # 用于给单帧推理加硬超时保护；估计耗时通常在几百毫秒量级，池子大小
        # 只需要能容纳"少数几帧同时卡住"的场景，不需要很大。
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="obstacle_local_infer")

        self._load_models()

    # ── 模型加载 ──────────────────────────────────────────────────────────

    def _resolve_providers(self) -> list[str]:
        import onnxruntime as ort

        device = os.environ.get("OBSTACLE_LOCAL_DEVICE", "auto").lower()
        available = ort.get_available_providers()

        if device == "cpu":
            return ["CPUExecutionProvider"]
        if device == "cuda":
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            log.warning("[obstacle][local] OBSTACLE_LOCAL_DEVICE=cuda but CUDAExecutionProvider "
                        "unavailable, falling back to CPU")
            return ["CPUExecutionProvider"]
        # auto: 量化模型的 ConvInteger/MatMulInteger 算子在部分 CUDA EP 版本上
        # 支持不稳定，优先尝试 CUDA，session 创建失败时在 _load_models 里整体
        # 回退到 CPU-only provider 列表重试。
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_models(self):
        import onnxruntime as ort

        providers = self._resolve_providers()

        for domain, filename in self._MODEL_FILENAME.items():
            path = self.model_dir / filename
            if not path.exists():
                log.error(f"[obstacle][local] model file not found for domain={domain}: {path}")
                continue

            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self._MODEL_SIZE_BUDGET_MB:
                log.warning(f"[obstacle][local] model {path.name} is {size_mb:.1f}MB, "
                            f"exceeds {self._MODEL_SIZE_BUDGET_MB}MB budget")

            try:
                session = ort.InferenceSession(str(path), providers=providers)
            except Exception as e:
                log.error(f"[obstacle][local] failed to load {path} with providers={providers}: {e}; "
                          f"retrying with CPUExecutionProvider only")
                session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

            actual_providers = session.get_providers()
            log.info(f"[obstacle][local] loaded {domain} model: {path.name} "
                     f"({size_mb:.1f}MB), providers={actual_providers}")

            self._sessions[domain] = session
            self._input_names[domain] = session.get_inputs()[0].name

    # ── 预处理 ────────────────────────────────────────────────────────────

    @classmethod
    def _preprocess(cls, image_bytes: bytes) -> np.ndarray:
        from PIL import Image

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        # 固定 resize 成 518x518（不保持长宽比）：onnx 图的输入形状是静态的
        # （导出/int8 量化都基于这个固定尺寸），对图片有一定拉伸，对深度估计
        # 精度影响较小（量化本身引入的误差是主要来源）。因为是各轴独立拉伸，
        # ROI 的行/列比例边界在原图和 518x518 depth map 上是等价的，所以
        # ROI 直接在 depth map 尺寸上按比例计算即可，不需要额外坐标换算。
        resized = img.resize((cls._INPUT_SIZE, cls._INPUT_SIZE), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - cls._IMAGENET_MEAN) / cls._IMAGENET_STD
        arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)  # NCHW
        return arr

    # ── 推理核心（跑在线程池里，配合超时保护）───────────────────────────────

    def _infer_once(self, image_bytes: bytes) -> dict:
        domain = _detect_scene_domain(image_bytes)
        session = self._sessions.get(domain)
        if session is None:
            log.error(f"[obstacle][local] no loaded session for domain={domain}")
            return {"pred_distance": self._FALLBACK_DISTANCE, "domain": domain}

        try:
            inp = self._preprocess(image_bytes)
        except Exception as e:
            log.error(f"[obstacle][local] preprocess failed: {e}", exc_info=True)
            return {"pred_distance": self._FALLBACK_DISTANCE, "domain": domain}

        try:
            outputs = session.run(None, {self._input_names[domain]: inp})
        except Exception as e:
            log.error(f"[obstacle][local] onnxruntime inference failed (domain={domain}): {e}",
                      exc_info=True)
            return {"pred_distance": self._FALLBACK_DISTANCE, "domain": domain}

        depth = np.squeeze(outputs[0]).astype(np.float32)  # HxW metric depth (meters)
        if depth.ndim != 2:
            log.error(f"[obstacle][local] unexpected depth output shape={depth.shape} (domain={domain})")
            return {"pred_distance": self._FALLBACK_DISTANCE, "domain": domain}

        row_start, row_end, col_start, col_end = _compute_roi_bounds(*depth.shape)
        roi = depth[row_start:row_end, col_start:col_end]

        dist = _robust_nearest_distance(roi, percentile=self._percentile)
        if dist is None:
            log.warning(f"[obstacle][local] empty ROI or no valid depth pixels (domain={domain}), "
                        f"falling back to default distance")
            return {"pred_distance": self._FALLBACK_DISTANCE, "domain": domain}

        return {"pred_distance": round(dist, 4), "domain": domain}

    # ── 对外接口 ──────────────────────────────────────────────────────────

    def estimate(self, image_bytes: bytes) -> dict:
        """返回 {"pred_distance": <float, 米>}，异常/超时一律 fallback 到常量默认值，不抛出。"""
        if not self._sessions:
            log.error("[obstacle][local] no models loaded, returning fallback distance")
            return {"pred_distance": self._FALLBACK_DISTANCE}

        t0 = time.monotonic()
        try:
            future = self._executor.submit(self._infer_once, image_bytes)
            try:
                result = future.result(timeout=self._TIMEOUT_SECONDS)
            except _FutureTimeoutError:
                # 硬超时：立即返回兜底值，不等待底层 onnxruntime 调用完成
                # （该线程会在池子里继续跑到结束，结果被丢弃）。这是唯一能在
                # onnxruntime.run() 本身异常慢时，仍然保证对外接口在预算内
                # 返回结果的办法——它是一次同步 C 调用，没法从外部中途打断。
                log.error(f"[obstacle][local] inference exceeded {self._TIMEOUT_SECONDS}s budget, "
                          f"returning fallback distance")
                return {"pred_distance": self._FALLBACK_DISTANCE}
        except Exception as e:
            log.error(f"[obstacle][local] unexpected error: {e}", exc_info=True)
            return {"pred_distance": self._FALLBACK_DISTANCE}

        elapsed = time.monotonic() - t0
        if elapsed > self._TIMEOUT_SECONDS:
            log.warning(f"[obstacle][local] inference finished late ({elapsed*1000:.0f}ms, "
                        f"budget={self._TIMEOUT_SECONDS*1000:.0f}ms)")
        return result


def _build_distance_adapter(cfg: dict) -> Optional[DistanceAdapter]:
    """根据配置创建距离估计适配器"""
    provider = cfg.get('provider', 'local')

    if provider == 'openai':
        url, key = cfg.get('url', ''), cfg.get('key', '')
        if not key:
            return None
        return OpenAIVisionDistanceAdapter(url, key, cfg.get('model', ''))

    elif provider == 'qwen':
        url, key = cfg.get('url', ''), cfg.get('key', '')
        if not key:
            return None
        return QwenVLDistanceAdapter(url, key, cfg.get('model', ''))

    elif provider == 'local':
        return LocalDistanceAdapter(cfg.get('model_path'))

    return None


# ── ROS2 Node (one per instance/topic) ────────────────────────────────────────

class _ObstacleNode(Node):
    """Per-topic obstacle distance estimation node."""

    def __init__(self, input_topic: str, adapter: DistanceAdapter,
                 node_suffix: str):
        super().__init__(f"obstacle_{node_suffix}")
        self._input_topic = input_topic
        self._output_topic = f"{input_topic}/obstacle"
        self._adapter = adapter

        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._sub: Optional[object] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._detect_count = 0
        self.state = "idle"

    def start(self) -> dict:
        if self._sub is not None:
            self.state = "running"
            return {"state": "running", "input": self._input_topic, "output": self._output_topic}
        self._stop_event.clear()
        self._sub = self.create_subscription(
            CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
        )
        self._worker = threading.Thread(target=self._inference_worker, daemon=True,
                                         name=f"obstacle_worker_{self._input_topic}")
        self._worker.start()
        self.state = "running"
        log.info(f"[obstacle] started: {self._input_topic} -> {self._output_topic}")
        return {"state": "running", "input": self._input_topic, "output": self._output_topic}

    def stop(self) -> dict:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self._worker = None
        self.state = "idle"
        log.info(f"[obstacle] stopped: {self._input_topic}")
        return {"state": "idle", "input": self._input_topic}

    def _image_cb(self, msg: CompressedImage):
        log.info(
            f"[obstacle] received image frame: size={len(msg.data)} bytes, format={msg.format}, topic={self._input_topic}")
        # Drop old frame if queue full (no backpressure)
        try:
            self._frame_queue.put_nowait(msg.data)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(msg.data)
            except queue.Full:
                pass

    def _inference_worker(self):
        while not self._stop_event.is_set():
            try:
                jpeg_bytes = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                result = self._adapter.estimate(jpeg_bytes)
                self._publish_result(result)
            except Exception as e:
                log.error(f"[obstacle] inference error: {e}", exc_info=True)

    def _publish_result(self, result: dict):
        self._detect_count += 1
        msg = String()
        msg.data = json.dumps({
            "pred_distance": result.get("pred_distance", 10.0),
        }, ensure_ascii=False)
        self._pub.publish(msg)


# ── Plugin class ──────────────────────────────────────────────────────────────

class ObstacleDistancePlugin:
    PREFIX = "obstacle"

    def __init__(self, plugin_cfg: dict, executor):
        self._executor = executor
        self._provider = plugin_cfg.get("provider", "local")
        self._url = plugin_cfg.get("url", "")
        self._key = plugin_cfg.get("key", "")
        self._model = plugin_cfg.get("model", "")
        self._model_path = plugin_cfg.get("model_path")
        self._adapter = _build_distance_adapter(plugin_cfg)
        self._nodes: dict[str, _ObstacleNode] = {}
        self._instance_configs: dict[str, dict] = {}

        log.info(f"[obstacle] plugin init: provider={self._provider}, "
                 f"key={'set' if self._key else 'MISSING'}")

        if not self._adapter:
            log.warning("[obstacle] adapter not configured (missing key or invalid provider)")

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            instances = {}
            for key, node in self._nodes.items():
                instances[key] = {
                    "input": node._input_topic,
                    "output": node._output_topic,
                    "detect_count": node._detect_count,
                }
            input_topic = args.get("input_topic", "")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                input_topic = node._input_topic
            elif not input_topic and self._nodes:
                first_node = next(iter(self._nodes.values()))
                input_topic = first_node._input_topic
            topics_in = [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else []
            topics_out = [{"topic": f"{input_topic}/obstacle", "format": "data/json"}] if input_topic else []
            state = "running" if instances else "idle"
            return {
                "name": "ObstacleDistance", "manufacture": "Embodied", "model": "obstacle",
                "state": state,
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "Obstacle distance estimation from camera feed",
            }

        elif action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if not input_topic:
                raise ValueError("input_topic is required")
            node_key = instance_id or input_topic
            if node_key not in self._nodes:
                icfg = self._instance_configs.get(node_key, {})
                # Build adapter for this instance if config differs
                adapter = self._adapter
                if icfg:
                    adapter = _build_distance_adapter(icfg) or self._adapter
                suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
                node = _ObstacleNode(input_topic, adapter, suffix)
                self._executor.add_node(node)
                self._nodes[node_key] = node
                node.start()
            return self._nodes[node_key].start()

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                result = node.stop()
                self._executor.remove_node(node)
                del self._nodes[instance_id]
                return result
            elif not instance_id and self._nodes:
                results = []
                for key in list(self._nodes.keys()):
                    node = self._nodes[key]
                    node.stop()
                    self._executor.remove_node(node)
                    del self._nodes[key]
                    results.append(key)
                return {"state": "idle", "stopped_instances": results}
            return {"state": "idle"}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}
            if instance_id:
                self._instance_configs[instance_id] = cfg
                if instance_id in self._nodes:
                    node = self._nodes[instance_id]
                    node.stop()
                    self._executor.remove_node(node)
                    del self._nodes[instance_id]
                return {"status": "configured", "instance_id": instance_id, "config": cfg}
            else:
                if "provider" in cfg:
                    self._provider = cfg["provider"]
                if "model" in cfg:
                    self._model = cfg["model"]
                if "key" in cfg:
                    self._key = cfg["key"]
                if "url" in cfg:
                    self._url = cfg["url"]
                # Rebuild global adapter
                self._adapter = _build_distance_adapter({
                    "provider": self._provider,
                    "url": self._url,
                    "key": self._key,
                    "model": self._model,
                    "model_path": self._model_path,
                })
                return {"status": "configured", "config": cfg}

        return None
