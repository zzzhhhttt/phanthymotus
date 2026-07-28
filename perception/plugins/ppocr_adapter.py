#!/usr/bin/env python3
"""
plugins/ppocr_adapter.py — PPOCRAdapter: 本地轻量 OCR 适配器

用于接入 plugins/ocr.py 中定义的 OCRAdapter 抽象接口，
基于 PaddleOCR 的轻量模型（PP-OCRv5_mobile / PP-OCRv6_tiny）在本地做
检测+识别两阶段推理，满足"模型 <15M 参数、GPU 占用 <10%"的榜单硬件约束。

设计要点：
1. 不依赖云端 API（不产生网络请求、不产生 key/quota 问题）。
2. 模型权重不进 git 仓库（>1MB 文件禁止上传），通过环境变量指定的
   本地路径加载（模型放 JuiceFS，镜像构建/启动时从内网地址下载到本地目录）。
3. 默认使用 onnxruntime 推理引擎，避免在 Jetson 上还要编译/安装完整的
   PaddlePaddle 推理框架，兼容性更好、镜像更轻。
4. 对外接口与 ocr.py 中其它 Adapter（OpenAIVisionAdapter / QwenVLAdapter /
   TesseractAdapter）保持一致：recognize(image_bytes, language) -> list[dict]，
   每个 dict 包含 "text" 和 "bbox": [x1, y1, x2, y2]（像素坐标，原图尺寸）。

依赖：
    pip install paddleocr onnxruntime-gpu opencv-python-headless numpy
    # Jetson 上如果 onnxruntime-gpu 官方 wheel 不适配 JetPack 版本，
    # 退化装 onnxruntime（CPU 也能跑，PP-OCRv5_mobile/v6_tiny 本身就很轻）。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .ocr import OCRAdapter

log = logging.getLogger(__name__)


class PPOCRAdapter(OCRAdapter):
    """PP-OCRv5_mobile / PP-OCRv6_tiny 本地推理适配器（det + rec 两阶段）"""

    # 语言 -> 若需要针对语言切换识别模型，可以在这里做映射；
    # PP-OCRv5_mobile_rec / PP-OCRv6_tiny_rec 本身已是多语言模型，
    # 通常不需要按语言切模型，这里保留扩展点。
    _LANG_HINT_MAP = {
        "zh": "ch",
        "zh-CN": "ch",
        "zh-TW": "chinese_cht",
        "en": "en",
        "ja": "japan",
        "ko": "korean",
    }

    def __init__(
        self,
        det_model_name: str = "PP-OCRv6_tiny_det",
        rec_model_name: str = "PP-OCRv6_tiny_rec",
        model_dir: Optional[str] = None,
        engine: str = "onnxruntime",
        device: str = "cpu",
        use_textline_orientation: bool = False,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        score_thresh: float = 0.5,
        max_side: int = 960,
    ):
        """
        Args:
            det_model_name: 检测模型名，如 PP-OCRv5_mobile_det / PP-OCRv6_tiny_det
            rec_model_name: 识别模型名，如 PP-OCRv5_mobile_rec / PP-OCRv6_tiny_rec
            model_dir: 本地模型权重目录（不要指向 git 仓库内！权重从 JuiceFS 下载到
                       容器内某个路径，例如 /opt/models/ppocr）。为 None 时使用
                       PaddleOCR 默认的模型缓存路径（~/.paddlex），需要保证该路径在
                       构建镜像时已经预热好，避免评测时现场联网下载导致超时/失败。
            engine: "onnxruntime"（推荐，Jetson 兼容性好） 或 "paddle"（需要装
                    paddlepaddle 推理框架，Jetson 上适配 JetPack 版本较麻烦）。
            device: "gpu" 或 "cpu"。Jetson Orin 用 "gpu" 走 CUDA/TensorRT。
            score_thresh: 识别置信度低于该阈值的结果会被过滤掉，避免噪声框拉低
                          precision（榜单评测里误检的框会直接计入 FP）。
            max_side: 推理前图片长边超过这个像素数就先缩小（省内存/加速），
                      可以在 config.yaml 里通过 max_side 字段调整，不用改代码。
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise RuntimeError(
                "PPOCRAdapter 需要安装 paddleocr：pip install paddleocr onnxruntime"
            ) from e

        self._score_thresh = score_thresh
        self._max_side = max_side

        # 权重不进 git，容器启动时从 JuiceFS 下载到 model_dir（已存在则跳过）。
        if model_dir:
            from utils.model_downloader import ensure_model
            ensure_model("ocr_det", os.path.join(model_dir, det_model_name))
            ensure_model("ocr_rec", os.path.join(model_dir, rec_model_name))

        kwargs = dict(
            text_detection_model_name=det_model_name,
            text_recognition_model_name=rec_model_name,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            device=device,
        )

        # 指定本地模型目录，避免评测环境现场联网拉模型
        if model_dir:
            kwargs["text_detection_model_dir"] = os.path.join(model_dir, det_model_name)
            kwargs["text_recognition_model_dir"] = os.path.join(model_dir, rec_model_name)

        # 部分 PaddleOCR 版本支持 engine 参数直接切 onnxruntime 后端；
        # 如果你安装的版本不支持该参数，把下面这行删掉，改为在各单独
        # 模块（TextDetection/TextRecognition）里传 engine="onnxruntime"。
        #
        # 同时传内存池优化（关闭 mem_pattern/cpu_mem_arena）和线程数限制
        # 这两组 onnxruntime 参数——注意：之前分别单独测过这两组参数，
        # 两次实测的崩溃点都比完全不传 engine_config 更早（分别是 case
        # 112、118，不带 engine_config 时是 case 147），当时判断是负优化
        # 已经撤回过一次。这次按你的判断重新加回来，用 try/except 包住，
        # 装的版本不认这个参数格式会自动退回不带 engine_config 的写法。
        ort_tuning_kwargs = dict(kwargs)
        ort_tuning_kwargs["engine_config"] = {
            "onnxruntime": {
                "enable_mem_pattern": False,
                "enable_cpu_mem_arena": False,
                "intra_op_num_threads": 1,
                "inter_op_num_threads": 1,
            }
        }

        try:
            self._ocr = PaddleOCR(engine=engine, **ort_tuning_kwargs)
            log.info("[ppocr] PaddleOCR 初始化成功（engine= + engine_config 内存优化+线程限制）")
        except Exception as e:
            log.warning(f"[ppocr] engine_config 内存优化+线程限制不被接受（{e}），退回仅 engine=")
            try:
                self._ocr = PaddleOCR(engine=engine, **kwargs)
            except TypeError:
                log.warning(
                    "[ppocr] installed paddleocr version does not accept `engine=`, "
                    "falling back to default backend"
                )
                self._ocr = PaddleOCR(**kwargs)

        log.info(
            f"[ppocr] adapter ready: det={det_model_name}, rec={rec_model_name}, "
            f"engine={engine}, device={device}"
        )

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        import io

        import cv2
        import numpy as np

        img, scale = self._decode_downscaled(image_bytes)
        if img is None:
            log.warning("[ppocr] failed to decode image bytes, skipping frame")
            return []

        results: list = []
        try:
            for res in self._ocr.predict(img):
                texts = res.get("rec_texts") or []
                scores = res.get("rec_scores") or []
                polys = res.get("rec_polys")
                if polys is None:
                    polys = res.get("dt_polys") or []

                for i, text in enumerate(texts):
                    if not text:
                        continue
                    score = float(scores[i]) if i < len(scores) else 1.0
                    if score < self._score_thresh:
                        continue
                    poly = polys[i] if i < len(polys) else None
                    bbox = self._poly_to_bbox(poly) if poly is not None else []
                    if bbox and scale != 1.0:
                        # 缩小图片上跑出来的 bbox，换算回原图坐标系，
                        # 不然定位框会整体偏移、拉低评测的定位准确率
                        inv = 1.0 / scale
                        bbox = [int(round(v * inv)) for v in bbox]
                    results.append({"text": text, "bbox": bbox, "score": score})
        except Exception as e:
            log.error(f"[ppocr] inference error: {e}", exc_info=True)
            raise

        return results

    def _decode_downscaled(self, image_bytes: bytes):
        """解码图片，长边超过 self._max_side 就缩小。

        跟"先完整解码原图、再resize缩小"不同，这里先用 PIL 只读文件头拿到
        原始宽高（几乎不占内存，不会真正解码整张图），据此选一个合适的
        cv2.IMREAD_REDUCED_COLOR_N 档位，让解码器在解码阶段就直接输出一张
        小很多的图（libjpeg 等格式的解码器原生支持按 1/2、1/4、1/8 比例
        解码，不需要先在内存里摊开一张几千万像素的完整图）。大图（比如评测
        数据集里出现过的 6MB+ 图片）能明显降低这一步的内存峰值。
        """
        import cv2
        import numpy as np
        from PIL import Image
        import io

        orig_w = orig_h = None
        try:
            with Image.open(io.BytesIO(image_bytes)) as probe:
                orig_w, orig_h = probe.size  # 只读头信息，不解码像素数据
        except Exception as e:
            log.debug(f"[ppocr] PIL header probe failed ({e}), falling back to full cv2 decode")

        img_array = np.frombuffer(image_bytes, dtype=np.uint8)

        if orig_w and orig_h:
            longer_side = max(orig_w, orig_h)
            if longer_side > self._max_side:
                # 选一个不会缩过头的档位：缩完之后长边仍 >= max_side，
                # 剩下的差距交给后面的精确 cv2.resize 补齐
                reduce_flag, reduce_factor = cv2.IMREAD_COLOR, 1
                for flag, factor in (
                    (cv2.IMREAD_REDUCED_COLOR_8, 8),
                    (cv2.IMREAD_REDUCED_COLOR_4, 4),
                    (cv2.IMREAD_REDUCED_COLOR_2, 2),
                ):
                    if longer_side / factor >= self._max_side:
                        reduce_flag, reduce_factor = flag, factor
                        break

                img = cv2.imdecode(img_array, reduce_flag)
                if img is not None:
                    h, w = img.shape[:2]
                    scale = w / orig_w  # 解码器实际给出的缩放比例（不一定精确等于 1/reduce_factor）
                    cur_longer = max(h, w)
                    if cur_longer > self._max_side:
                        final_scale = self._max_side / cur_longer
                        new_w = max(1, int(round(w * final_scale)))
                        new_h = max(1, int(round(h * final_scale)))
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        scale *= final_scale
                    log.debug(
                        f"[ppocr] reduced decode: {orig_w}x{orig_h} -> {img.shape[1]}x{img.shape[0]} "
                        f"(reduce_factor={reduce_factor}, final scale={scale:.3f})"
                    )
                    return img, scale
                # 缩放解码失败（比如该格式的解码器不支持这个 flag），
                # 落回完整解码 + resize 这条老路

        # 走到这里说明：图片本来就不算大，或者头信息读取/缩放解码失败,
        # 用最基础的方式完整解码，必要时再 resize 缩小
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return None, 1.0
        h, w = img.shape[:2]
        longer_side = max(h, w)
        if longer_side > self._max_side:
            scale = self._max_side / longer_side
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            log.debug(f"[ppocr] full-decode fallback + resize: {w}x{h} -> {new_w}x{new_h} (scale={scale:.3f})")
            return img, scale
        return img, 1.0

    @staticmethod
    def _poly_to_bbox(poly) -> list:
        """将检测多边形（4点或多点）转成外接矩形 [x1, y1, x2, y2]"""
        try:
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            return [int(round(min(xs))), int(round(min(ys))),
                    int(round(max(xs))), int(round(max(ys)))]
        except (TypeError, IndexError):
            return []