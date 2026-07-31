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
        enhance_contrast: bool = False,
        sharpen: bool = True,
        unclip_ratio: float = 2.0,
        two_stage_recognition: bool = False,
        max_recrop_boxes: int = 30,
        recrop_padding: int = 18,
        tiled_recognition: bool = False,
        tile_size: int = 960,
        tile_overlap_ratio: float = 0.25,
        tile_nms_iou_thresh: float = 0.5,
        tile_pre_downscale_max_side: int = 2400,
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
            unclip_ratio: 检测框往外扩张的比例，默认2.0偏保守，容易把文字边缘
                          切掉；调大（比如2.5）能让框更完整，尤其中文，代价是
                          可能把相邻内容也框进来。跟图片尺寸/内存无关，纯粹是
                          检测框后处理的一个参数。
            two_stage_recognition: 默认关闭。开启后，先在缩小后的图上做检测拿到
                          粗略文字框位置，再从"未缩小的原图"上把每个框对应区域
                          （四周留 recrop_padding 像素的余量）裁出来，对每一小块
                          单独重新跑一次完整的检测+识别（复用 self._ocr.predict()，
                          这条路径本身已经反复验证过能正常工作）——比整体缩小
                          再识别更清晰，尤其是小字/密集文字。代价：(1) 需要额外
                          解码一次原始分辨率的大图用于裁剪，部分抵消了之前
                          "先缩小再解码省内存"的优化；(2) 文字框很多的图片
                          （评测里见过194个框的case）会显著增加推理次数、拉长
                          耗时，用 max_recrop_boxes 限制上限。曾经用过一个未经
                          验证的独立"只识别"接口，实测分数不升反降（怀疑是那个
                          接口调用方式不对），已改为这个更保守、复用已验证代码
                          路径的写法。
            max_recrop_boxes: two_stage_recognition 开启时，最多重新识别多少个
                          框（按检测框面积从大到小排序，优先处理更大的文字区域）。
            recrop_padding: 裁剪每个文字框时，四周多留出的像素余量，避免刚好把
                          文字边缘切掉。
            tiled_recognition: 默认关闭，跟 two_stage_recognition 是两套独立、
                          互斥的策略（不要同时开）。开启后不做"先检测再挑区域
                          裁剪"，而是不管图片内容，把原图（未缩小）机械切成一块
                          块固定大小、互相有重叠的小方块（重叠是为了不让文字正好
                          卡在切割线上被截断），每一块单独跑一次完整检测+识别，
                          串行处理（一次只处理一块，不批量喂），推理时显存/内存
                          峰值锁定在"一块"的水平，不会因为原图很大而暴涨。切完
                          之后用 NMS 把重叠区域里被重复检测到的同一段文字去重、
                          合并。计算量比 two_stage_recognition 大得多（不管有没有
                          文字都要处理每一块），耗时会明显增加。
            tile_size: tiled_recognition 开启时，每块的边长（正方形）。
            tile_overlap_ratio: 相邻两块之间的重叠比例（0.25 = 25%），重叠是为了
                          避免文字刚好被切割线切断、两边都识别不完整。
            tile_nms_iou_thresh: 判断两个来自不同切块的框是不是"同一段文字被
                          重复检测"的重合度阈值，超过这个值就用 NMS 去重、只保留
                          置信度更高的那个。
            tile_pre_downscale_max_side: 切块之前，先把原图整体降到这个像素
                          上限（默认2400，比正常路径的960宽松很多），再在这张
                          "预缩小后的图"上切块——避免解码环节直接摊开一张原始
                          分辨率的巨图（这是切块模式之前 OOM 的主要来源），同时
                          依然比960清晰得多，保留切块方案本身想要的效果。
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise RuntimeError(
                "PPOCRAdapter 需要安装 paddleocr：pip install paddleocr onnxruntime"
            ) from e

        self._score_thresh = score_thresh
        self._max_side = max_side
        self._enhance_contrast = enhance_contrast
        self._sharpen = sharpen
        self._two_stage_recognition = two_stage_recognition
        self._max_recrop_boxes = max_recrop_boxes
        self._recrop_padding = recrop_padding
        self._tiled_recognition = tiled_recognition
        self._tile_size = tile_size
        self._tile_overlap_ratio = tile_overlap_ratio
        self._tile_nms_iou_thresh = tile_nms_iou_thresh
        self._tile_pre_downscale_max_side = tile_pre_downscale_max_side

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

        # unclip_ratio 单独放在一个可选的叠加层里，不直接混进上面这份
        # "已验证能跑通的基础配置"——万一装的版本不认这个参数，最终还是
        # 能退回到不带它的写法，不会连基础配置都跑不起来。
        unclip_overlay = {"text_det_unclip_ratio": unclip_ratio}

        # 部分 PaddleOCR 版本支持 engine 参数直接切 onnxruntime 后端；
        # 如果你安装的版本不支持该参数，把下面这行删掉，改为在各单独
        # 模块（TextDetection/TextRecognition）里传 engine="onnxruntime"。
        #
        # engine_config 里同时传内存池优化（关闭 mem_pattern/cpu_mem_arena）
        # 和线程数限制——注意：之前分别单独测过这两组参数，两次实测的
        # 崩溃点都比完全不传 engine_config 更早（分别是 case 112、118，
        # 不带 engine_config 时是 case 147），当时判断是负优化已经撤回
        # 过一次。这次按你的判断重新加回来。
        ort_overlay = {
            "engine_config": {
                "onnxruntime": {
                    "enable_mem_pattern": False,
                    "enable_cpu_mem_arena": False,
                    "intra_op_num_threads": 1,
                    "inter_op_num_threads": 1,
                }
            }
        }

        # 依次尝试：unclip+内存调优全带 → 只带unclip → 只带内存调优（不带engine=）
        # → 什么额外参数都不带的基础配置。前一种失败了就自动退到下一种，
        # 保证最坏情况下也能跑起来。
        attempts = [
            (dict(kwargs, **unclip_overlay, **ort_overlay), {"engine": engine}, "unclip + engine_config 内存优化+线程限制"),
            (dict(kwargs, **unclip_overlay), {"engine": engine}, "仅 unclip_ratio + engine="),
            (dict(kwargs, **unclip_overlay), {}, "仅 unclip_ratio（不带 engine=）"),
            (dict(kwargs, **ort_overlay), {"engine": engine}, "仅 engine_config 内存优化+线程限制（不带unclip）"),
            (kwargs, {"engine": engine}, "仅 engine="),
            (kwargs, {}, "默认后端（最基础配置）"),
        ]

        self._ocr = None
        last_err = None
        for extra_kwargs, extra_args, desc in attempts:
            try:
                self._ocr = PaddleOCR(**extra_args, **extra_kwargs)
                log.info(f"[ppocr] PaddleOCR 初始化成功（{desc}）")
                break
            except Exception as e:
                last_err = e
                log.debug(f"[ppocr] 初始化失败（{desc}）：{e}，尝试下一种方式")
                continue

        if self._ocr is None:
            raise RuntimeError(
                f"[ppocr] PaddleOCR 初始化失败，所有已知参数组合都不被当前版本接受: {last_err}"
            ) from last_err

        log.info(
            f"[ppocr] adapter ready: det={det_model_name}, rec={rec_model_name}, "
            f"engine={engine}, device={device}"
        )

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        import io

        import cv2
        import numpy as np

        if self._tiled_recognition:
            try:
                return self._tiled_predict(image_bytes)
            except Exception as e:
                log.error(f"[ppocr] tiled recognition failed ({e}), falling back to normal single-pass", exc_info=True)
                # 摔倒了不整个失败，退回正常的单遍识别（下面这条老路）

        img, scale = self._decode_downscaled(image_bytes)
        if img is None:
            log.warning("[ppocr] failed to decode image bytes, skipping frame")
            return []

        if self._enhance_contrast:
            img = self._apply_clahe(img)

        if self._sharpen:
            img = self._apply_sharpen(img)

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

        if self._two_stage_recognition and results:
            try:
                results = self._recrop_and_recognize(image_bytes, results)
            except Exception as e:
                log.warning(f"[ppocr] two-stage re-recognition failed ({e}), keeping first-pass results")

        return results

    @staticmethod
    def _apply_clahe(img):
        """局部对比度增强（CLAHE）——只增强原有对比度，不是"要么全用要么
        全不用"的二元转换（跟反色/二值化那种赌注式操作不一样），对光线
        不均、字迹偏淡的照片通常有帮助，即使图片本来就清晰，副作用一般
        也比较小。只在 YUV 的亮度通道上做，不碰颜色信息，避免引入色偏；
        跟输入图片同样大小，不会明显增加内存。"""
        import cv2

        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def _apply_sharpen(img):
        """轻量锐化——一次3x3卷积，突出文字笔画边缘，对小字体/略模糊的
        图片通常有帮助。跟CLAHE同一类操作：不改变图片尺寸，卷积核本身
        只有9个数，几乎不占额外内存。用的是温和的锐化核（中心权重5，
        不是更激进的写法），避免把正常清晰的图片过度锐化、引入噪点。"""
        import cv2
        import numpy as np

        kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    def _decode_downscaled(self, image_bytes: bytes, max_side_override: Optional[int] = None):
        """解码图片，长边超过 max_side（默认用 self._max_side，可以传
        max_side_override 覆盖，比如切块模式想用一个更宽松的上限）就缩小。

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

        max_side = max_side_override if max_side_override is not None else self._max_side

        orig_w = orig_h = None
        try:
            with Image.open(io.BytesIO(image_bytes)) as probe:
                orig_w, orig_h = probe.size  # 只读头信息，不解码像素数据
        except Exception as e:
            log.debug(f"[ppocr] PIL header probe failed ({e}), falling back to full cv2 decode")

        img_array = np.frombuffer(image_bytes, dtype=np.uint8)

        if orig_w and orig_h:
            longer_side = max(orig_w, orig_h)
            if longer_side > max_side:
                # 选一个不会缩过头的档位：缩完之后长边仍 >= max_side，
                # 剩下的差距交给后面的精确 cv2.resize 补齐
                reduce_flag, reduce_factor = cv2.IMREAD_COLOR, 1
                for flag, factor in (
                    (cv2.IMREAD_REDUCED_COLOR_8, 8),
                    (cv2.IMREAD_REDUCED_COLOR_4, 4),
                    (cv2.IMREAD_REDUCED_COLOR_2, 2),
                ):
                    if longer_side / factor >= max_side:
                        reduce_flag, reduce_factor = flag, factor
                        break

                img = cv2.imdecode(img_array, reduce_flag)
                if img is not None:
                    h, w = img.shape[:2]
                    scale = w / orig_w  # 解码器实际给出的缩放比例（不一定精确等于 1/reduce_factor）
                    cur_longer = max(h, w)
                    if cur_longer > max_side:
                        final_scale = max_side / cur_longer
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
        if longer_side > max_side:
            scale = max_side / longer_side
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

    def _recrop_and_recognize(self, image_bytes: bytes, results: list) -> list:
        """按第一遍（低分辨率）检测出来的粗略框，从原始未缩小的图片上把
        每个框对应区域裁出来（四周留 padding，避免边缘文字被切掉），
        再对每一小块单独跑一次完整的检测+识别（复用 self._ocr.predict()，
        这条路径已经反复验证过能正常工作，不再依赖没验证过的独立接口）。
        跟第一遍相比，这一步是在更接近原图的分辨率下重新定位+识别，
        对小字/密集文字区域通常更准。"""
        import cv2
        import numpy as np

        def box_area(r):
            b = r.get("bbox") or []
            if len(b) != 4:
                return 0
            return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

        # 按框的面积从大到小排，优先重新识别更大的文字区域，
        # 用 max_recrop_boxes 控制总数上限，避免文字框特别多的图片
        # 耗时暴涨
        ordered = sorted(range(len(results)), key=lambda i: box_area(results[i]), reverse=True)
        to_recrop = ordered[: self._max_recrop_boxes]
        if not to_recrop:
            return results

        # 这一步需要原始分辨率的图，之前为了省内存一直在避免完整解码
        # 大图——这里是 two_stage_recognition 功能本身需要付出的代价，
        # 默认关闭正是因为这个原因
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        full_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if full_img is None:
            raise RuntimeError("无法解码原始图片用于裁剪")
        full_h, full_w = full_img.shape[:2]

        replaced_indices = set()
        new_results: list = []

        for i in to_recrop:
            bbox = results[i].get("bbox") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            pad = self._recrop_padding
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(full_w, x2 + pad), min(full_h, y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = full_img[y1:y2, x1:x2]

            # 极少数情况下（原始框本身就很大），裁出来的小图可能还是
            # 不小——照搬跟主流程一样的降尺寸保护，避免这一步反而喂进去
            # 一张超大图，抵消掉本来想要的内存/速度收益
            crop_h, crop_w = crop.shape[:2]
            crop_scale = 1.0
            longer = max(crop_h, crop_w)
            if longer > self._max_side:
                crop_scale = self._max_side / longer
                crop = cv2.resize(
                    crop,
                    (max(1, int(round(crop_w * crop_scale))), max(1, int(round(crop_h * crop_scale)))),
                    interpolation=cv2.INTER_AREA,
                )

            try:
                crop_preds = list(self._ocr.predict(crop))
            except Exception as e:
                log.debug(f"[ppocr] recrop predict failed on box {i} ({e}), keeping original result")
                continue

            found_any = False
            for res in crop_preds:
                texts = res.get("rec_texts") or []
                scores = res.get("rec_scores") or []
                polys = res.get("rec_polys") or res.get("dt_polys") or []
                for j, text in enumerate(texts):
                    if not text:
                        continue
                    score = float(scores[j]) if j < len(scores) else 1.0
                    if score < self._score_thresh:
                        continue
                    poly = polys[j] if j < len(polys) else None
                    sub_bbox = self._poly_to_bbox(poly) if poly is not None else []
                    if sub_bbox:
                        if crop_scale != 1.0:
                            inv = 1.0 / crop_scale
                            sub_bbox = [v * inv for v in sub_bbox]
                        # 裁块内部坐标 -> 换算回完整原图坐标系
                        sub_bbox = [
                            int(round(sub_bbox[0] + x1)), int(round(sub_bbox[1] + y1)),
                            int(round(sub_bbox[2] + x1)), int(round(sub_bbox[3] + y1)),
                        ]
                    new_results.append({"text": text, "bbox": sub_bbox, "score": score})
                    found_any = True

            if found_any:
                replaced_indices.add(i)

        # 被成功重新识别过的框，用新结果替换掉原来（低分辨率下识别）的；
        # 没有被选中重裁、或者重裁后什么都没识别出来的，保留原结果
        kept = [r for idx, r in enumerate(results) if idx not in replaced_indices]
        return kept + new_results

    def _tiled_predict(self, image_bytes: bytes) -> list:
        """滑动窗口切块识别：把图片（不是完全不缩放，而是先降到一个比正常
        识别宽松很多的中等分辨率上限，见 tile_pre_downscale_max_side）机械
        切成一块块固定大小、互相有重叠的小方块，每一块单独、串行（一次一
        块）跑一次完整检测+识别，推理时的内存/显存峰值锁定在"一块"的
        水平。重叠区域可能让同一段文字在相邻两块里都被识别到，最后用
        NMS 去重。"""
        import cv2
        import numpy as np

        # 关键：不是完整解码原始分辨率的大图，而是复用跟正常识别路径一样
        # 经过验证的"边解码边缩小"逻辑，只是这里用一个宽松很多的上限
        # （tile_pre_downscale_max_side，默认比如 2400），而不是正常路径
        # 那个960——既避免了解码环节直接摊开一张几千万像素的原图（这是
        # 之前 OOM 的主要来源），又依然比960高清很多，保留切块方案本身
        # 想要的效果。pre_scale 记录了这一步缩小的比例，最后要用它把
        # bbox 坐标换算回真正的原图坐标系。
        full_img, pre_scale = self._decode_downscaled(
            image_bytes, max_side_override=self._tile_pre_downscale_max_side
        )
        if full_img is None:
            log.warning("[ppocr] failed to decode image bytes for tiled recognition")
            return []
        full_h, full_w = full_img.shape[:2]

        tile = self._tile_size
        stride = max(1, int(round(tile * (1 - self._tile_overlap_ratio))))

        # 生成切块的左上角坐标网格，横向、纵向都按 stride 步进；
        # 最后一块如果超出图片边界，往回收一点，保证块本身仍然是满尺寸
        # （不产生边缘上奇形怪状的小残块），代价是最后一块和倒数第二块
        # 的重叠区域会比设定的重叠比例更大一些，不影响正确性
        xs = list(range(0, max(1, full_w - tile) + 1, stride)) or [0]
        ys = list(range(0, max(1, full_h - tile) + 1, stride)) or [0]
        if xs[-1] + tile < full_w:
            xs.append(max(0, full_w - tile))
        if ys[-1] + tile < full_h:
            ys.append(max(0, full_h - tile))

        all_results: list = []
        for ty in ys:
            for tx in xs:
                x2 = min(full_w, tx + tile)
                y2 = min(full_h, ty + tile)
                patch = full_img[ty:y2, tx:x2]
                if patch.size == 0:
                    continue

                if self._enhance_contrast:
                    patch = self._apply_clahe(patch)
                if self._sharpen:
                    patch = self._apply_sharpen(patch)

                # 串行：一次只处理一块，不把多块拼成一个batch一起喂进去，
                # 这样任意时刻显存/内存占用都只对应一块的大小
                try:
                    patch_preds = list(self._ocr.predict(patch))
                except Exception as e:
                    log.debug(f"[ppocr] tile predict failed at ({tx},{ty}): {e}")
                    continue

                for res in patch_preds:
                    texts = res.get("rec_texts") or []
                    scores = res.get("rec_scores") or []
                    polys = res.get("rec_polys") or res.get("dt_polys") or []
                    for i, text in enumerate(texts):
                        if not text:
                            continue
                        score = float(scores[i]) if i < len(scores) else 1.0
                        if score < self._score_thresh:
                            continue
                        poly = polys[i] if i < len(polys) else None
                        bbox = self._poly_to_bbox(poly) if poly is not None else []
                        if bbox:
                            # 切块内部坐标 -> 换算回"预缩小后的完整图"坐标系
                            bbox = [bbox[0] + tx, bbox[1] + ty, bbox[2] + tx, bbox[3] + ty]
                            if pre_scale != 1.0:
                                # 再从"预缩小后的完整图"坐标系 -> 换算回真正
                                # 的原图坐标系，跟正常路径的坐标换算是同一个道理
                                inv = 1.0 / pre_scale
                                bbox = [int(round(v * inv)) for v in bbox]
                        all_results.append({"text": text, "bbox": bbox, "score": score})

        return self._nms_dedupe(all_results)

    def _nms_dedupe(self, results: list) -> list:
        """相邻切块的重叠区域，可能让同一段文字被重复检测出来——按置信度
        从高到低排序，两个框的重合度（IoU）超过阈值就认为是同一段文字，
        只保留分数更高的那个，丢弃其余的。"""
        if not results:
            return results

        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        ordered = sorted(
            [r for r in results if len(r.get("bbox") or []) == 4],
            key=lambda r: r["score"],
            reverse=True,
        )
        no_bbox = [r for r in results if len(r.get("bbox") or []) != 4]

        kept: list = []
        for r in ordered:
            if all(iou(r["bbox"], k["bbox"]) < self._tile_nms_iou_thresh for k in kept):
                kept.append(r)

        return kept + no_bbox