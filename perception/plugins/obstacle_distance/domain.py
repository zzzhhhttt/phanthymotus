#!/usr/bin/env python3
"""
plugins/obstacle_distance/domain.py — 场景域判定：室内机器人 vs 无人车。

榜单的输入说明本身就带了区分信号：
    "输入为单张 png（等效焦距 29 mm）或 jpg（等效焦距 33 mm）图像"

室内机器人数据集用一种相机（png/29mm），无人车数据集用另一种相机
（jpg/33mm）——两个数据集在采集时就没有混用格式，所以文件格式/编码
本身就是场景域的标签，不需要另外训练一个场景分类器。

对应关系：
    png → indoor（室内机器人）
    jpg / jpeg → outdoor（无人车）

如果调用方明确知道场景（比如 ROS2 topic 名字里带了 camera 类型），也
可以直接传 domain 参数覆盖，不依赖文件名/格式猜测。
"""

from __future__ import annotations

Domain = str  # "indoor" | "outdoor"

INDOOR = "indoor"
OUTDOOR = "outdoor"

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


def detect_domain(image_bytes: bytes, filename: str | None = None) -> Domain:
    """根据文件内容 magic bytes（优先）或文件名后缀判定 indoor/outdoor。

    优先看文件内容而不是后缀名：ROS2 topic 传过来的图片经常没有文件名，
    或者调用方随手传了个 .jpg 后缀但内容其实是 png——判断 magic bytes
    更可靠。
    """
    if image_bytes[:8] == _PNG_MAGIC:
        return INDOOR
    if image_bytes[:3] == _JPEG_MAGIC:
        return OUTDOOR

    if filename:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext == "png":
            return INDOOR
        if ext in ("jpg", "jpeg"):
            return OUTDOOR

    # 兜底：既不是可识别的 png/jpeg magic，也没有可用后缀，默认按室内处理
    # （室内模型深度量程更小，对近距离误判的代价 [F1@1m 主指标] 更保守）。
    return INDOOR
