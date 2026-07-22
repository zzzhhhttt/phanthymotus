#!/usr/bin/env python3
"""
test_ocr.py — 端到端测试 OCR 插件：喂一张真实图片进去，看识别结果对不对。

用法（在容器内部执行，比如 docker exec 进去跑，因为需要 rclpy 环境）：
    python3 test_ocr.py /path/to/your/test_image.jpg

如果不传图片路径，会自动生成一张写着 "HelloWorld000" 的测试图（跟郭洪杰那次
测试同一个内容，方便对照结果）。

做的事：
1. 通过 MCP HTTP 接口，告诉 OCR 插件"开始监听 /ocr_test_image 这个 topic"
2. 把图片编码成 JPEG，发布到 /ocr_test_image
3. 订阅 /ocr_test_image/ocr，打印 OCR 插件识别出来的结果
"""
import sys
import json
import time
import threading
import urllib.request

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

MCP_PORT = 15720  # 跟你启动容器时传的 MCP_PORT 保持一致
INPUT_TOPIC = "/ocr_test_image"


def start_ocr_listener():
    """通过 MCP 接口告诉 OCR 插件订阅哪个 topic。"""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "ocr",
            "arguments": {"action": "start", "input_topic": INPUT_TOPIC},
        },
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{MCP_PORT}/mcp",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        print("[start] MCP response:", resp.read().decode())


def make_test_image_bytes(image_path: str | None) -> bytes:
    """读一张真实图片，或者没传路径的话生成一张写字的测试图。"""
    import cv2
    import numpy as np

    if image_path:
        with open(image_path, "rb") as f:
            raw = f.read()
        img_array = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法解码图片: {image_path}")
    else:
        img = np.full((200, 640, 3), 255, dtype=np.uint8)
        cv2.putText(img, "HelloWorld000", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


class OcrTestNode(Node):
    def __init__(self, image_bytes: bytes):
        super().__init__("ocr_test_client")
        self._pub = self.create_publisher(CompressedImage, INPUT_TOPIC, 10)
        self._sub = self.create_subscription(
            String, f"{INPUT_TOPIC}/ocr", self._on_result, 10
        )
        self._image_bytes = image_bytes
        self._got_result = threading.Event()
        # 稍微等一下再发布，确保订阅关系已经建立好
        self.create_timer(1.0, self._publish_once)
        self._published = False

    def _publish_once(self):
        if self._published:
            return
        self._published = True
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = self._image_bytes
        self._pub.publish(msg)
        print(f"[test] 已发布测试图片到 {INPUT_TOPIC}，等待识别结果...")

    def _on_result(self, msg: String):
        print("\n=== OCR 识别结果 ===")
        try:
            parsed = json.loads(msg.data)
            print(json.dumps(parsed, ensure_ascii=False, indent=2))
        except (json.JSONDecodeError, TypeError):
            print(msg.data)
        print("====================\n")
        self._got_result.set()

    def wait_for_result(self, timeout: float = 15.0) -> bool:
        return self._got_result.wait(timeout)


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("[1/3] 启动 OCR 监听...")
    start_ocr_listener()
    time.sleep(1)

    print("[2/3] 准备测试图片...")
    image_bytes = make_test_image_bytes(image_path)
    print(f"      图片大小: {len(image_bytes)} bytes")

    print("[3/3] 发布图片、等待识别结果...")
    rclpy.init()
    node = OcrTestNode(image_bytes)
    try:
        deadline = time.time() + 20
        while rclpy.ok() and time.time() < deadline and not node._got_result.is_set():
            rclpy.spin_once(node, timeout_sec=0.5)
        if not node._got_result.is_set():
            print("超时了，没收到识别结果 —— 检查一下 OCR 插件是不是正常启动了")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
