#!/usr/bin/env python3
"""
test_obstacle.py — Local sanity check for LocalDistanceAdapter (plugins/obstacle.py),
no ROS2/rclpy required.

Usage:
  cd perception
  python3 test_obstacle.py /path/to/image.png [image2.jpg ...]

Prints, for each image: detected scene domain, pred_distance (meters), and
elapsed wall-clock time — use this against a few known-distance sample images
before submitting to the leaderboard, to sanity-check the pipeline isn't
obviously broken (empty ROI, wrong domain, model load failure, etc.).

Only imports the pieces of plugins/obstacle.py that don't need rclpy
(DistanceAdapter/LocalDistanceAdapter and the standalone domain/ROI/percentile
helper functions), so this runs on a plain dev machine without ROS2 installed
— rclpy is only imported at module scope in plugins/obstacle.py for the ROS2
Node classes, which this script never touches.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)


def _load_local_adapter_class():
    """Import LocalDistanceAdapter without triggering rclpy import failures.

    plugins/obstacle.py imports rclpy at module scope (needed for the ROS2
    Node classes further down in the file). On a dev machine without ROS2
    installed this import would fail before we ever reach LocalDistanceAdapter,
    even though LocalDistanceAdapter itself has no ROS2 dependency. If rclpy
    is missing, stub it out just enough to satisfy the module-level imports.
    """
    try:
        import rclpy  # noqa: F401
        from plugins.obstacle import LocalDistanceAdapter, _detect_scene_domain
        return LocalDistanceAdapter, _detect_scene_domain
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

        from plugins.obstacle import LocalDistanceAdapter, _detect_scene_domain
        return LocalDistanceAdapter, _detect_scene_domain


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="image file paths (png/jpg)")
    parser.add_argument("--model-path", default=None, help="override model dir (see LocalDistanceAdapter docstring)")
    args = parser.parse_args()

    LocalDistanceAdapter, detect_scene_domain = _load_local_adapter_class()

    print(f"# loading models from {args.model_path or '(default perception/models/depth_anything_v2)'}...",
          file=sys.stderr)
    adapter = LocalDistanceAdapter(args.model_path)

    exit_code = 0
    for image_path in args.images:
        path = Path(image_path)
        if not path.exists():
            print(json.dumps({"image": image_path, "error": "file not found"}, ensure_ascii=False))
            exit_code = 1
            continue

        image_bytes = path.read_bytes()
        domain = detect_scene_domain(image_bytes, filename=path.name)

        t0 = time.monotonic()
        result = adapter.estimate(image_bytes)
        elapsed_ms = (time.monotonic() - t0) * 1000

        print(json.dumps({
            "image": str(path),
            "domain": domain,
            "pred_distance": result.get("pred_distance"),
            "elapsed_ms": round(elapsed_ms, 1),
        }, ensure_ascii=False))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
