# predict_distance 只依赖 torch/transformers/PIL，不依赖 rclpy，本地
# 开发机（没装 ROS2）也能跑 predict.py / eval.py。ObstacleDistancePlugin
# 需要 rclpy（ROS2 节点），只在真正用到时才 import，避免在没有 ROS2 的
# 环境里 `import plugins.obstacle_distance` 直接报 ModuleNotFoundError。
from .predict import predict_distance

__all__ = ["ObstacleDistancePlugin", "predict_distance"]


def __getattr__(name):
    if name == "ObstacleDistancePlugin":
        from .plugin import ObstacleDistancePlugin
        return ObstacleDistancePlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
