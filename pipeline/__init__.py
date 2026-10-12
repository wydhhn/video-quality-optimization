"""分段 → 超分 → 插帧 的串联流水线。

| 模块 | 职责 |
|---|---|
| `shotaware` | 与模型无关的纯计算：切点对齐、帧号追踪、帧数守恒（可单独测试） |
| `sr`        | BasicVSR++ 超分 ×4，切点处重置时序传播 |
| `fi`        | RIFE 插帧 ×2，跨切点跳过插帧 |
| `flow`      | 串联编排：解码 → 微批 → 推理 → 编码（三线程重叠 + 有界队列） |

依赖是**延迟导入**的：只用 `shotaware` 时完全不需要 torch / CUDA。
"""

from . import shotaware  # noqa: F401

__all__ = ["shotaware", "SrEngine", "FiEngine", "Flow", "Progress"]


def __getattr__(name):
    """延迟导入：避免 `import pipeline` 就拉起 torch。"""
    if name == "SrEngine":
        from .sr import SrEngine
        return SrEngine
    if name == "FiEngine":
        from .fi import FiEngine
        return FiEngine
    if name in ("Flow", "Progress"):
        from .flow import Flow, Progress
        return {"Flow": Flow, "Progress": Progress}[name]
    raise AttributeError(name)
