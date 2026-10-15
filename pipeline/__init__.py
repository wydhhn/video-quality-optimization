"""分段 → 超分 → 插帧 的串联流水线。

| 模块 | 职责 |
|---|---|
| `shotaware` | 与模型无关的纯计算：切点对齐、帧号追踪、帧数守恒（可单独测试） |
| `sr`        | BasicVSR++ 超分 ×4（PyTorch），切点处重置时序传播 |
| `sr_trt`    | **同一模型的 TensorRT 后端**，接口与 `sr` 完全一致，可直接互换 |
| `fi`        | RIFE 插帧 ×2，跨切点跳过插帧 |
| `flow`      | 串联编排：解码 → 微批 → 推理 → 编码（三线程重叠 + 有界队列） |

`SrEngine` 与 `SrTrtEngine` 接口一致，命令行用 `run.py --sr-backend {torch,trt}` 切换：

    from pipeline import SrEngine              # PyTorch（默认，无需预先准备）
    from pipeline import SrTrtEngine           # TensorRT（需先备好引擎）

    sr = SrTrtEngine(chunk=8)                  # 自动找 t=8 的引擎
    Flow(sr, fi).run(...)

TensorRT 后端实测（RTX 5070 Ti，480×836 → ×4，对 PyTorch fp32 真值）：
TF32 版 78.54 dB / 最大差 1/255 / 1.69×，详见 `docs/TRT引擎构建全流程.md`。

依赖是**延迟导入**的：只用 `shotaware` 时完全不需要 torch / CUDA。
"""

from . import shotaware  # noqa: F401

__all__ = ["shotaware", "SrEngine", "SrTrtEngine", "FiEngine", "Flow", "Progress"]


def __getattr__(name):
    """延迟导入：避免 `import pipeline` 就拉起 torch。"""
    if name == "SrEngine":
        from .sr import SrEngine
        return SrEngine
    if name == "SrTrtEngine":
        from .sr_trt import SrTrtEngine
        return SrTrtEngine
    if name == "FiEngine":
        from .fi import FiEngine
        return FiEngine
    if name in ("Flow", "Progress"):
        from .flow import Flow, Progress
        return {"Flow": Flow, "Progress": Progress}[name]
    raise AttributeError(name)
