"""路径与运行参数的统一配置。

★ 这个文件被改造成了**双布局自适应**：同一份代码在两种目录结构下都能跑。

| 布局 | 典型场景 | 模型源码 | 权重 |
|---|---|---|---|
| **A. 本仓库**（video-quality-optimization） | 别人 clone 下来直接用 | `vendor/` | `weights/` |
| **B. 原项目**（FrameInterpolation） | 开发时的原始工作区 | `超分模型/` `插帧模型/` | 各自目录下 |

判定方式是**看目录存不存在**，优先用本仓库布局。
想强制指定根目录，设环境变量 `FI_BASE` 即可。
"""
from __future__ import annotations

import os

# 项目根目录：本文件在 webapp/ 下，故往上一级就是仓库根。
# 这样整个目录搬到哪台机器都能跑（原来这里是写死的绝对路径，打包发给别人就失效）。
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("FI_BASE") or os.path.dirname(_HERE)

# --------------------------------------------------------------------------
# 布局 A：本仓库（vendor/ + weights/）—— 优先
# --------------------------------------------------------------------------
_VENDOR = os.path.join(BASE, "vendor")
_WEIGHTS = os.path.join(BASE, "weights")

# 布局 B：原项目（超分模型/ + 插帧模型/）—— 回退
_VSR_DIR = os.path.join(BASE, "超分模型", "VSR")
_RIFE_DIR = os.path.join(BASE, "插帧模型", "RIFE")


def _pick(primary: str, fallback: str) -> str:
    """存在就用 primary，否则用 fallback。"""
    return primary if os.path.isdir(primary) else fallback


# 模型源码目录（会被加进 sys.path，所以要是**包的父目录**）
#   vendor/         → 里面是 vsr/ 包
#   vendor/rife     → 里面是 rife_ms/ 包
VSR_SRC = _pick(_VENDOR, os.path.join(_VSR_DIR, "src"))
RIFE_SRC = _pick(os.path.join(_VENDOR, "rife"), os.path.join(_RIFE_DIR, "src"))

# 权重
VSR_WEIGHT = (
    os.path.join(_WEIGHTS, "basicvsr_plusplus_reds4.pth")
    if os.path.isfile(os.path.join(_WEIGHTS, "basicvsr_plusplus_reds4.pth"))
    else os.path.join(_VSR_DIR, "weights", "basicvsr_plusplus_reds4.pth")
)
RIFE_WEIGHT = (
    os.path.join(_WEIGHTS, "flownet.pkl")
    if os.path.isfile(os.path.join(_WEIGHTS, "flownet.pkl"))
    else os.path.join(_RIFE_DIR, "weights", "flownet.pkl")
)

# Real-ESRGAN（可选后端）：本仓库没有它的源码/权重，只在布局 B 下可用
RESR_DIR = os.path.join(BASE, "超分模型", "RealESRGAN")
RESR_WEIGHT = os.path.join(RESR_DIR, "weights", "RealESRGAN_x4plus.pth")

# TensorRT 引擎目录。默认找 webapp/cache，也兼容引擎直接放在仓库根的 trt5/
CACHE = os.path.join(_HERE, "cache")
TRT_DIR = _pick(os.path.join(BASE, "trt5"), CACHE)

FFMPEG = None            # 延迟从 imageio_ffmpeg 取，避免 import 时的副作用


def ffmpeg() -> str:
    global FFMPEG
    if FFMPEG is None:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMPEG


def ensure_dirs() -> None:
    os.makedirs(CACHE, exist_ok=True)


def describe() -> str:
    """打印一次路径判定结果，便于排查"为什么找不到模型"。"""
    return (
        f"BASE       = {BASE}\n"
        f"  布局     = {'A. 本仓库(vendor/weights)' if os.path.isdir(_VENDOR) else 'B. 原项目(超分模型/插帧模型)'}\n"
        f"  VSR_SRC  = {VSR_SRC}\n"
        f"  RIFE_SRC = {RIFE_SRC}\n"
        f"  VSR_WEIGHT  = {VSR_WEIGHT}  {'✅' if os.path.isfile(VSR_WEIGHT) else '❌ 缺失'}\n"
        f"  RIFE_WEIGHT = {RIFE_WEIGHT}  {'✅' if os.path.isfile(RIFE_WEIGHT) else '❌ 缺失'}\n"
        f"  TRT_DIR  = {TRT_DIR}"
    )


if __name__ == "__main__":
    print(describe())
