"""公共引导：路径、设备、外部依赖的初始化。

所有入口脚本第一行都 import 这个模块，它负责：
  1. 把 vendor/ 加入 sys.path（TransNetV2 网络定义 + OmniShotCut 官方包）
  2. 把 tools/bin 加入 PATH（ffmpeg-python 需要一个 PATH 上的 ffmpeg）
  3. 输出统一的日志格式
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
WEIGHTS = ROOT / "weights"
BIN = ROOT / "tools" / "bin"

if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


def _ffmpeg_works(exe) -> bool:
    """验证一个 ffmpeg 可执行文件**真的能跑起来**。

    只判断文件存在是不够的 —— 本机 PATH 上就有一个 0 字节的
    `...\\WinGet\\Links\\ffmpeg.exe`（坏掉的 WinGet shim），
    `shutil.which()` 找得到它，但 CreateProcess 会报
    `OSError: [WinError 193] %1 不是有效的 Win32 应用程序`。
    """
    import subprocess
    try:
        r = subprocess.run([str(exe), "-version"], capture_output=True,
                           timeout=20)
        return r.returncode == 0
    except Exception:                                        # noqa: BLE001
        return False


def ensure_ffmpeg():
    """返回一个可用的 ffmpeg 路径，并把它所在目录放进 PATH。

    顺序：**优先项目自带的**（版本可控、不依赖系统环境），
    自带的不可用才回退到系统 PATH 上的（且必须通过实际运行验证）。
    ffmpeg-python 通过 subprocess 调用 'ffmpeg'，所以必须放进 PATH。
    """
    import shutil

    bundled = BIN / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if bundled.exists() and _ffmpeg_works(bundled):
        os.environ["PATH"] = str(BIN) + os.pathsep + os.environ.get("PATH", "")
        return str(bundled)

    found = shutil.which("ffmpeg")
    if found and _ffmpeg_works(found):
        return found
    return None


def device():
    """有 CUDA 用 CUDA，否则回落到 CPU。"""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    print(msg, flush=True)


def fmt_time(t):
    """秒 → mm:ss.ss"""
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def to_segments(cut_frames, total_frames, fps):
    """切点帧号列表 → 统一的分段结构。

    cut_frames: 每个新镜头起始的帧号（不含 0）
    """
    edges = [0] + sorted({int(c) for c in cut_frames if 0 < c < total_frames}) + [total_frames]
    segs = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        segs.append({
            "id": i,
            "start": round(a / fps, 3),
            "end": round(b / fps, 3),
            "duration": round((b - a) / fps, 3),
            "start_frame": a,
            "end_frame": b,
        })
    return segs
