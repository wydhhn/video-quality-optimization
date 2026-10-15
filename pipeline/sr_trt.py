"""超分引擎（TensorRT 版）—— BasicVSR++ ×4，走编译好的 TensorRT 引擎。

和 `pipeline.sr.SrEngine` 的关系
--------------------------------
**接口完全一致**（`__init__` / `warmup` / `upscale`），所以可以整体替换：

    from pipeline.sr import SrEngine            # PyTorch 版
    from pipeline.sr_trt import SrTrtEngine     # TensorRT 版
    sr = SrTrtEngine(chunk=8)                   # 默认自动找引擎

但**内部机制有两个本质区别**，用之前必须知道：

1. **引擎是固定形状的**。`t`（每次前向吃几帧）在编译时就写死进图里了，
   不能像 PyTorch 那样随便喂任意长度。序列长了要切窗口，短了要补帧。

2. **没有 `overlap` 这个概念**（默认 0）。因为引擎的 `t` 已经固定，
   想给段间留上下文就得把 `t` 编译得比 `chunk` 大
   （例：`chunk=8, overlap=2` ↔ 引擎 `t=10`）。

实测精度/速度（RTX 5070 Ti，480×836 → ×4，对 PyTorch fp32 真值）：

| 实现 | ms/帧 | 相对 PyTorch fp32 | PSNR | 最大差 |
|---|---|---|---|---|
| PyTorch fp32 | 441.9 | 1.00× | — | 0 |
| **TensorRT TF32** | **260.9** | **1.69×** | **78.54 dB** | **1.0** |
| TensorRT fp16 | 93.3 | 3.97× | 58.93 dB | 21.0 |

推荐 **TF32**：精度最高、且比 PyTorch fp32 快 1.69×。
（fp16 快 4 倍但最大差 21/255，不满足 ≤2/255 的要求。）

引擎怎么来
----------
见 `tools/trt/README.md`。一句话：
**构建时必须用真实视频帧做 dummy，绝不能用 `torch.zeros`** ——
否则会触发 `is_mirror_extended` 分支，导出一个缺了正向光流的残缺图，
精度只剩 31 dB，而且这个问题极难从现象反推。详见 `docs/TRT引擎构建全流程.md`。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from bootstrap import ROOT, log

# 引擎搜索路径：本仓库 trt5/ → webapp/cache/ → 环境变量指定
_TRT_DIRS = [
    os.environ.get("FI_TRT_DIR"),
    str(ROOT / "trt5"),
    str(ROOT / "webapp" / "cache"),
    str(ROOT / "weights"),
]
_TRT_DIRS = [d for d in _TRT_DIRS if d]


def find_engine(t: int, h: int, w: int, mode: str = "tf32") -> str | None:
    """按命名约定找引擎文件。返回绝对路径或 None。

    命名约定（与 tools/trt/export_vsr_onnx.py 的输出一致）：
        vsr_{mode}_t{t}_{h}x{w}_x4.trt
    """
    names = [
        f"vsr_{mode}_t{t}_{h}x{w}_x4.trt",
        f"vsr_decomp_t{t}_{h}x{w}_x4_{mode}.trt",
    ]
    for d in _TRT_DIRS:
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    return None


class SrTrtEngine:
    """BasicVSR++ 的 TensorRT 后端。接口与 `SrEngine` 一致。"""

    def __init__(self, chunk: int = 8, overlap: int = 0, t: int | None = None,
                 mode: str = "tf32", engine: str | None = None,
                 device: str | None = None, verbose: bool = True):
        """
        Args:
            chunk:   每段输出多少帧。**应当等于引擎的 t**（见下方说明）。
            overlap: 段间上下文帧数。**引擎没按 t=chunk+overlap 编译时必须是 0**，
                     否则喂进去的长度对不上，会直接报错。
            t:       引擎的帧数。默认 = chunk + overlap。
            mode:    tf32（推荐）/ fp32 / fp16
            engine:  直接指定引擎路径，跳过自动搜索。
        """
        self.verbose = verbose
        self.t = int(t if t is not None else chunk + overlap)
        self.chunk = int(chunk)
        self.overlap = int(overlap)
        self.mode = mode
        self._engine_path = engine
        self._trt = None
        self._built = False

        if self.chunk + self.overlap != self.t:
            # 刻意允许，但要提醒：这时流水线喂的长度和引擎不符
            self._log(f"⚠️ chunk({self.chunk}) + overlap({self.overlap}) "
                      f"≠ t({self.t})。除非你清楚在做什么，否则建议 chunk=t、overlap=0。")

        import torch
        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ------------------------------------------------------------------ #
    def _log(self, *a):
        if self.verbose:
            log(" ".join(str(x) for x in a))

    def _ensure(self, h: int, w: int):
        if self._built:
            return
        if self.device.type != "cuda":
            raise RuntimeError("TensorRT 引擎需要 CUDA 设备")
        path = self._engine_path or find_engine(self.t, h, w, self.mode)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"找不到 t={self.t} {h}x{w} 的 {self.mode} 引擎。\n"
                f"  搜索过: {_TRT_DIRS}\n"
                f"  构建方法见 tools/trt/README.md，或下载：\n"
                f"      python tools/download_trt_engine.py")

        from tools.trt.trt_runner import TrtRunner
        self._trt = TrtRunner(path)
        self._built = True
        self._log(f"[sr-trt] 引擎就绪 {os.path.basename(path)}  t={self.t}  "
                  f"输入 {w}x{h}  dtype={self._trt.in_dtype}")

    def warmup(self, h: int, w: int):
        """按实际尺寸跑一次，把 cudnn/TRT 的首次开销付掉。"""
        self._ensure(h, w)
        dummy = np.zeros((self.t, h, w, 3), np.uint8)
        with self.torch.inference_mode():
            self._forward(dummy)
        self._log(f"[sr-trt] 预热完成 {w}x{h}")

    # ------------------------------------------------------------------ #
    def _forward(self, frames: np.ndarray) -> np.ndarray:
        """(t,H,W,3) uint8 → (t,4H,4W,3) uint8。长度必须恰好是 t。"""
        t = self.torch.from_numpy(np.ascontiguousarray(frames))
        t = t.permute(0, 3, 1, 2)[None]              # (1,t,3,H,W)
        t = t.float().div_(255.0).to(self._trt.in_dtype)
        y = self._trt(t)                             # (1,t,3,4H,4W)
        y = y[0].permute(0, 2, 3, 1).float().clamp(0, 1).mul(255.0).round()
        return y.to(np.uint8).cpu().numpy()

    def upscale(self, frames: np.ndarray, cut_frames=None,
                shot_aware: bool = True, on_progress=None) -> np.ndarray:
        """(T,H,W,3) uint8 → (T,4H,4W,3) uint8。

        签名与 `SrEngine.upscale` 一致（`cut_frames` / `shot_aware` 接受但
        **暂未使用** —— 引擎的窗口是固定 t 帧，要接镜头感知得把切点先灌进
        分段计划再喂，见 `pipeline.shotaware.plan_chunks`）。
        """
        T = len(frames)
        if T == 0:
            return np.zeros((0, 0, 0, 3), np.uint8)
        h, w = frames.shape[1], frames.shape[2]
        self._ensure(h, w)

        out = np.empty((T, h * 4, w * 4, 3), np.uint8)
        step = self.chunk
        pos = 0
        with self.torch.inference_mode():
            while pos < T:
                end = min(pos + step, T)
                ctx = 0 if pos == 0 else min(self.overlap, pos)

                # 喂给引擎的长度必须恰好是 t
                src = frames[pos - ctx:end]
                n = len(src)
                if n < self.t:
                    # 尾部重复最后一帧补齐。
                    # ⚠️ **这不是无损的**：BasicVSR++ 有 backward_1/backward_2
                    # 两条反向传播路径，尾部多出的帧会沿反向路径影响**前面每一帧**
                    # （实测 8帧 vs 8帧+补2帧：最大差 27/255、PSNR 58.22 dB）。
                    # 所以最好让 chunk 正好等于 t，让这条分支根本不触发。
                    src = np.concatenate(
                        [src, np.repeat(src[-1:], self.t - n, axis=0)], 0)
                y = self._forward(src)
                out[pos:end] = y[ctx:ctx + (end - pos)]
                if on_progress:
                    on_progress(end, T)
                pos = end
        return out
