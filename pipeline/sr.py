"""超分引擎 —— BasicVSR++ ×4，带分段感知。

与原始 `VsrEngine` 的区别：
1. **切点作为强制分段边界**（见 `shotaware.plan_chunks`），传播在切点重置，
   不会把镜头 A 的特征带到镜头 B。
2. 上下文的取法也受限：一段的 `read` 起点不会越过它所在镜头的起点。
3. `--no-shot-aware` 时退化为固定 chunk 切分，用于 A/B 对比。

其余沿用原项目的优化：`inference_mode` + 内存直传 + uint8 进出 + cudnn 预热。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from bootstrap import WEIGHTS, ROOT, log
from pipeline import shotaware

VSR_SRC = str(ROOT / "vendor" / "vsr")
if VSR_SRC not in sys.path:
    sys.path.insert(0, VSR_SRC)

VSR_WEIGHT = WEIGHTS / "basicvsr_plusplus_reds4.pth"
SPYNET_WEIGHT = WEIGHTS / "spynet.pth"

PRESET = dict(mid_channels=64, num_blocks=7, is_low_res_input=True)


class SrEngine:
    """BasicVSR++ 超分（scale=4）。"""

    scale = 4

    def __init__(self, fp16: bool = True, chunk: int = 8, overlap: int = 2,
                 device: str | None = None, verbose: bool = True):
        import torch
        self.torch = torch
        self.verbose = verbose
        self.chunk = int(chunk)
        self.overlap = int(overlap)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # fp16 只在 CUDA 上有意义（CPU 上 fp16 反而更慢）
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        self._warmed: tuple | None = None

        torch.backends.cudnn.benchmark = True

        import vsr as _vsr
        net = _vsr.BasicVSRPlusPlusNet(**PRESET)
        from vsr.checkpoint import load_checkpoint
        load_checkpoint(net, str(VSR_WEIGHT), strict=True)
        net = net.eval().to(self.device)
        if self.fp16:
            net = net.half()
        self.net = net
        self.n_params = sum(p.numel() for p in net.parameters())
        self._log(f"BasicVSR++ 就绪 params={self.n_params/1e6:.2f}M "
                  f"device={self.device} fp16={self.fp16} chunk={self.chunk}")

    def _log(self, *a):
        if self.verbose:
            log("  [sr] " + " ".join(str(x) for x in a))

    # ------------------------------------------------------------------ #
    def _prep(self, frames: np.ndarray):
        """(T,H,W,3) uint8 → (1,T,3,H,W) float。"""
        torch = self.torch
        t = torch.from_numpy(np.ascontiguousarray(frames))
        t = t.to(self.device, non_blocking=True).permute(0, 3, 1, 2)
        t = t.unsqueeze(0).float().div_(255.0)
        return t.half() if self.fp16 else t

    def _forward(self, t):
        with self.torch.inference_mode():
            out = self.net(t)
        return out[0] if isinstance(out, (tuple, list)) else out

    def warmup(self, h: int, w: int):
        """按实际尺寸把 cudnn 算法选型跑一遍。

        不预热的话这段一次性开销会全算进第一个任务（原项目实测能到 82 秒）。
        """
        if self._warmed == (h, w):
            return
        t0 = time.time()
        for n in sorted({self.chunk, self.chunk + self.overlap, 2}):
            if n >= 2:
                self._forward(self._prep(np.zeros((n, h, w, 3), np.uint8)))
        self._warmed = (h, w)
        self._log(f"已预热 {w}x{h}（{time.time()-t0:.1f}s，含 cudnn 算法选型）")

    # ------------------------------------------------------------------ #
    def upscale(self, frames: np.ndarray, cut_frames=None,
                shot_aware: bool = True, on_progress=None) -> np.ndarray:
        """(T,H,W,3) uint8 → (T,4H,4W,3) uint8。

        `cut_frames` 是切点帧号列表（全局）。`shot_aware=False` 时忽略它，
        退化成固定 chunk 切分（用于 A/B 对比）。
        """
        with self.torch.inference_mode():
            return self._upscale_inner(frames, cut_frames, shot_aware, on_progress)

    def _upscale_inner(self, frames, cut_frames, shot_aware, on_progress):
        T = len(frames)
        if T == 0:
            return np.zeros((0, 0, 0, 3), np.uint8)
        h, w = frames.shape[1], frames.shape[2]
        cuts = shotaware.normalize_cuts(cut_frames or [], T)

        if shot_aware:
            plan = shotaware.plan_chunks(T, cuts, self.chunk, self.overlap)
        else:
            plan = shotaware.plan_chunks_naive(T, self.chunk, self.overlap)

        out = np.empty((T, h * self.scale, w * self.scale, 3), np.uint8)
        for k, c in enumerate(plan):
            lo, hi = c["read"]
            vs, ve = c["valid"]
            blk = frames[lo:hi]
            ctx = vs - lo                      # 该段前面有多少帧是纯上下文
            y = self._forward(self._prep(blk))   # (1, T, 3, 4H, 4W)
            # ⚠️ 模型输出是 5 维 (n,t,c,h,w)，必须先去掉 batch 维再按时间切片，
            #    否则会切错维度（写成 y[ctx:...] 是在切 batch 维，直接报
            #    "permute: input.dim() = 5 is not equal to len(dims) = 4"）
            y = y[0, ctx:ctx + (ve - vs)].clamp_(0, 1).mul_(255.0).round_()
            out[vs:ve] = y.permute(0, 2, 3, 1).to(self.torch.uint8).cpu().numpy()
            del y, blk
            if on_progress:
                on_progress(k + 1, len(plan))
        return out
