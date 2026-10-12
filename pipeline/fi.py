"""插帧引擎 —— RIFE ×2，带分段感知。

与原始 `RifeEngine` 的区别：
**跨切点的帧对不调用 RIFE**，直接用前一帧的副本代替（见 `shotaware`）。
这样输出帧数严格保持 2N，同时避免"把两个不同场景融合成一帧"的鬼影。

沿用原项目的优化：
  · `PAD = 32` —— 输入尺寸补齐到 32 的倍数（shape 恒定，CUDA Graph / ONNX 才可用）
  · `FORCE_FP32_WARP` —— 5K 分辨率下必须把 warp 的网格采样放回 fp32，
    否则 `grid_sample` 的归一化坐标在 fp16 里区分不开相邻像素（PSNR 会崩）
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from bootstrap import WEIGHTS, ROOT, log
from pipeline import shotaware

RIFE_SRC = str(ROOT / "vendor" / "rife")
if RIFE_SRC not in sys.path:
    sys.path.insert(0, RIFE_SRC)

RIFE_WEIGHT = WEIGHTS / "flownet.pkl"

PAD = 32


class FiEngine:
    """RIFE 插帧（2×）。"""

    PAD = PAD

    def __init__(self, fp16: bool = True, batch: int = 2,
                 device: str | None = None, warp_fp32: bool = True,
                 verbose: bool = True):
        import torch
        self.torch = torch
        self.verbose = verbose
        self.batch = max(1, int(batch))
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        self.dtype = torch.float16 if self.fp16 else torch.float32

        import rife_ms.warplayer as warplayer
        from rife_ms.IFNet_HDv3 import IFNet
        # 5K 下必须把网格采样放回 fp32（原因见 warplayer.py 顶部注释）
        warplayer.FORCE_FP32_WARP = bool(self.fp16 and warp_fp32)

        net = IFNet()
        sd = torch.load(str(RIFE_WEIGHT), map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items() if "module." in k}
        net.load_state_dict(sd)
        net = net.eval().to(self.device)
        if self.fp16:
            net = net.half()
        self.net = net
        self.n_params = sum(p.numel() for p in net.parameters())
        self._log(f"RIFE 就绪 params={self.n_params/1e6:.2f}M "
                  f"device={self.device} fp16={self.fp16} "
                  f"fp32_warp={warplayer.FORCE_FP32_WARP} batch={self.batch}")

    def _log(self, *a):
        if self.verbose:
            log("  [fi] " + " ".join(str(x) for x in a))

    # ------------------------------------------------------------------ #
    def _pad_size(self, h: int, w: int):
        ph = (h + PAD - 1) // PAD * PAD
        pw = (w + PAD - 1) // PAD * PAD
        return ph, pw

    def _run_batch(self, a_np: np.ndarray, b_np: np.ndarray,
                   h: int, w: int) -> np.ndarray:
        """a_np/b_np: (N,H,W,3) uint8 → 中间帧 (N,H,W,3) uint8。"""
        torch = self.torch
        ph, pw = self._pad_size(h, w)

        def to_tensor(arr):
            t = torch.from_numpy(np.ascontiguousarray(arr))
            t = t.to(self.device, non_blocking=True).permute(0, 3, 1, 2)
            t = t.float().div_(255.0)
            if ph != h or pw != w:
                t = torch.nn.functional.pad(t, (0, pw - w, 0, ph - h),
                                            mode="replicate")
            return t.to(self.dtype)

        ta, tb = to_tensor(a_np), to_tensor(b_np)
        n = ta.shape[0]
        if n != self.batch:                      # 补齐到固定 batch
            k = self.batch - n
            ta = torch.cat([ta, ta[-1:].expand(k, -1, -1, -1)], 0)
            tb = torch.cat([tb, tb[-1:].expand(k, -1, -1, -1)], 0)
        with torch.inference_mode():
            _, _, merged = self.net(torch.cat([ta, tb], 1),
                                    scale_list=[4.0, 2.0, 1.0])
        y = merged[-1]
        y = y[:n, :, :h, :w].float().clamp(0, 1).mul(255.0).round()
        return y.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()

    def _run_pairs(self, frames: np.ndarray, pairs: list[int],
                   on_progress=None) -> dict[int, np.ndarray]:
        """只对指定的帧对跑 RIFE。返回 {pair_index: 中间帧}。"""
        if not pairs:
            return {}
        h, w = frames.shape[1], frames.shape[2]
        out: dict[int, np.ndarray] = {}
        bs = self.batch
        for k in range(0, len(pairs), bs):
            chunk = pairs[k:k + bs]
            a = np.stack([frames[i] for i in chunk])
            b = np.stack([frames[i + 1] for i in chunk])
            mids = self._run_batch(a, b, h, w)
            for j, i in enumerate(chunk):
                out[i] = mids[j]
            if on_progress:
                on_progress(min(k + bs, len(pairs)), len(pairs))
        return out

    # ------------------------------------------------------------------ #
    def interpolate(self, frames: np.ndarray, cut_frames=None,
                    shot_aware: bool = True, g0: int = 0,
                    total: int | None = None, lead_pairs: int = 0,
                    on_progress=None) -> np.ndarray:
        """对一批帧做 2× 插帧。

        `frames` 是这一批源帧；如果 `lead_pairs=1`，说明 `frames[0]` 是上一批
        留下的**上下文帧**（carry）—— 它本身**不再输出**，但要用它生成
        「它 → frames[1]」这一对的中间帧，否则批边界处会缺一个插帧。

        输出帧数：
            lead_pairs=0 → 2n - 1
            lead_pairs=1 → 2n
        （n = len(frames)）。整段视频的总输出 = 2N，其中最后补一帧由调用方负责，
        见 `pipeline.flow`。

        `g0` 是 `frames[0]` 的**全局帧号** —— 微批时不是 0，跨批的帧对
        必须靠它才能正确判断是否跨切点。
        """
        T = len(frames)
        if T == 0:
            return np.zeros((0, 0, 0, 3), np.uint8)
        total = total or T
        cuts = shotaware.normalize_cuts(cut_frames or [], total) if shot_aware else []

        mask = shotaware.crossing_mask(g0, T, cuts, total)   # 长度 T-1
        need = [i for i in range(T - 1) if not mask[i]]
        skipped = [i for i in range(T - 1) if mask[i]]
        mids = self._run_pairs(frames, need, on_progress=on_progress)

        out = []
        # 前 lead_pairs 帧只当上下文，不输出；但它们参与的帧对仍要插帧
        for i in range(lead_pairs):
            if i < T - 1:
                out.append(frames[i] if mask[i] else mids[i])
        for i in range(lead_pairs, T):
            out.append(frames[i])
            if i < T - 1:
                # 跨切点 → 复制前一帧；否则用 RIFE 的中间帧
                out.append(frames[i] if mask[i] else mids[i])

        if skipped:
            self._log(f"跳过 {len(skipped)} 个跨切点帧对（全局帧号 "
                      f"{[g0 + i + 1 for i in skipped][:8]}"
                      f"{'...' if len(skipped) > 8 else ''}）")
        return np.stack(out)
