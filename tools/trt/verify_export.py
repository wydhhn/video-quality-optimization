"""导出后的**第一道自检**：用 ONNX Runtime 跑导出的图，和 PyTorch 对比。

为什么这一步不能省
------------------
导出 dummy 用错（比如 `torch.zeros`）会让导出的图**结构残缺**，
而现象只是"精度差一点"，你会去怀疑精度、怀疑算子、怀疑混合精度 ——
在错误的方向上耗很久。

**这一步 30 秒，能直接告诉你图对不对。**

判定标准：
    > 90 dB   ✅ 图完整，可以往下编译引擎
    50~60 dB  ❌ 图残缺，回去检查 dummy（必须是真实视频帧）
    < 20 dB   ❌ 可能喂错了 dtype

    python tools/trt/verify_export.py --onnx xxx.onnx --t 2 --size 836 480
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (_ROOT, _ROOT + "/vendor", HERE):
    sys.path.insert(0, _p)

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--t", type=int, default=2)
    ap.add_argument("--size", type=int, nargs=2, default=[836, 480], help="H W")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--clip", default=None, help="真实视频（默认用 testdata 里的）")
    args = ap.parse_args()
    h, w = args.size

    # ── 输入：必须是**真实视频帧**，不能用 zeros 或噪声（见文档第 8 节）──
    clip = args.clip
    if not clip:
        for cand in ("testdata/test_video.mp4", "testdata/test_video_real.mp4"):
            if os.path.isfile(os.path.join(_ROOT, cand)):
                clip = os.path.join(_ROOT, cand)
                break
    if not clip or not os.path.isfile(clip):
        print("❌ 找不到测试视频，用 --clip 指定一个")
        return 1

    from tools.trt.export_vsr_onnx import decode_frames
    src = decode_frames(clip, args.t)
    x = torch.from_numpy(np.ascontiguousarray(src)).permute(0, 3, 1, 2)[None]
    x = x.float().div_(255.0)
    if x.shape[-2:] != (h, w):
        x = torch.nn.functional.interpolate(x[0], size=(h, w), mode="bilinear",
                                            align_corners=False)[None]
    print(f"  输入 {args.t} 帧 {w}x{h}   （真实帧，非 zeros）")

    # ── PyTorch 参考（直接调 net，不分块）──
    from pipeline.sr_trt import find_engine  # noqa: F401  （仅为统一路径）
    import vsr as _vsr
    from vsr.checkpoint import load_checkpoint
    from bootstrap import WEIGHTS

    net = _vsr.BasicVSRPlusPlusNet(num_feat=64, num_block=7, spynet_pretrained=None)
    load_checkpoint(net, str(WEIGHTS / "basicvsr_plusplus_reds4.pth"), strict=False)
    net = net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
        x = x.cuda()
    with torch.inference_mode():
        ref = net(x)[0].permute(0, 2, 3, 1).float().clamp(0, 1).mul(255).round()
        ref = ref.to(torch.uint8).cpu().numpy()
    del net
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ── ONNX Runtime ──
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(args.onnx)) or ".")
    try:
        sess = ort.InferenceSession(os.path.basename(args.onnx), so,
                                    providers=["CUDAExecutionProvider",
                                               "CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        o = sess.run(None, {name: x.cpu().numpy().astype(np.float32)})[0]
    finally:
        os.chdir(cwd)

    o = np.transpose(o[0], (0, 2, 3, 1))
    o = np.round(np.clip(o, 0, 1) * 255).astype(np.uint8)
    d = np.abs(ref.astype(np.float64) - o.astype(np.float64))
    p = psnr(ref, o)

    print()
    print(f"  ORT vs PyTorch :  PSNR {p:.2f} dB   最大差 {d.max():.1f}")
    print()
    if p > 90:
        print("  ✅ 图完整，可以往下编译 TensorRT 引擎")
    elif p > 45:
        print("  ❌ 图残缺！最可能的原因：**导出 dummy 用了 torch.zeros**")
        print("     修法见 docs/TRT引擎构建全流程.md 第 3.1 节")
    else:
        print("  ❌ 差异过大。检查输入 dtype / 输入的帧数是否与导出时一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
