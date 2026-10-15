"""编译 VSR 的 TensorRT 引擎。

三种模式实测（480×836 → ×4，对 PyTorch fp32 真值）：

| mode | ms/帧 | 加速 | PSNR | 最大差 | 建议 |
| --- | --- | --- | --- | --- | --- |
| **tf32** | 260.9 | 1.69x | **78.54 dB** | **1.0** | ⭐ 默认 |
| fp32 | 262.8 | 1.41x | 72.22 dB | 1.0 | 精度够但更慢 |
| fp16 | 93.3 | 3.97x | 58.93 dB | 21.0 | 快但精度不达标 |

**为什么默认 TF32 而不是 fp16**：
TF32 是 10 位尾数 + fp32 指数范围，对卷积的精度远好于 fp16，
速度又比纯 fp32 快。而 `BuilderFlag.FP16` 是"整体降精度"，
等价于 PyTorch 里那个会掉 37 dB 的 `net.half()`。

    python tools/trt/build_trt_vsr.py --onnx <导出.onnx> --mode tf32 --out xxx.trt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (_ROOT, HERE):
    sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="export_vsr_onnx.py 导出的 ONNX")
    ap.add_argument("--out", default=None, help="输出 .trt 路径")
    ap.add_argument("--mode", default="tf32", choices=["tf32", "fp32", "fp16"])
    ap.add_argument("--workspace-gb", type=float, default=10.0)
    args = ap.parse_args()

    import tensorrt as trt

    onnx_path = os.path.abspath(args.onnx)
    if not os.path.isfile(onnx_path):
        print(f"❌ 找不到 {onnx_path}")
        return 1
    out = args.out
    if not out:
        base = os.path.basename(onnx_path).replace(".onnx", "").replace("vsr_decomp_", "vsr_")
        out = os.path.join(os.path.dirname(onnx_path), f"{base}_{args.mode}.trt")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # ONNX 可能带外部数据（.weight / _Constant_* 等），TRT 按 CWD 找它们，
    # 所以要先切到 ONNX 所在目录再解析。
    cwd = os.getcwd()
    os.chdir(os.path.dirname(onnx_path) or ".")
    try:
        if not parser.parse_from_file(os.path.basename(onnx_path)):
            print("❌ ONNX 解析失败：")
            for i in range(parser.num_errors):
                print("   ", parser.get_error(i))
            return 1
        print(f"  网络 {network.num_layers} 层，{network.num_inputs} 入 / "
              f"{network.num_outputs} 出")

        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                  int(args.workspace_gb * 2**30))
        if args.mode == "tf32":
            cfg.set_flag(trt.BuilderFlag.TF32)       # ★ 只开 TF32
        elif args.mode == "fp16":
            cfg.set_flag(trt.BuilderFlag.FP16)       # ☆ 快但精度掉得厉害
        else:
            cfg.clear_flag(trt.BuilderFlag.TF32)
            cfg.clear_flag(trt.BuilderFlag.FP16)

        print(f"  编译中（{args.mode}，约 4~5 分钟）...", flush=True)
        t0 = time.time()
        ser = builder.build_serialized_network(network, cfg)
        if ser is None:
            print("❌ 构建返回空（通常是显存/内存不足，或图有 TRT 不支持的结构）")
            return 1
        with open(os.path.abspath(out), "wb") as f:
            f.write(ser)
    finally:
        os.chdir(cwd)

    size = os.path.getsize(out) / 2**20
    print(f"  ✅ 完成 {size:.1f} MB  {time.time()-t0:.0f}s")
    print(f"     {out}")
    print()
    print("  下一步：验证精度")
    print(f"     python tools/trt/bench_trt_vsr.py --engine {out} --t 8 --size 836 480")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
