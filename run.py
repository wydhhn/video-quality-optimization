"""视频增强流水线 —— 分段 → 超分 → 插帧

三个选项：
    --mode sr      超分 ×4（BasicVSR++）
    --mode fi      插帧 ×2（RIFE）
    --mode sr+fi   超分 ×4 再插帧 ×2

三个选项**都会先跑一次镜头分段**，再用分段结果指导后续处理：

  · 超分：切点作为强制分段边界，BasicVSR++ 的时序传播在切点重置
          → 否则 SPyNet 会去算「镜头 A 末帧 ↔ 镜头 B 首帧」的光流（垃圾），
            并把垃圾光流用于特征 warp，导致切点附近拖影/串色
  · 插帧：跨切点的帧对不调用 RIFE，用重复帧代替
          → 否则会把两个不同场景融合成一帧（鬼影）

加 `--no-shot-aware` 可关闭上述处理，用于 A/B 对比。

示例：
    python run.py --video in.mp4 --mode sr
    python run.py --video in.mp4 --mode sr+fi --realtime
    python run.py --video in.mp4 --mode fi --slowmo
    python run.py --video in.mp4 --mode sr+fi --no-shot-aware   # 对照组
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import bootstrap
from bootstrap import ROOT, fmt_time, log

MODES = ["sr", "fi", "sr+fi"]
ENGINES = ["omnishotcut", "transnetv2"]


# --------------------------------------------------------------------------
def get_cuts(video, engine, min_shot, threshold, cuts_file, total_hint=None):
    """跑分段，返回切点帧号列表。"""
    if cuts_file:
        data = json.load(open(cuts_file, encoding="utf-8"))
        if isinstance(data, dict):
            segs = data.get("segments", [])
            cuts = [int(s["start_frame"]) for s in segs[1:]]
        else:
            cuts = [int(c) for c in data]
        log(f"    使用外部切点文件：{len(cuts)} 个切点")
        return cuts, {"source": "file", "file": str(cuts_file), "cuts": cuts}

    mod = __import__(f"engines.{engine}", fromlist=["run"])
    if engine == "transnetv2":
        segs, meta = mod.run(video, threshold=threshold, min_shot=min_shot)
    else:
        segs, meta = mod.run(video, min_shot=min_shot)
    cuts = [int(s["start_frame"]) for s in segs[1:]]
    return cuts, {"source": engine, "cuts": cuts,
                  "n_segments": len(segs), "fps": meta["fps"],
                  "total_frames": meta["total_frames"]}


def build_engines(args, need_sr, need_fi):
    sr = fi = None
    if need_sr:
        backend = getattr(args, "sr_backend", "torch")
        if backend == "trt":
            # TensorRT 版 BasicVSR++。接口与 SrEngine 完全一致，可直接互换。
            # 引擎要预先准备好（见 tools/trt/README.md），没装 TensorRT 会直接报错。
            from pipeline import SrTrtEngine
            # ★ TRT 引擎是**固定形状**的：t 在编译时写死，所以 chunk 必须等于引擎的 t。
            #   默认用 tools/download_trt_engine.py 下到的那个 t=8 引擎。
            t = args.trt_t if getattr(args, "trt_t", 0) else args.chunk
            sr = SrTrtEngine(chunk=t, overlap=0, device=args.device)
        else:
            from pipeline import SrEngine
            sr = SrEngine(fp16=args.fp16, chunk=args.chunk, overlap=args.overlap,
                          device=args.device)
    if need_fi:
        from pipeline import FiEngine
        fi = FiEngine(fp16=args.fp16, batch=args.batch, device=args.device)
    return sr, fi


def main():
    ap = argparse.ArgumentParser(
        description="视频增强流水线：分段 → 超分 → 插帧",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="输入视频")
    ap.add_argument("--mode", required=True, choices=MODES,
                    help="sr=超分 / fi=插帧 / sr+fi=超分+插帧")
    ap.add_argument("--out", default=None, help="输出 mp4（默认自动命名）")
    ap.add_argument("--out-dir", default=str(ROOT / "out"), help="输出目录")

    g = ap.add_argument_group("分段")
    g.add_argument("--engine", default="omnishotcut", choices=ENGINES,
                   help="分段引擎（默认 omnishotcut，当前 SOTA）")
    g.add_argument("--min-shot", type=float, default=0.4, help="最短镜头秒数")
    g.add_argument("--threshold", type=float, default=0.5,
                   help="[transnetv2] 转场概率阈值")
    g.add_argument("--cuts-file", default=None,
                   help="手工指定切点 json（含 segments[].start_frame），跳过分段模型")
    g.add_argument("--no-shot-aware", action="store_true",
                   help="关闭分段感知（用于 A/B 对比）")

    g = ap.add_argument_group("插帧")
    g.add_argument("--realtime", action="store_true", default=True,
                   help="帧率翻倍、原速播放（默认）")
    g.add_argument("--slowmo", dest="realtime", action="store_false",
                   help="帧率不变、时长翻倍（便于逐帧看插入帧）")

    g = ap.add_argument_group("性能")
    g.add_argument("--sr-backend", choices=["torch", "trt"], default="torch",
                   help="[超分] 推理后端。torch=PyTorch（默认，无需预先准备）；"
                        "trt=TensorRT（需先备好引擎，见 tools/trt/README.md。"
                        "实测 TF32 版 78.54 dB / 最大差 1/255，比 PyTorch fp32 快 1.69x）")
    g.add_argument("--trt-t", type=int, default=0,
                   help="[超分] 仅 --sr-backend trt 时用：引擎的帧数 t。"
                        "0=跟随 --chunk。TensorRT 引擎是固定形状，这里必须与引擎实际"
                        "编译的 t 一致，否则会报形状不符")
    g.add_argument("--chunk", type=int, default=8, help="[超分] 分段长度")
    g.add_argument("--overlap", type=int, default=2, help="[超分] 分段重叠帧数")
    g.add_argument("--batch", type=int, default=2, help="[插帧] 批量（5K 下别超 2）")
    g.add_argument("--micro-batch", type=int, default=12, help="微批帧数")
    g.add_argument("--fp16", action="store_true", default=True)
    g.add_argument("--no-fp16", dest="fp16", action="store_false")
    g.add_argument("--device", default=None, help="cuda / cpu（默认自动）")

    g = ap.add_argument_group("输出")
    g.add_argument("--crf", type=int, default=16)
    g.add_argument("--preset", default="medium")

    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        log(f"找不到视频：{video}")
        sys.exit(1)

    do_sr = "sr" in args.mode
    do_fi = "fi" in args.mode
    shot_aware = not args.no_shot_aware

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.mode.replace("+", "")
    if not shot_aware:
        tag += "_noshot"
    out_mp4 = Path(args.out) if args.out else out_dir / f"{video.stem}_{tag}.mp4"

    log(f"\n{'='*66}")
    log(f"  输入  {video}")
    log(f"  模式  {args.mode}"
        + ("（分段感知）" if shot_aware else "（不分段感知，对照组）"))
    log(f"  输出  {out_mp4}")
    log(f"{'='*66}")

    t_all = time.time()

    # ---- 1. 分段 ----
    log(f"\n[1/3] 镜头分段（{args.engine}）")
    if shot_aware:
        cuts, seg_info = get_cuts(video, args.engine, args.min_shot,
                                  args.threshold, args.cuts_file)
        log(f"    {len(cuts)} 个切点 → {len(cuts)+1} 个镜头")
    else:
        cuts, seg_info = [], {"source": "disabled", "cuts": []}
        log("    已关闭（--no-shot-aware）")

    # ---- 2. 引擎 ----
    log(f"\n[2/3] 加载模型")
    sr, fi = build_engines(args, do_sr, do_fi)

    # ---- 3. 流水线 ----
    log(f"\n[3/3] 处理")
    from pipeline import Flow, Progress
    flow = Flow(sr, fi, micro_batch=args.micro_batch)
    prog = Progress()
    rep = flow.run(str(video), str(out_mp4), cut_frames=cuts,
                   shot_aware=shot_aware, do_sr=do_sr, do_fi=do_fi,
                   realtime=args.realtime, crf=args.crf, preset=args.preset,
                   progress=prog)

    # ---- 输出 ----
    seg_path = out_dir / f"{video.stem}_{tag}_segments.json"
    with open(seg_path, "w", encoding="utf-8") as f:
        json.dump({"video": str(video), **seg_info,
                   "shot_aware": shot_aware}, f, ensure_ascii=False, indent=2)

    rep_path = out_dir / f"{video.stem}_{tag}_report.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    log(f"\n{'='*66}")
    log(f"  完成   {rep['resolution'][0]}x{rep['resolution'][1]} @{rep['fps_out']:g}fps  "
        f"{rep['frames_in']} → {rep['frames_out']} 帧  {rep['size_mb']} MB")
    log(f"  分段   {rep['n_cuts']} 个切点"
        + ("（已用于指导处理）" if rep["shot_aware"] else "（未启用）"))
    if "sr_s" in rep["stages_s"]:
        log(f"  超分段 {rep['stages_s']['sr_s']}s")
    if "fi_s" in rep["stages_s"]:
        log(f"  插帧段 {rep['stages_s']['fi_s']}s")
    log(f"  总耗时 {rep['elapsed_s']}s")
    log(f"{'='*66}")
    log(f"  视频     {out_mp4}")
    log(f"  分段     {seg_path}")
    log(f"  报告     {rep_path}")


if __name__ == "__main__":
    main()
