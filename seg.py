"""视频镜头切分 —— 统一入口

两种引擎：
  omnishotcut   ★ 默认。OmniShotCut v1.5（NeurIPS 2026 SOTA），Shot-Query Transformer
                直接预测切点位置 + 转场类型（201MB）
  transnetv2    TransNetV2，逐帧转场概率 + 阈值（30MB，轻量、无外部依赖）

用法：
  python seg.py --video 视频.mp4                        # 默认 OmniShotCut（SOTA）
  python seg.py --video 视频.mp4 --engine transnetv2    # 换轻量模型
  python seg.py --video 视频.mp4 --engine both --slice --overlay

输出（默认写到 out/）：
  segments_<engine>.json    分段结果（起止时间 / 帧号）
  segments_<engine>.csv     同上，表格格式
  clips_<engine>/           按段切好的视频（--slice）
  overlay_<engine>.mp4      带分段标注的完整视频（--overlay）
  compare.json              两个引擎的切点对比（--engine both）
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import bootstrap
from bootstrap import ROOT, fmt_time, log

ENGINES = ["omnishotcut", "transnetv2"]


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
def save_segments(out_dir, name, segs, meta):
    json_path = out_dir / f"segments_{name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "video": meta.pop("video_path", None),
            "engine": name,
            "fps": meta["fps"],
            "total_frames": meta["total_frames"],
            "duration": round(meta["duration"], 3),
            "resolution": f"{meta['width']}x{meta['height']}",
            "n_segments": len(segs),
            "options": {k: v for k, v in meta.items()
                        if k not in ("prob", "fps", "total_frames",
                                     "duration", "width", "height")},
            "segments": segs,
        }, f, ensure_ascii=False, indent=2)

    csv_path = out_dir / f"segments_{name}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "start_sec", "end_sec", "duration_sec",
                    "start_frame", "end_frame"])
        for s in segs:
            w.writerow([s["id"], s["start"], s["end"], s["duration"],
                        s["start_frame"], s["end_frame"]])
    return json_path, csv_path


def slice_video(video_path, segs, out_dir):
    """按段切分视频（重编码，精确到帧）。"""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.iterdir():                       # 清掉上次的切片
        if f.name.startswith("shot_") and f.suffix == ".mp4":
            f.unlink()
    for s in segs:
        dst = out_dir / f"shot_{s['id']:03d}_{s['start']:.2f}s-{s['end']:.2f}s.mp4"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                        "-ss", str(s["start"]), "-to", str(s["end"]),
                        "-i", str(video_path),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", str(dst)], check=False)
    log(f"    切片 {len(segs)} 个 → {out_dir}")


def render_overlay(video_path, segs, out_path, label):
    cmd = [sys.executable, str(ROOT / "tools" / "render_overlay.py"),
           "--video", str(video_path),
           "--scenes-json", json.dumps(segs, ensure_ascii=False),
           "--out", str(out_path), "--label", label]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"    标注视频生成失败：{(r.stderr or r.stdout)[-300:]}")
        return None
    log(f"    标注视频 → {out_path}")
    return out_path


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run_engine(name, video, args, out_dir):
    mod = __import__(f"engines.{name}", fromlist=["run"])
    if name == "transnetv2":
        segs, meta = mod.run(video, threshold=args.threshold,
                             min_shot=args.min_shot,
                             flash_suppress=args.flash_suppress,
                             step=args.step,
                             center_select=args.select == "center")
    else:
        segs, meta = mod.run(video, min_shot=args.min_shot, overlap=args.overlap,
                             label_filter=args.label_filter)
    meta["video_path"] = str(video)
    return segs, meta


def print_result(name, segs, meta):
    log(f"\n  ── {name} ── {len(segs)} 段")
    log(f"  {'段':>4} {'起始':>10} {'结束':>10} {'时长':>8}")
    log("  " + "-" * 38)
    for s in segs:
        log(f"  {s['id']:>4} {fmt_time(s['start']):>10} {fmt_time(s['end']):>10} "
            f"{s['duration']:7.2f}s")
    log("  " + "-" * 38)
    covered = sum(s["duration"] for s in segs)
    if abs(covered - meta["duration"]) > 0.05:
        log(f"  ⚠ 覆盖异常：分段合计 {covered:.2f}s / 视频 {meta['duration']:.2f}s")


def main():
    ap = argparse.ArgumentParser(
        description="视频镜头切分（TransNetV2 / OmniShotCut）")
    ap.add_argument("--video", required=True)
    ap.add_argument("--engine", default="omnishotcut",
                    choices=ENGINES + ["both"],
                    help="默认 omnishotcut（当前 SOTA，NeurIPS 2026）；"
                         "transnetv2 更轻量（30MB，无外部依赖）")
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--min-shot", type=float, default=0.4, help="最短镜头秒数")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="[transnetv2] 转场概率阈值，漏切调低")
    ap.add_argument("--flash-suppress", action="store_true",
                    help="[transnetv2] 开启闪光抑制（默认关闭）。默认会把闪光当成"
                         "正常切换——闪光在成片里通常就是一次真实的场景变化")
    ap.add_argument("--step", type=int, default=50,
                    help="[transnetv2] 滑窗步长，越小候选位置越多越准但越慢")
    ap.add_argument("--select", default="center", choices=["center", "first_half"],
                    help="[transnetv2] 每帧取哪次预测：center=最接近窗口正中（推荐，"
                         "实测能把漏检切点从 0.44 救回 0.63）；first_half=官方做法")
    ap.add_argument("--overlap", type=int, default=20,
                    help="[omnishotcut] 相邻窗口重叠帧数（默认 20；官方 CLI 是 10，"
                         "但 10 会让落在窗口起点附近的切点被标成 Padding 而漏检）")
    ap.add_argument("--label-filter", default="all",
                    choices=["all", "non_padding", "hard_only"],
                    help="[omnishotcut] 按转场类型过滤区间：all=全保留（默认，最稳）；"
                         "non_padding=丢掉 Padding；hard_only=只留 General（等价官方 clean_shot）")
    ap.add_argument("--slice", action="store_true", help="导出按段切好的视频")
    ap.add_argument("--overlay", action="store_true", help="导出带标注的完整视频")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        log(f"找不到视频：{video}")
        sys.exit(1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = ENGINES if args.engine == "both" else [args.engine]

    results = {}
    for name in targets:
        log(f"\n[{name}]")
        segs, meta = run_engine(name, video, args, out_dir)
        results[name] = (segs, meta)
        print_result(name, segs, meta)
        j, c = save_segments(out_dir, name, segs, dict(meta))
        log(f"    已保存 {j.name} / {c.name}")
        if args.slice:
            slice_video(video, segs, out_dir / f"clips_{name}")
        if args.overlay:
            render_overlay(video, segs, out_dir / f"overlay_{name}.mp4", name)

    # 两个引擎的对比
    if len(results) == 2:
        a, b = results["transnetv2"][0], results["omnishotcut"][0]
        ca = [s["start"] for s in a[1:]]
        cb = [s["start"] for s in b[1:]]
        log(f"\n[对比]")
        log(f"  TransNetV2   {len(a)} 段")
        log(f"  OmniShotCut  {len(b)} 段")
        log(f"\n  {'TransNetV2 切点':>16} {'OmniShotCut 切点':>18} {'差异':>8}")
        log("  " + "-" * 46)
        for i in range(max(len(ca), len(cb))):
            x = f"{ca[i]:.2f}s" if i < len(ca) else "—"
            y = f"{cb[i]:.2f}s" if i < len(cb) else "—"
            d = f"{abs(ca[i] - cb[i]):.2f}s" if i < len(ca) and i < len(cb) else ""
            log(f"  {x:>16} {y:>18} {d:>8}")
        only_a = [f"{c:.2f}" for c in ca if all(abs(c - d2) > 0.3 for d2 in cb)]
        only_b = [f"{c:.2f}" for c in cb if all(abs(c - d2) > 0.3 for d2 in ca)]
        if only_a:
            log(f"\n  仅 TransNetV2 检出：{', '.join(only_a)}")
        if only_b:
            log(f"  仅 OmniShotCut 检出：{', '.join(only_b)}")
        with open(out_dir / "compare.json", "w", encoding="utf-8") as f:
            json.dump({"transnetv2": ca, "omnishotcut": cb,
                       "only_transnetv2": only_a, "only_omnishotcut": only_b},
                      f, ensure_ascii=False, indent=2)

    log(f"\n全部结果已写入：{out_dir}")


if __name__ == "__main__":
    main()
