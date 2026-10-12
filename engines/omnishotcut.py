"""OmniShotCut 引擎 —— Shot-Query Transformer 直接预测镜头区间

模型：OmniShotCut v1.5（NeurIPS 2026，UVA CV Lab）
  · ResNet18 backbone + DETR 风格的 Shot-Query Transformer（24 query / 6 层 decoder）
  · 输入 96×128（像素数是 TransNetV2 的 9.5 倍），52.8M 参数
  · 不像 TransNetV2 那样逐帧打分，而是**直接预测切点位置 + 转场类型**
  · 官方 v1.5 专门优化过「连续运镜 / 光照变化下的误切」和「密集快切召回」

本地适配（vendor/omnishotcut 里已打好补丁）：
  · 官方代码把设备硬编码成 "cuda" → 改成自动选择，无显卡也能跑
  · 官方用 ffmpeg.probe() 取帧率，但 ffprobe 不在依赖里 → 改用 OpenCV
  · ffmpeg-python 需要 PATH 上有 ffmpeg → 项目自带 tools/bin/ffmpeg.exe
"""

import os

import cv2
import numpy as np

from bootstrap import WEIGHTS, ensure_ffmpeg, log, to_segments

WEIGHT_FILE = WEIGHTS / "OmniShotCut_ckpt.pth"
WEIGHT_URL = ("https://hf-mirror.com/uva-cv-lab/OmniShotCut_v1.5"
              "/resolve/main/OmniShotCut_ckpt.pth")


def ensure_weights():
    if WEIGHT_FILE.exists() and WEIGHT_FILE.stat().st_size > 100_000_000:
        return WEIGHT_FILE
    import urllib.request
    WEIGHT_FILE.parent.mkdir(parents=True, exist_ok=True)
    log(f"    下载 OmniShotCut v1.5 权重（约 201MB）→ {WEIGHT_FILE}")
    urllib.request.urlretrieve(WEIGHT_URL, WEIGHT_FILE)
    return WEIGHT_FILE


_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    ensure_ffmpeg()
    import omnishotcut
    log(f"    加载 OmniShotCut v1.5（{WEIGHT_FILE.stat().st_size / 1024 / 1024:.0f} MB）")
    _MODEL = omnishotcut.load(str(ensure_weights()))
    return _MODEL


def _filter_cuts(cuts, total, fps, min_shot_sec):
    """最短镜头过滤：相邻切点间距不足 min_shot 的丢弃。

    这是本引擎**唯一**的过滤手段。理由见 `run()` 的说明。
    """
    min_frames = max(min_shot_sec * fps, 1.0)
    out, prev = [], 0
    for c in sorted({int(x) for x in cuts}):
        if 0 < c < total and c - prev >= min_frames:
            out.append(c)
            prev = c
    return out


def run(video_path, min_shot=0.4, overlap=20, label_filter="all"):
    """返回 (segments, meta)。

    ── 关于 overlap（默认 20，官方 CLI 是 10）────────────────────────────
    这不是可有可无的调参。官方 overlap=10 时，落在窗口起点附近的切点会被模型
    打成 `Padding` 标签，再被官方的 `mode="clean_shot"` **静默丢弃**。

    实测 mailbox 视频（11 个真值切点，每 4.08s 一个）：
        overlap=10 → 只检出 10 个，漏掉第 1 个切点
        overlap=20 → 11 个全中
    原因是该切点在窗口内的局部位置从 7 移到 17，标签由 Padding 恢复正常。

    ── 关于 label_filter ────────────────────────────────────────────────
    官方 `mode="clean_shot"` 会按「转场类型」筛掉所有非 General 的区间
    （Dissolve / Wipes / Push / Slide / Zoom / Fade / Doorway / Padding）。
    但这些**全都是真实的镜头边界**——Push/Slide/Zoom 本身就是运镜，
    Dissolve/Fade/Wipe 是软转场。丢掉它们等于漏检。

    三个视频的实测对照（真值已知）：
        策略             测试集（10 切点）      mailbox（11 切点）
        clean_shot       10/10，最大误差 0.080s  10/11（漏 1）✗
        non_padding      10/10，最大误差 0.000s  10/11（漏 1）✗
        all（默认）      10/10，最大误差 0.000s  11/11 ✓

    所以默认 **不做标签过滤**，只用 `min_shot` 去噪——短命区间（例如测试集里
    被误切成 2 帧的那段）会被 min_shot 自然滤掉，而真实边界全部保留。
    标签只作为元信息记录在 meta 里。

    label_filter:
      "all"         全部保留（默认，实测三个视频全部满分）
      "non_padding" 丢掉 Padding 区间
      "hard_only"   只保留 General（等价于官方 clean_shot）

    注意：不做闪光抑制。闪光（相机补光、闪白转场）在成片里往往就是一次真实的
    场景切换，切断是合理的；而且实测它对闪白的反应和 TransNetV2 一致。
    """
    ensure_ffmpeg()
    model = _load_model()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()

    log(f"    OmniShotCut 推理（overlap={overlap}, 标签过滤={label_filter}）…")
    # 用 "all" 模式拿回「区间 + 转场类型标签」，由我们自己决定保留哪些。
    # （官方 clean_shot 会在模型内部就丢掉非 General 区间，我们连丢了多少都看不见）
    # 官方 inference() 只认 str（传 Path 会被当成 numpy 数组解析而报错）
    ranges, intra_labels, inter_labels = model.inference(
        str(video_path), mode="all", overlap=overlap)

    dist = {}
    for lab in intra_labels:
        dist[lab] = dist.get(lab, 0) + 1
    if dist:
        summary = ", ".join(f"{k}×{v}" for k, v in
                            sorted(dist.items(), key=lambda kv: -kv[1]))
        log(f"    转场类型分布: {summary}")

    if label_filter == "hard_only":
        keep = [i for i, l in enumerate(intra_labels) if l == "General"]
    elif label_filter == "non_padding":
        keep = [i for i, l in enumerate(intra_labels) if l != "Padding"]
    else:
        keep = list(range(len(ranges)))

    if len(keep) != len(ranges):
        log(f"    标签过滤（{label_filter}）丢弃了 "
            f"{len(ranges) - len(keep)} / {len(ranges)} 个区间")

    # ranges 是 [start, end) 的镜头区间列表 → 取每段起点作为切点
    raw_cuts = [int(ranges[i][0]) for i in keep if int(ranges[i][0]) > 0]
    raw = len(raw_cuts)
    cuts = _filter_cuts(raw_cuts, total, fps, min_shot)

    meta = {
        "engine": "omnishotcut",
        "overlap": overlap,
        "label_filter": label_filter,
        "label_dist": dist,
        "min_shot": min_shot,
        "raw_cuts": raw,
        "n_ranges": len(ranges),
        "fps": fps, "total_frames": total,
        "width": w, "height": h,
        "duration": total / fps,
    }
    return to_segments(cuts, total, fps), meta
