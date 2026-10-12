"""生成「分段算法测试集」视频：把各种该切/不该切的场景串成一段，带真值标注。

场景清单（每个场景都是刻意构造的边界情况）：
  1  静态镜头 A                      → 不该切（基线）
  2  静态镜头 B                      → 段末硬切
  3  缓慢横摇 A（5 秒摇过整幅画面）    → 内部不该切
  4  缓慢推近 B（5 秒推近 1.4 倍）     → 内部不该切
  5  全景 C                          → 段末硬切（同一空间换机位）
  6  特写 C（同一张图的局部放大）      → 段末硬切（同一空间换机位）
  7  静态 A + 中间闪白 2 帧           → 闪白处应切（闪光视为真实场景切换）
  8  快速甩镜 B（1.5 秒横摇整幅）      → 内部不该切（极限）
  9  淡入淡出转场 A→B（2 秒）         → 转场窗口内应有切点
  10 静态镜头 C                      → 结尾

用法：
    python make_test_suite.py --out test_suite.mp4
"""

import argparse
import json
import os

import cv2
import numpy as np

FPS = 25
W, H = 640, 360
ASSETS = ["assets/scene_a.jpg", "assets/scene_b.jpg", "assets/scene_c.jpg"]
BIG_H = 900


def load(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def fit(img, w=W, h=H):
    """等比缩放并居中裁切到 w×h。"""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (int(iw * s) + 1, int(ih * s) + 1), interpolation=cv2.INTER_AREA)
    rh, rw = r.shape[:2]
    y0, x0 = (rh - h) // 2, (rw - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


def static(img, n):
    base = fit(img)
    return [base.copy() for _ in range(n)]


def pan(img, n):
    """连续横摇：在放大的画布上滑窗。"""
    ih, iw = img.shape[:2]
    s = BIG_H / ih
    big = cv2.resize(img, (int(iw * s), BIG_H), interpolation=cv2.INTER_AREA)
    bh, bw = big.shape[:2]
    y0 = (bh - H) // 2
    max_x = max(bw - W, 1)
    return [big[y0:y0 + H, int(max_x * i / max(n - 1, 1)):int(max_x * i / max(n - 1, 1)) + W].copy()
            for i in range(n)]


def zoom(img, n, s0=1.0, s1=1.4):
    """以画面中心为心缓慢推近。"""
    base = fit(img)
    out = []
    for i in range(n):
        s = s0 + (s1 - s0) * (i / max(n - 1, 1))
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
        out.append(cv2.warpAffine(base, M, (W, H), borderMode=cv2.BORDER_REPLICATE))
    return out


def with_flash(img, n, flash_at=0.5, flash_frames=2):
    """在中间插入若干帧全白，模拟闪光灯。"""
    fr = static(img, n)
    k = int(n * flash_at)
    for j in range(flash_frames):
        if k + j < n:
            fr[k + j] = np.full((H, W, 3), 255, dtype=np.uint8)
    return fr


def crossfade(img_a, img_b, n):
    """A 淡出、B 淡入。"""
    a, b = fit(img_a).astype(np.float32), fit(img_b).astype(np.float32)
    out = []
    for i in range(n):
        k = i / max(n - 1, 1)
        out.append(((1 - k) * a + k * b).astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_suite.mp4")
    ap.add_argument("--gt", default=None, help="真值 json 输出路径")
    args = ap.parse_args()

    A, B, C = (load(p) for p in ASSETS)
    # 场景 5/6：同一张图的「全景」和「局部放大」
    c_close = fit(C[60:460, 150:790] if C.shape[0] > 460 and C.shape[1] > 790 else C)

    segs = []
    t = 0.0

    def add(name, frames, note=""):
        nonlocal t
        start = t
        t += len(frames) / FPS
        segs.append({"name": name, "start": round(start, 2),
                     "end": round(t, 2), "note": note, "frames": frames})

    add("静态镜头 A（基线）", static(A, 3 * FPS), "不应切")
    add("静态镜头 B", static(B, 3 * FPS))
    add("缓慢横摇 A（5 秒摇过整幅）", pan(A, 5 * FPS), "内部不应切")
    add("缓慢推近 B（5 秒推近 1.4 倍）", zoom(B, 5 * FPS), "内部不应切")
    add("全景 C", static(C, 2 * FPS))
    add("特写 C（同一空间换机位）", [c_close.copy() for _ in range(2 * FPS)], "同一空间换机位，期望切")
    add("静态 A + 中间闪白 2 帧", with_flash(A, 3 * FPS), "闪白不该切")
    add("快速甩镜 B（1.5 秒）", pan(B, int(1.5 * FPS)), "极限情况")
    add("淡入淡出转场 A→B", crossfade(A, B, 2 * FPS), "转场窗口内应有切点")
    add("静态镜头 C", static(C, 3 * FPS), "不应切")

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter 打开失败")
    for s in segs:
        for f in s["frames"]:
            writer.write(f)
    writer.release()

    # ---- 真值 ----
    # 闪光（相机补光 / 闪白转场）视为一次真实的场景切换，允许在其处切断，
    # 所以闪白位置算「预期切点」，不进 forbidden。
    bounds = [round(s["end"], 2) for s in segs[:-1]]
    for s in segs:
        if "闪白" in s["name"]:
            n_frames = round((s["end"] - s["start"]) * FPS)
            bounds.append(round(s["start"] + int(n_frames * 0.5) / FPS, 2))
    bounds = sorted(bounds)

    forbidden = []
    for s in segs:
        if s["note"] == "内部不应切":
            forbidden.append({"start": round(s["start"] + 0.4, 2),
                              "end": round(s["end"] - 0.4, 2), "name": s["name"]})
    forbidden.append({"start": 0.4, "end": round(segs[0]["end"] - 0.4, 2),
                      "name": segs[0]["name"]})
    forbidden.append({"start": round(segs[-1]["start"] + 0.4, 2),
                      "end": round(segs[-1]["end"] - 0.4, 2), "name": segs[-1]["name"]})
    for s in segs:
        if "甩镜" in s["name"]:
            forbidden.append({"start": round(s["start"] + 0.2, 2),
                              "end": round(s["end"] - 0.2, 2), "name": s["name"]})

    gt = {
        "duration": round(t, 2),
        "fps": FPS,
        "size": [W, H],
        "scenarios": [{k: v for k, v in s.items() if k != "frames"} for s in segs],
        "expected_cuts": bounds,
        "transition_window": {"start": segs[8]["start"], "end": segs[8]["end"],
                              "name": "淡入淡出转场（窗口内应有切点）"},
        "forbidden": forbidden,
    }
    dst = args.gt or os.path.splitext(args.out)[0] + "_gt.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)

    print(f"已生成 {args.out}（{t:.2f} 秒 / {int(t * FPS)} 帧）")
    print(f"已生成真值 {dst}")
    print(f"  预期切点：{bounds}")
    print(f"  转场窗口：{gt['transition_window']['start']}–{gt['transition_window']['end']}")
    print(f"  不应切的区间：{len(forbidden)} 个")
    for s in segs:
        print(f"    {s['start']:>6.2f}-{s['end']:>6.2f}s  {s['name']}"
              + (f"   [{s['note']}]" if s["note"] else ""))


if __name__ == "__main__":
    main()
