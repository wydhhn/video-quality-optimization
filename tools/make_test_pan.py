"""生成「长镜头缓慢摇镜」测试视频，用来验证分段算法会不会把运镜误当成切换。

结构：3 张照片，每张做一段 6 秒的**连续缓慢横摇**（一个镜头内，无剪辑），
      段与段之间是**硬切**。真值：3 个镜头（2 个硬切点），0 个运镜切点。

用法：
    python make_test_pan.py --out test_video_pan.mp4
"""

import argparse
import cv2
import numpy as np

FPS = 25
W, H = 640, 360
PAN_SECONDS = 6.0
IMAGES = ["assets/scene_a.jpg", "assets/scene_b.jpg", "assets/scene_c.jpg"]
BIG_H = 900          # 放大后的画布高度，保证有足够的横摇余量


def load_big(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = BIG_H / h
    return cv2.resize(img, (int(w * s), BIG_H), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_video_pan.mp4")
    ap.add_argument("--pan-seconds", type=float, default=PAN_SECONDS)
    args = ap.parse_args()

    n_frames = int(FPS * args.pan_seconds)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter 打开失败")

    for path in IMAGES:
        big = load_big(path)
        bh, bw = big.shape[:2]
        y0 = (bh - H) // 2
        max_x = bw - W
        for f in range(n_frames):
            x = int(max_x * f / max(n_frames - 1, 1))       # 从最左缓慢摇到最右
            writer.write(big[y0:y0 + H, x:x + W].copy())
    writer.release()

    total = len(IMAGES) * args.pan_seconds
    print(f"已生成 {args.out}")
    print(f"  真值：{len(IMAGES)} 个镜头 / {len(IMAGES) - 1} 个硬切点 / "
          f"0 个运镜切点 / 共 {total:.0f} 秒")


if __name__ == "__main__":
    main()
