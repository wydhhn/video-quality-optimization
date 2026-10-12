"""用三张真实照片合成"多场景"测试视频（模拟同一场景内的不同机位）

每个场景 = 一张照片，场景内 4 个镜头用不同裁切/缩放模拟不同机位；
场景之间是彻底的语义切换（雪山风景 / 城市街道 / 食物特写）。
真值：3 个场景 / 12 个镜头。

用法：
    python make_test_video_real.py --out test_video_real.mp4
"""

import argparse
import cv2
import numpy as np

FPS = 25
W, H = 640, 360
SHOT_SECONDS = 3
SHOTS_PER_SCENE = 4

IMAGES = ["assets/scene_a.jpg", "assets/scene_b.jpg", "assets/scene_c.jpg"]
# 场景内 4 个镜头的裁切方式（x0,y0,x1,y1 相对比例）
CROPS = [
    (0.00, 0.00, 1.00, 1.00),   # 全景
    (0.15, 0.10, 0.70, 0.70),   # 中景
    (0.45, 0.25, 0.95, 0.85),   # 另一侧中景
    (0.30, 0.00, 0.75, 0.55),   # 特写
]


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"读不到图片：{path}")
    h, w = img.shape[:2]
    scale = max(W / w, H / h)
    img = cv2.resize(img, (int(w * scale) + 1, int(h * scale) + 1))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_video_real.mp4")
    ap.add_argument("--shot-seconds", type=float, default=SHOT_SECONDS)
    args = ap.parse_args()

    frames_per_shot = int(FPS * args.shot_seconds)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter 打开失败")

    for s, path in enumerate(IMAGES):
        big = load_image(path)
        bh, bw = big.shape[:2]
        for k, (x0, y0, x1, y1) in enumerate(CROPS):
            crop = big[int(y0 * bh):int(y1 * bh), int(x0 * bw):int(x1 * bw)]
            base = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
            for f in range(frames_per_shot):
                # 轻微推镜，避免完全静止
                zoom = 1.0 + 0.06 * (f / frames_per_shot)
                M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, zoom)
                fr = cv2.warpAffine(base, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
                writer.write(fr)
    writer.release()
    print(f"已生成 {args.out}")
    print(f"  真值：{len(IMAGES)} 个场景 / {len(IMAGES)*len(CROPS)} 个镜头 / "
          f"{len(IMAGES)*len(CROPS)*args.shot_seconds:.0f} 秒")


if __name__ == "__main__":
    main()
