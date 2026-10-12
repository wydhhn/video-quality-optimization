"""生成一段合成测试视频：3 个大场景，每个场景内含 4 个镜头（硬切）。

用途：验证「镜头切分」和「场景聚合」是否真的在工作——
正确的输出应该是 12 个镜头、3 个场景。

用法：
    python make_test_video.py --out test_video.mp4
"""

import argparse
import numpy as np
import cv2

FPS = 25
W, H = 640, 360
SHOT_SECONDS = 2          # 每个镜头 2 秒
SHOTS_PER_SCENE = 4       # 每个场景 4 个镜头


def draw_shot(scene_id, shot_id, frame_idx_in_shot, total):
    """按 (场景, 镜头) 画一帧：场景决定主色调，镜头决定构图。"""
    palettes = [
        {"bg": (40, 40, 200), "fg": (255, 255, 255)},    # 场景0 红
        {"bg": (200, 120, 30), "fg": (255, 255, 255)},   # 场景1 蓝
        {"bg": (60, 160, 60), "fg": (10, 10, 10)},       # 场景2 绿
    ]
    p = palettes[scene_id % len(palettes)]
    img = np.full((H, W, 3), p["bg"], dtype=np.uint8)

    phase = frame_idx_in_shot / max(total, 1)
    if shot_id % 2 == 0:
        # 偶数镜头：画面中央一个圆（缓慢放大，模拟镜头内运动）
        r = int(60 + 20 * phase)
        cv2.circle(img, (W // 2, H // 2), r, p["fg"], -1)
    else:
        # 奇数镜头：画面右侧一个方块 + 左侧竖条（构图完全不同）
        cv2.rectangle(img, (W - 200, 60), (W - 60, 200), p["fg"], -1)
        cv2.rectangle(img, (40, 40), (110, H - 40), p["fg"], -1)

    cv2.putText(img, f"S{scene_id}-SHOT{shot_id}", (20, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, p["fg"], 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_video.mp4")
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--shots-per-scene", type=int, default=SHOTS_PER_SCENE)
    args = ap.parse_args()

    frames_per_shot = FPS * SHOT_SECONDS
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter 打开失败")

    total_shots = 0
    for s in range(args.scenes):
        for k in range(args.shots_per_scene):
            for f in range(frames_per_shot):
                writer.write(draw_shot(s, k, f, frames_per_shot))
            total_shots += 1
    writer.release()

    duration = args.scenes * args.shots_per_scene * SHOT_SECONDS
    print(f"已生成 {args.out}")
    print(f"  真值：{args.scenes} 个场景 / {total_shots} 个镜头 / {duration} 秒")


if __name__ == "__main__":
    main()
