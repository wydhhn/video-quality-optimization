"""把分段结果直接画在完整视频上，输出一个新的 mp4。

叠加内容：
  - 底部时间轴：按场景着色，带镜头刻度 + 当前播放位置指示线
  - 左上角：当前场景编号 / 总段数、场景起止时间、当前时间
  - 切换瞬间：屏幕中央弹出「▶ 场景 N 开始」横幅（持续 1 秒）

用法：
  python render_overlay.py --video test_video_real.mp4 \
      --scenes out/scenes_bassl.json \
      --out "视频分段结果/完整视频_bassl_分段标注.mp4" \
      --label bassl
"""

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

# 与 report.html 一致的配色（浅色系填充 + 深色描边）
PALETTE = [(230, 241, 251), (225, 245, 238), (250, 238, 218), (251, 234, 240),
           (238, 237, 254), (250, 236, 231), (234, 243, 222), (241, 239, 232)]
STROKE = [(24, 95, 165), (15, 110, 86), (133, 79, 11), (153, 53, 86),
          (83, 74, 183), (153, 60, 29), (59, 109, 17), (95, 94, 90)]

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]


def load_font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def fmt(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def scene_at(scenes, t):
    for i, s in enumerate(scenes):
        if s["start"] <= t < s["end"]:
            return i, s
    return len(scenes) - 1, scenes[-1]


def scenario_at(labels, t):
    """在测试集视频里找到当前所处的测试场景（名称 + 备注）。"""
    if not labels:
        return None
    for s in labels:
        if s["start"] <= t < s["end"]:
            return s
    return labels[-1]


def draw_timeline(img, scenes, shots, t, total, fps):
    """底部时间轴。img 是 BGR numpy 数组，原地修改。"""
    h, w = img.shape[:2]
    bar_h = max(int(h * 0.095), 30)
    top = h - bar_h - int(h * 0.05)
    x0, x1 = int(w * 0.03), int(w * 0.97)
    span = max(total, 1e-6)

    def X(tt):
        return x0 + (x1 - x0) * min(max(tt / span, 0.0), 1.0)

    # 半透明底板
    pad = 8
    y0 = max(top - pad, 0)
    roi = img[y0:h, max(x0 - pad, 0):x1 + pad]
    if roi.size:
        img[y0:h, max(x0 - pad, 0):x1 + pad] = cv2.addWeighted(
            roi, 0.2, np.zeros_like(roi), 0.8, 0)

    # 场景色块
    for s in scenes:
        c = s["scene_id"] % len(PALETTE)
        cv2.rectangle(img, (int(X(s["start"])), top), (int(X(s["end"])), top + bar_h),
                      PALETTE[c][::-1], -1)
        cv2.rectangle(img, (int(X(s["start"])), top), (int(X(s["end"])), top + bar_h),
                      STROKE[c][::-1], 2)

    # 镜头刻度（时间轴上方的小竖线）
    for sh in shots:
        xx = int(X(sh["start"]))
        cv2.line(img, (xx, top - 6), (xx, top - 1), (255, 255, 255), 2)

    # 播放位置指示线
    px = int(X(t))
    cv2.line(img, (px, top - 10), (px, top + bar_h + 6), (255, 255, 255), 3)
    cv2.line(img, (px, top - 10), (px, top + bar_h + 6), (30, 30, 30), 1)
    return top, bar_h


def draw_frame_border(img, accent_bgr, thick):
    """整帧彩色边框：场景一变颜色就变，扫一眼就知道切在哪。"""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), accent_bgr, thick)


def draw_banner(d, w, h, text, font, fill, accent, alpha):
    """屏幕中央的「场景 N 开始」横幅（三角用手绘，避免字体缺字形）。"""
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad_l, pad_r, pad_v = 34, 16, 10
    bw = tw + pad_l + pad_r
    bx = (w - bw) // 2
    by = int(h * 0.42)
    d.rounded_rectangle([bx, by - pad_v, bx + bw, by + th + pad_v + 4],
                        radius=6, fill=fill + (alpha,),
                        outline=accent + (alpha,), width=2)
    tx, ty = bx + pad_l, by
    d.polygon([(tx - 20, ty), (tx - 20, ty + th + 4),
               (tx - 6, ty + (th + 4) / 2)], fill=accent + (alpha,))
    d.text((tx, ty), text, font=font, fill=accent + (alpha,))


def render(video, scenes, shots, out_path, label="", labels=None):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = n / fps

    f_big = load_font(max(int(h * 0.055), 15))
    f_mid = load_font(max(int(h * 0.042), 12))
    f_small = load_font(max(int(h * 0.032), 10))

    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", out_path],
        stdin=subprocess.PIPE)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        top, bar_h = draw_timeline(frame, scenes, shots, t, total, fps)

        sid, sc = scene_at(scenes, t)
        c = sid % len(PALETTE)
        accent = STROKE[c][::-1]
        fill = PALETTE[c][::-1]

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil, "RGBA")

        # 左上角信息面板
        lines = [
            (f"场景 {sid + 1} / {len(scenes)}", f_big, tuple(accent)),
            (f"{fmt(sc['start'])} - {fmt(sc['end'])}", f_mid, (60, 60, 60)),
            (f"当前 {fmt(t)}   {label}", f_small, (110, 110, 110)),
        ]
        scn = scenario_at(labels, t)
        if scn:
            lines.append((f"测试项：{scn['name']}", f_mid, (30, 30, 30)))
            if scn.get("note"):
                lines.append((scn["note"], f_small, (153, 60, 29)))
        pad = 8
        widths, heights = [], []
        for txt, fnt, _ in lines:
            bb = d.textbbox((0, 0), txt, font=fnt)
            widths.append(bb[2] - bb[0])
            heights.append(bb[3] - bb[1])
        bw = max(widths) + pad * 2
        bh = sum(heights) + 5 * (len(lines) - 1) + pad * 2
        x, y = int(w * 0.025), int(h * 0.05)
        d.rounded_rectangle([x, y, x + bw, y + bh], radius=6,
                            fill=(255, 255, 255, 225), outline=accent + (255,),
                            width=2)
        cy = y + pad
        for (txt, fnt, col), lh in zip(lines, heights):
            d.text((x + pad, cy), txt, font=fnt, fill=col)
            cy += lh + 5

        # 场景切换横幅：每段开头 1 秒内显示，1 秒内淡出
        since = t - sc["start"]
        if since < 1.0:
            alpha = int(255 * min(1.0, (1.0 - since) * 2.5))
            draw_banner(d, w, h, f"场景 {sid + 1} 开始", f_big,
                        fill, accent, alpha)

        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # 场景边框 + 切换瞬间的整屏闪色（画在最上层，绝对看得见）
        base = max(int(h * 0.022), 5)
        since = t - sc["start"]
        if since < 0.7:
            k = 1.0 - since / 0.7
            tint = np.full_like(frame, accent, dtype=np.uint8)
            frame = cv2.addWeighted(frame, 1 - 0.28 * k, tint, 0.28 * k, 0)
            draw_frame_border(frame, accent, base * 2)
        else:
            draw_frame_border(frame, accent, base)

        proc.stdin.write(frame.tobytes())
        idx += 1
        if idx % 100 == 0:
            print(f"  已渲染 {idx}/{n} 帧", flush=True)

    cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"完成：{out_path}（{n} 帧 / {total:.2f} 秒）")


def normalize_scenes(raw):
    """兼容两种分段结构：
      · 项目内统一格式  {"id", "start", "end", "duration", "start_frame", "end_frame"}
      · 旧格式          {"scene_id", "start", "end", "duration", "n_shots", "shot_range"}
    """
    out = []
    for i, s in enumerate(raw):
        out.append({
            "scene_id": s.get("scene_id", s.get("id", i)),
            "start": float(s["start"]),
            "end": float(s["end"]),
            "duration": float(s.get("duration", float(s["end"]) - float(s["start"]))),
            "n_shots": int(s.get("n_shots", 1)),
            "shot_range": s.get("shot_range", [0, 0]),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--scenes", default=None, help="分段结果 json 文件路径")
    ap.add_argument("--scenes-json", default=None,
                    help="直接传入分段结果的 JSON 字符串（优先于 --scenes）")
    ap.add_argument("--shots", default=None, help="镜头列表 json（可选，用于画镜头刻度）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="", help="左上角显示的方法名")
    ap.add_argument("--labels", default=None,
                    help="测试集真值 json（含 scenarios），会在画面上标出当前测试项名称")
    args = ap.parse_args()

    if args.scenes_json:
        raw = json.loads(args.scenes_json)
        if isinstance(raw, dict):
            raw = raw.get("segments", [])
        scenes = normalize_scenes(raw)
        shots = []
    else:
        if not args.scenes:
            raise SystemExit("需要 --scenes 或 --scenes-json")
        raw = json.load(open(args.scenes, encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("segments", [])
        scenes = normalize_scenes(raw)
        shots_path = args.shots or os.path.join(os.path.dirname(args.scenes),
                                                "shots.json")
        shots = json.load(open(shots_path, encoding="utf-8")) \
            if os.path.exists(shots_path) else []
        if shots and "shot_id" not in shots[0]:
            shots = [{"shot_id": i, "start": s["start"], "end": s["end"]}
                     for i, s in enumerate(shots)]

    labels = None
    if args.labels and os.path.exists(args.labels):
        labels = json.load(open(args.labels, encoding="utf-8")).get("scenarios")
    render(args.video, scenes, shots, args.out, args.label, labels)


if __name__ == "__main__":
    main()
