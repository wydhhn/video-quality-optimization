"""下载多条 AI 生成视频，拼成一条 1080p、超过 1 分钟的视频，并记录拼接点真值。

为什么要拼：公开渠道的 AI 生成视频（Sora / Kling / Hunyuan / Pika / Wan 等）
基本都只有 5~10 秒，没有单条 1 分钟以上的 1080p 成品。拼接之后：
  - 分辨率统一到 1920×1080
  - 总时长 > 60 秒
  - 拼接点就是**已知真值的硬切点**，可以直接用来验证分段算法

用法：
    python make_ai_test_video.py --clips 18 --out ai_video_1080p.mp4
"""

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

W, H = 1920, 1080
FPS = 30
HF = "https://hf-mirror.com/datasets"
KLING = f"{HF}/bitmind/klingai-videos/resolve/main/batch_000/video_{{:06d}}.mp4"
RAPIDATA = (f"{HF}/Rapidata/text-2-video-human-preferences-wan2.1"
            "/resolve/main/videos/")
RAPIDATA_NAMES = ["0000_sora_0.mp4", "0000_sora_1.mp4", "0000_hunyuan_1724.mp4",
                  "0000_hunyuan_421.mp4", "0000_pika_1657819738.mp4",
                  "0000_pika_1980618835.mp4", "0000_ray2_1.mp4", "0000_ray2_2.mp4",
                  "0000_alpha_357741999.mp4"]
CACHE = "_tmp/aivid/cache"


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 10000:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dst)
        return dst
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"    下载失败 {os.path.basename(dst)}: {e}")
        return None


def probe(path):
    c = cv2.VideoCapture(path)
    if not c.isOpened():
        return None
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = c.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(c.get(3)), int(c.get(4))
    c.release()
    if n < 5 or w < 320:
        return None
    return {"path": path, "frames": n, "fps": fps,
            "duration": n / fps, "w": w, "h": h}


def fit_1080(frame):
    """等比缩放到 1920×1080 画布并居中填充黑边（不裁掉内容）。"""
    h, w = frame.shape[:2]
    s = min(W / w, H / h)
    nw, nh = max(int(w * s), 1), max(int(h * s), 1)
    r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=18, help="最终使用多少条片段")
    ap.add_argument("--pool", type=int, default=40, help="先下载多少条候选")
    ap.add_argument("--out", default="ai_video_1080p.mp4")
    ap.add_argument("--gt", default=None)
    args = ap.parse_args()

    # 1) 下载候选：Kling 批量 + Rapidata 多模型
    tasks = [(KLING.format(i), os.path.join(CACHE, f"kling_{i:06d}.mp4"))
             for i in range(args.pool)]
    tasks += [(RAPIDATA + n, os.path.join(CACHE, n)) for n in RAPIDATA_NAMES]

    print(f"[1/4] 下载 {len(tasks)} 条候选 AI 视频 …")
    with ThreadPoolExecutor(10) as ex:
        list(ex.map(lambda t: fetch(*t), tasks))

    files = [p for _, p in tasks if os.path.exists(p)]
    print(f"      实际拿到 {len(files)} 条")

    # 2) 探测并筛选
    infos = [x for x in (probe(p) for p in files) if x]
    infos.sort(key=lambda x: (-x["w"] * x["h"], -x["duration"]))
    print(f"[2/4] 分辨率分布（按面积降序前 10）：")
    for x in infos[:10]:
        print(f"      {x['w']}x{x['h']}  {x['duration']:.1f}s  "
              f"{os.path.basename(x['path'])}")

    # 优先选高分辨率、时长足够的；同一分辨率下按文件名稳定排序
    picked, total = [], 0.0
    for x in infos:
        if x["duration"] < 2.0:
            continue
        picked.append(x)
        total += x["duration"]
        if len(picked) >= args.clips and total >= 62:
            break
    print(f"[3/4] 选用 {len(picked)} 条，预计总时长 {total:.1f}s")

    # 3) 统一到 1080p 并拼接
    print(f"[4/4] 渲染 {W}x{H} 输出 …")
    import imageio_ffmpeg
    import subprocess
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", args.out],
        stdin=subprocess.PIPE)

    cuts, t, n_written = [], 0.0, 0
    for x in picked:
        cap = cv2.VideoCapture(x["path"])
        n_use = int(round(x["duration"] * FPS))
        for _ in range(n_use):
            ok, fr = cap.read()
            if not ok:
                break
            proc.stdin.write(fit_1080(fr).tobytes())
            n_written += 1
        cap.release()
        t = n_written / FPS
        cuts.append(round(t, 2))
        print(f"      + {os.path.basename(x['path']):<28} → 累计 {t:6.2f}s", flush=True)
    proc.stdin.close()
    proc.wait()

    # 4) 真值：拼接点（最后一个不算，它是结尾）
    gt = {
        "duration": round(t, 2), "fps": FPS, "size": [W, H],
        "source": "多条公开 AI 生成视频拼接（Sora/Kling/Hunyuan/Pika/Wan 等）",
        "expected_cuts": cuts[:-1],
        "n_clips": len(picked),
        "clips": [{"name": os.path.basename(x["path"]),
                   "src_size": f"{x['w']}x{x['h']}",
                   "duration": round(x["duration"], 2)} for x in picked],
    }
    dst = args.gt or os.path.splitext(args.out)[0] + "_gt.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)

    print(f"\n完成：{args.out}")
    print(f"  {W}x{H} / {t:.2f} 秒 / {n_written} 帧 / {len(picked)} 条 AI 片段拼接")
    print(f"  真值切点（拼接点）{len(gt['expected_cuts'])} 个：{gt['expected_cuts']}")
    print(f"  已保存真值：{dst}")


if __name__ == "__main__":
    main()
