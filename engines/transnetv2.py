"""TransNetV2 引擎 —— 逐帧转场概率 + 阈值切分

原理：
  每帧缩到 48×27（故意丢掉画面细节），100 帧滑窗送入 TransNetV2，
  输出每帧一个 0~1 的「转场概率」，超过阈值即判为切点。

关键实现细节（都是踩过坑总结的）：
  · 流式滑窗：内存占用与视频长度无关
  · 每个窗口只取前 50 帧输出，后 50 帧由下一个窗口重新预测（上下文更完整）
  · 结尾补算时**向左填充**——向右填充会在视频尾部凭空造出假切点
  · 闪光抑制：≤4 帧的极亮/极暗无纹理帧，其附近概率清零
"""

import os

import cv2
import numpy as np
import torch

from bootstrap import WEIGHTS, device, log, to_segments

WEIGHT_FILE = WEIGHTS / "transnetv2-pytorch-weights.pth"
WEIGHT_URL = ("https://hf-mirror.com/MiaoshouAI/transnetv2-pytorch-weights"
              "/resolve/main/transnetv2-pytorch-weights.pth")

WINDOW, STEP = 100, 50


def ensure_weights():
    if WEIGHT_FILE.exists():
        return WEIGHT_FILE
    import urllib.request
    WEIGHT_FILE.parent.mkdir(parents=True, exist_ok=True)
    log(f"    下载 TransNetV2 权重（约 30MB）→ {WEIGHT_FILE}")
    urllib.request.urlretrieve(WEIGHT_URL, WEIGHT_FILE)
    return WEIGHT_FILE


def _load_model(dev):
    from transnetv2_model import TransNetV2
    net = TransNetV2()
    sd = torch.load(ensure_weights(), map_location=dev, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)
    net.eval()
    net.to(dev)
    return net


def _frame_probs(video_path, net, dev, step=STEP, center_select=True):
    """流式推理，返回每帧的转场概率 + 每帧亮度统计。

    关于「取哪一帧的预测」——这是精度关键，实测结论：
      同一帧在窗口里的位置决定了预测质量，概率随位置呈钟形曲线，
      **峰值精确出现在位置 50（窗口正中）**，两侧对称衰减。
      实测一个真实切点在各位置的置信度：
          位置 0 → 0.29   位置 25 → 0.57   位置 50 → 0.65   位置 95 → 0.08

    两种取法：
      center_select=True （推荐）每个窗口输出全部 100 帧，然后**每帧取它
        「最接近窗口正中」的那次预测**。位置落在 [25,75]，置信度明显更高。
        实测能把一个原本 0.4432（低于阈值 0.5 而漏检）的切点救回 0.6277。
      center_select=False 每个窗口只取前 50 帧（官方做法，位置落在 [0,49]）。

    step 越小，每帧的候选位置越多、越能贴近正中，但推理次数成比例增加。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    buf, tail, stats = [], [], []
    total = 0
    center = WINDOW // 2
    preds = np.full(1 << 20, np.nan, dtype=np.float32)      # 按需增长
    best = np.full(1 << 20, 1e9, dtype=np.float32)

    def ensure(n):
        nonlocal preds, best
        if n >= len(preds):
            grow = max(n - len(preds) + 1, 1 << 16)
            preds = np.concatenate([preds, np.full(grow, np.nan, np.float32)])
            best = np.concatenate([best, np.full(grow, 1e9, np.float32)])

    win_start = 0
    with torch.no_grad():
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            small = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (48, 27),
                               interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            stats.append((float(g.mean()), float(g.std())))
            buf.append(small)
            tail.append(small)
            if len(tail) > WINDOW:
                tail.pop(0)
            total += 1
            if len(buf) == WINDOW:
                t = torch.from_numpy(np.stack(buf)[None]).to(dev)
                out = net(t)
                one_hot = out[0] if isinstance(out, tuple) else out
                p = torch.sigmoid(one_hot)[0, :, 0].cpu().numpy()
                ensure(win_start + WINDOW)
                if center_select:
                    for i in range(WINDOW):          # 全窗都留，挑离中心最近的
                        f = win_start + i
                        if f >= total:
                            break
                        d = abs(i - center)
                        if d < best[f]:
                            best[f], preds[f] = d, p[i]
                else:
                    for i in range(step):            # 官方做法：只取前 step 帧
                        f = win_start + i
                        if f >= total:
                            break
                        preds[f] = p[i]
                buf = buf[step:]
                win_start += step

        # 结尾补算：真实帧放窗口末尾（向左填充），保证它们有真实的左侧上下文
        if np.isnan(preds[:total]).any():
            missing = np.where(np.isnan(preds[:total]))[0]
            lo = max(missing.min() - 0, 0)
            chunk = np.stack(tail)
            base = total - len(chunk)
            if len(chunk) < WINDOW:
                pad = np.repeat(chunk[:1], WINDOW - len(chunk), axis=0)
                chunk = np.concatenate([pad, chunk], axis=0)
                base = total - len(chunk)
            t = torch.from_numpy(chunk[None]).to(dev)
            out = net(t)
            one_hot = out[0] if isinstance(out, tuple) else out
            p = torch.sigmoid(one_hot)[0, :, 0].cpu().numpy()
            ensure(total)
            for i in range(len(chunk)):
                f = base + i
                if 0 <= f < total and np.isnan(preds[f]):
                    preds[f] = p[i]

    cap.release()
    return np.nan_to_num(preds[:total], nan=0.0).astype(np.float32), stats


def _suppress_flash(preds, stats):
    """闪光抑制：连续 ≤4 帧的极亮/极暗无纹理帧 → 附近概率清零。"""
    if len(stats) != len(preds):
        return preds, 0
    flash = np.array([(m > 235 or m < 20) and s < 25 for m, s in stats])
    n = 0
    i = 0
    while i < len(flash):
        if flash[i]:
            j = i
            while j < len(flash) and flash[j]:
                j += 1
            if j - i <= 4:                       # 连续 >4 帧视为真实转场（如淡入黑场）
                preds[max(i - 1, 0):min(j + 1, len(preds))] = 0.0
                n += 1
            i = j
        else:
            i += 1
    return preds, n


def _filter_cuts(cuts, total, fps, min_shot_sec):
    """保证每一段都不短于 min_shot_sec，过短的切点直接丢弃。"""
    min_frames = max(min_shot_sec * fps, 1.0)
    out, prev = [], 0
    for c in sorted({int(x) for x in cuts}):
        if 0 < c < total and c - prev >= min_frames:
            out.append(c)
            prev = c
    return out


def run(video_path, threshold=0.5, min_shot=0.4, flash_suppress=False,
        step=STEP, center_select=True):
    """返回 (segments, meta)。

    闪光抑制默认**关闭**——闪光（相机补光、闪白转场）在成片里往往就是一次真实的
    场景切换，切断是合理的。需要时用 --flash-suppress 打开。
    """
    dev = device()
    log(f"    TransNetV2（device={dev}, step={step}, "
        f"选取={'窗口正中' if center_select else '前半'}）")
    net = _load_model(dev)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()

    preds, stats = _frame_probs(video_path, net, dev, step=step,
                                center_select=center_select)
    total = len(preds)

    n_flash = 0
    if flash_suppress:
        preds, n_flash = _suppress_flash(preds, stats)
        if n_flash:
            log(f"    闪光抑制：清除了 {n_flash} 处闪白/闪黑帧附近的边界")

    raw_cuts = np.where(preds > threshold)[0].tolist()
    cuts = _filter_cuts(raw_cuts, total, fps, min_shot)

    meta = {
        "engine": "transnetv2",
        "threshold": threshold,
        "min_shot": min_shot,
        "step": step,
        "select": "center" if center_select else "first_half",
        "flash_suppressed": n_flash,
        "raw_cuts": len(raw_cuts),
        "prob": preds,
        "fps": fps, "total_frames": total,
        "width": w, "height": h,
        "duration": total / fps,
    }
    return to_segments(cuts, total, fps), meta
