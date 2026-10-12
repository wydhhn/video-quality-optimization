#!/usr/bin/env python
"""下载全部模型权重到 weights/。

用法：
    python tools/download_weights.py            # 下载缺失的
    python tools/download_weights.py --check    # 只检查，不下载
    python tools/download_weights.py --force    # 强制重新下载

共 5 个文件、约 275 MB。所有地址都实测可用（中国大陆网络环境）。
"""

import argparse
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"

# (文件名, 字节数, [候选 URL...])
MANIFEST = [
    (
        "OmniShotCut_ckpt.pth",
        211261519,
        ["https://hf-mirror.com/uva-cv-lab/OmniShotCut_v1.5/resolve/main/OmniShotCut_ckpt.pth"],
    ),
    (
        "transnetv2-pytorch-weights.pth",
        30510466,
        ["https://hf-mirror.com/MiaoshouAI/transnetv2-pytorch-weights"
         "/resolve/main/transnetv2-pytorch-weights.pth"],
    ),
    (
        "basicvsr_plusplus_reds4.pth",
        29349005,
        ["https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus"
         "/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"],
    ),
    (
        "spynet.pth",
        5770988,
        ["https://download.openmmlab.com/mmediting/restorers/basicvsr"
         "/spynet_20210409-c6c1bd09.pth"],
    ),
    (
        "flownet.pkl",
        12186817,
        ["https://modelscope.cn/models/AI-ModelScope/RIFE/resolve/master/flownet.pkl",
         "https://www.modelscope.cn/models/AI-ModelScope/RIFE/resolve/master/flownet.pkl"],
    ),
]


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dest: Path, expect: int) -> bool:
    """下载到 dest.part，成功后原子改名。返回是否成功。"""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or expect)
            got, t0, last = 0, time.time(), 0.0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                now = time.time()
                if now - last > 0.3:
                    last = now
                    pct = got / total * 100 if total else 0
                    speed = got / max(now - t0, 1e-6)
                    sys.stdout.write(
                        f"\r      {pct:5.1f}%  {human(got)}/{human(total)}"
                        f"  {human(speed)}/s   ")
                    sys.stdout.flush()
        sys.stdout.write("\r" + " " * 60 + "\r")
    except Exception as e:                                      # noqa: BLE001
        sys.stdout.write("\r" + " " * 60 + "\r")
        print(f"      失败：{type(e).__name__}: {str(e)[:90]}")
        tmp.unlink(missing_ok=True)
        return False

    if tmp.stat().st_size != expect:
        print(f"      大小不符：得到 {tmp.stat().st_size}，期望 {expect}")
        tmp.unlink(missing_ok=True)
        return False
    shutil.move(str(tmp), str(dest))
    return True


def main():
    ap = argparse.ArgumentParser(description="下载全部模型权重")
    ap.add_argument("--check", action="store_true", help="只检查，不下载")
    ap.add_argument("--force", action="store_true", help="强制重新下载")
    args = ap.parse_args()

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    print(f"权重目录：{WEIGHTS}\n")

    ok = miss = 0
    for name, size, urls in MANIFEST:
        dest = WEIGHTS / name
        have = dest.exists() and dest.stat().st_size == size

        if have and not args.force:
            print(f"  [已有] {name}  ({human(size)})")
            ok += 1
            continue
        if args.check:
            print(f"  [缺失] {name}  (需 {human(size)})")
            miss += 1
            continue

        print(f"  [下载] {name}  ({human(size)})")
        done = False
        for u in urls:
            print(f"      源：{u}")
            if download(u, dest, size):
                print(f"      ✓ 完成")
                ok += 1
                done = True
                break
        if not done:
            miss += 1

    print()
    if miss:
        print(f"完成：{ok} 个就绪，{miss} 个失败。"
              f"失败的可以手动下载后放进 {WEIGHTS}")
        return 1
    print(f"全部 {ok} 个权重就绪，可以开始跑了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
