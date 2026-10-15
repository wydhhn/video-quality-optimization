"""下载 TensorRT 超分引擎。

和 `tools/download_weights.py` 是同一个套路：**大文件不进 git，用脚本下载**
（GitHub 单文件上限 100 MB，而引擎是 135 MB）。

用法：
    python tools/download_trt_engine.py              # 下载推荐的 TF32 引擎
    python tools/download_trt_engine.py --list       # 看有哪些
    python tools/download_trt_engine.py --name vsr_tf32_t8_836x480_x4.trt
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRT_DIR = ROOT / "trt5"

# 引擎发布在 GitHub Release 上（tag → 资产）
RELEASE = "https://github.com/wydhhn/video-quality-optimization/releases/download/v1.1.0-trt"

# (文件名, 字节数, 说明)
MANIFEST = [
    (
        "vsr_tf32_t8_836x480_x4.trt",
        141032847,
        "VSR 主引擎 · TensorRT TF32 · t=8 · 输入 836x480 → ×4 输出 · 78.54 dB / 1.69×",
    ),
]


def _human(n: int) -> str:
    return f"{n / 2**20:.1f} MB"


def download(url: str, dest: Path, expect: int) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"      源：{url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or expect)
            got, last = 0, 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                pct = got * 100 // max(total, 1)
                if pct >= last + 5:
                    last = pct
                    print(f"      {pct:3d}%  {_human(got)}/{_human(total)}", flush=True)
    except Exception as e:                                       # noqa: BLE001
        print(f"      ❌ 下载失败：{type(e).__name__}: {e}")
        tmp.unlink(missing_ok=True)
        return False

    if expect and tmp.stat().st_size != expect:
        print(f"      ⚠️ 字节数不符：期望 {expect}，实得 {tmp.stat().st_size}")
        print("         （上游若重新上传过引擎，只需更新 MANIFEST 里的字节数）")
        # 不删：字节数不符也可能只是上游更新了，让调用方自己判断
    tmp.replace(dest)
    print(f"      ✓ 完成  {_human(dest.stat().st_size)}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只列出可下载的项")
    ap.add_argument("--name", help="指定要下载的文件名（默认全部）")
    ap.add_argument("--dir", default=str(TRT_DIR), help="下载到哪个目录")
    args = ap.parse_args()

    out_dir = Path(args.dir)
    print(f"引擎目录：{out_dir}")
    print()

    if args.list:
        for name, size, desc in MANIFEST:
            mark = "✓ 已有" if (out_dir / name).exists() else "  未下载"
            print(f"  [{mark}] {name}  ({_human(size)})")
            print(f"           {desc}")
        return 0

    todo = [(n, s, d) for n, s, d in MANIFEST
            if args.name is None or n == args.name]
    if not todo:
        print(f"❌ MANIFEST 里没有 {args.name}")
        return 1

    ok = True
    for name, size, desc in todo:
        dest = out_dir / name
        if dest.exists() and (not size or dest.stat().st_size == size):
            print(f"  ⏭  {name} 已存在（{_human(dest.stat().st_size)}），跳过")
            continue
        print(f"  [下载] {name}  ({_human(size)})")
        print(f"      {desc}")
        if not download(f"{RELEASE}/{name}", dest, size):
            ok = False

    print()
    if ok:
        print("  ✅ 完成。用 pipeline/sr_trt.py 的 SrTrtEngine 即可加载：")
        print("       from pipeline.sr_trt import SrTrtEngine")
        print("       sr = SrTrtEngine(chunk=8)     # 自动找到 t=8 的引擎")
    else:
        print("  ⚠️ 有下载失败。若 Release 还没建好，可用 tools/trt/ 自己构建：")
        print("       python tools/trt/export_vsr_onnx.py --t 8 --size 836 480")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
