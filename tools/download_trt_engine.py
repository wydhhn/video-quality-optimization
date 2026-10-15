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
RELEASE = ("https://github.com/Liqiu-Chen/video-enhance-service"
           "/releases/download/v1.1.0-engines")

# ★ 引擎体积超 GitHub 的 100 MB 单文件上限，所以放在另一个仓库的 Release 上
#   （video-enhance-service 是我们的开发仓库；二进制和源码分开是常见做法）。
#
# (归档文件名, 落盘文件名, 字节数, 说明)
#   两个名字不同是因为 **GitHub 会把资产名里的 `+` 替换成 `.`** ——
#   上传时叫 rife_..._warp+resize.trt，下载 URL 里却必须是 warp.resize.trt。
MANIFEST = [
    (
        "vsr_tf32_t8_836x480_x4.trt", "vsr_tf32_t8_836x480_x4.trt",
        141037172,
        "VSR 主引擎 · TensorRT TF32 · t=8 · 输入 836x480 → ×4 · 78.54 dB / 1.69×",
    ),
    (
        "vsr_tf32_t2_836x480_x4.trt", "vsr_tf32_t2_836x480_x4.trt",
        162586804,
        "VSR 引擎（t=2）· TF32 · 80.28 dB / 1.86× · 短片段或快速验证用",
    ),
    (
        "rife_3360x1920_warp.resize.trt",            # 下载 URL 里的名字
        "rife_3360x1920_warp+resize.trt",            # 落盘用的名字（代码按这个找）
        354141916,
        "RIFE 4x 路径引擎 · GridSample/Resize 留 fp32 · 49.88 dB",
    ),
]

def _human(n: int) -> str:
    return f"{n / 2**20:.1f} MB"


def _opener(proxy: str | None):
    """构造 urlopen 用的 opener。

    ★ 为什么要自己构造、而不是直接用 `urllib.request.urlopen`：
    后者会**自动读取环境变量里的 `http_proxy` / `https_proxy`**。
    如果机器上恰好有个到不了 GitHub 的本地代理（很常见：各种软件装的
    内置代理会往环境变量里塞 `127.0.0.1:xxxxx`），下载就会莫名其妙地
    `502 Bad Gateway`，而报错完全看不出是代理问题。

    所以这里的策略是**显式控制**：
      · 传了 --proxy 或设了 FI_PROXY → 只走那个代理
      · 没传 → **不走任何代理**（直连）
    想用系统代理的话显式设置 FI_PROXY 即可。
    """
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        print(f"      代理：{proxy}")
    else:
        handler = urllib.request.ProxyHandler({})      # 空字典 = 显式禁用代理
    return urllib.request.build_opener(handler)


def download(url: str, dest: Path, expect: int, proxy: str | None = None) -> bool:
    """url 里的文件名可能与 dest 不同（GitHub 会把 + 换成 .）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"      源：{url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with _opener(proxy).open(req, timeout=60) as r, open(tmp, "wb") as f:
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
    ap.add_argument("--proxy", default=os.environ.get("FI_PROXY", ""),
                    help="走哪个代理（如 http://127.0.0.1:7897）。"
                         "不传则**直连**；也可用环境变量 FI_PROXY。"
                         "注意：不会使用环境变量里的 http_proxy —— 那通常是别的"
                         "软件塞进来的、到不了 GitHub 的内置代理")
    args = ap.parse_args()

    out_dir = Path(args.dir)
    print(f"引擎目录：{out_dir}")
    print()

    if args.list:
        for urlname, localname, size, desc in MANIFEST:
            mark = "✓ 已有" if (out_dir / localname).exists() else "  未下载"
            print(f"  [{mark}] {localname}  ({_human(size)})")
            print(f"           {desc}")
        return 0

    todo = [(u, l, s, d) for u, l, s, d in MANIFEST
            if args.name is None or l == args.name or u == args.name]
    if not todo:
        print(f"❌ MANIFEST 里没有 {args.name}")
        return 1

    ok = True
    for urlname, localname, size, desc in todo:
        dest = out_dir / localname
        if dest.exists() and (not size or dest.stat().st_size == size):
            print(f"  ⏭  {localname} 已存在（{_human(dest.stat().st_size)}），跳过")
            continue
        print(f"  [下载] {localname}  ({_human(size)})")
        print(f"      {desc}")
        if not download(f"{RELEASE}/{urlname}", dest, size, args.proxy or None):
            ok = False

    print()
    if ok:
        print("  ✅ 完成。用 pipeline/sr_trt.py 的 SrTrtEngine 即可加载：")
        print("       from pipeline.sr_trt import SrTrtEngine")
        print("       sr = SrTrtEngine(chunk=8)     # 自动找到 t=8 的引擎")
    else:
        print("  ⚠️ 有下载失败。常见原因：")
        print("       1) 网络到不了 GitHub —— 试试指定代理：")
        print("            python tools/download_trt_engine.py --proxy http://127.0.0.1:7897")
        print("          （端口按你自己的代理软件填，Clash 系通常是 7897）")
        print("       2) 若是国内网络且不想配代理，可改用镜像或自己构建引擎：")
        print("            python tools/trt/export_vsr_onnx.py --t 8 --size 836 480")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
