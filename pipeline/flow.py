"""串联编排 —— 分段 → 超分 → 插帧 → 编码。

线程模型（沿用原项目的设计）：
    读取线程(ffmpeg 解码) ──▶ queue_in ──▶ 主线程(超分+插帧) ──▶ queue_out ──▶ 写线程(ffmpeg 编码)
解码/编码是 CPU+IO 密集、推理是 GPU 密集，三者重叠可以把总时间压到
`max(IO, GPU)` 而不是相加。

帧数约定
--------
* `sr`     : N → N   （帧率、时长都不变）
* `fi`     : N → 2N  （帧率翻倍=原速；或帧率不变=慢动作）
* `sr+fi`  : N → 2N

插帧时**跨切点的帧对不调用 RIFE**，用重复帧代替 —— 输出帧数仍然是 2N，
与切点数量无关（见 `pipeline.shotaware`）。
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time

import numpy as np

from bootstrap import log
from pipeline import shotaware


class Progress:
    def __init__(self):
        self.stage = "init"
        self.done = 0
        self.total = 0
        self.message = ""
        self.t0 = time.time()
        self.marks: dict[str, float] = {}

    def as_dict(self):
        el = time.time() - self.t0
        return {
            "stage": self.stage, "done": self.done, "total": self.total,
            "pct": round(self.done / self.total * 100, 1) if self.total else 0.0,
            "message": self.message, "elapsed_s": round(el, 1),
            "stages": self.marks,
        }


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def count_frames(ff: str, path: str) -> int | None:
    import re
    p = subprocess.run([ff, "-hide_banner", "-i", path, "-map", "0:v:0",
                        "-c", "copy", "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")
    m = re.findall(r"frame=\s*(\d+)", p.stderr)
    return int(m[-1]) if m else None


def probe(path: str) -> dict:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    # 尺寸取偶数（yuv420p 要求）
    info["width"] -= info["width"] % 2
    info["height"] -= info["height"] % 2
    return info


class Flow:
    """一次任务的执行体。"""

    def __init__(self, sr_engine=None, fi_engine=None, *,
                 micro_batch: int = 12, queue_in: int = 6,
                 queue_out: int = 24, verbose: bool = True):
        self.sr = sr_engine
        self.fi = fi_engine
        self.micro_batch = int(micro_batch)
        self.qin_size = int(queue_in)
        self.qout_size = int(queue_out)
        self.verbose = verbose

    def _log(self, *a):
        if self.verbose:
            log("  [flow] " + " ".join(str(x) for x in a))

    @staticmethod
    def _probe_ok(path: str) -> bool:
        """产物能不能被正常解析（编码器没写完时缺 moov atom，文件照样有几百 MB）。"""
        if not os.path.exists(path) or os.path.getsize(path) < 1024:
            return False
        p = subprocess.run([_ffmpeg(), "-hide_banner", "-i", path],
                           capture_output=True, text=True, errors="replace")
        return "Duration:" in p.stderr and "Video:" in p.stderr

    # ------------------------------------------------------------------ #
    def run(self, src: str, out: str, *, cut_frames=None, shot_aware: bool = True,
            do_sr: bool = True, do_fi: bool = True, realtime: bool = True,
            crf: int = 16, preset: str = "medium",
            progress: Progress | None = None) -> dict:
        if not do_sr and not do_fi:
            raise ValueError("至少要选超分或插帧之一")

        ff = _ffmpeg()
        pg = progress or Progress()
        info = probe(src)
        W0, H0, fps_src = info["width"], info["height"], info["fps"]
        n_src = count_frames(ff, src) or info["frames"]
        cuts = shotaware.normalize_cuts(cut_frames or [], n_src) if shot_aware else []

        factor = 2 if do_fi else 1
        fps_out = fps_src * (factor if (do_fi and realtime) else 1)
        n_out_expect = n_src * factor
        W1 = W0 * (self.sr.scale if do_sr else 1)
        H1 = H0 * (self.sr.scale if do_sr else 1)

        pg.total = n_src
        pg.stage = "decode"
        pg.message = (f"{W0}x{H0}@{fps_src:g}fps {n_src}帧 → "
                      f"{W1}x{H1}@{fps_out:g}fps {n_out_expect}帧"
                      f"（{'超分' if do_sr else ''}{'+' if do_sr and do_fi else ''}"
                      f"{'插帧' if do_fi else ''}"
                      f"{'，分段感知' if shot_aware and cuts else ''}）")

        qin: queue.Queue = queue.Queue(maxsize=self.qin_size)
        qout: queue.Queue = queue.Queue(maxsize=self.qout_size)
        errors: list = []
        dead = threading.Event()
        # reader 结束标志。
        # 只靠「往队列里投一个 None 哨兵」是不够的：哨兵被消费掉之后，
        # 主循环下一次 qin.get() 会永久阻塞（reader 已经退出，不会再有数据）。
        # 实测就卡在这里 —— 处理完最后一批后进程再也不动。
        reader_done = threading.Event()

        def _put(q, item, timeout=1.0):
            """带超时的入队。

            「队列满 + 消费者已退出」是死锁的经典组合 —— 生产者会永远卡在 put 上。
            超时后检查 dead 标志，及时抛错而不是无限等。
            """
            while True:
                try:
                    q.put(item, timeout=timeout)
                    return
                except queue.Full:
                    if dead.is_set():
                        raise RuntimeError("下游已停止（编码器/解码器异常退出），流水线中止")

        # ---- 解码 ----
        dec = subprocess.Popen(
            [ff, "-v", "error", "-i", src, "-fps_mode", "passthrough",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-vf", f"crop={W0}:{H0}:0:0", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 22)

        def drain(proc, tag):
            """必须把 stderr 读干净：管道缓冲区写满会阻塞 ffmpeg 自己。"""
            try:
                for line in iter(proc.stderr.readline, b""):
                    t = line.decode("utf-8", "replace").strip()
                    if t:
                        errors.append(f"{tag}: {t}")
            except Exception:                                   # noqa: BLE001
                pass

        def reader():
            fsz = W0 * H0 * 3
            try:
                while not dead.is_set():
                    buf = dec.stdout.read(fsz)
                    if len(buf) < fsz:
                        break
                    _put(qin, np.frombuffer(buf, np.uint8).reshape(H0, W0, 3))
            except Exception as e:                              # noqa: BLE001
                errors.append(f"decode: {e}")
            finally:
                reader_done.set()
                try:
                    qin.put_nowait(None)
                except queue.Full:
                    pass

        # ---- 编码 ----
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        enc_cmd = [ff, "-y", "-v", "error",
                   "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{W1}x{H1}", "-framerate", f"{fps_out}", "-i", "-",
                   "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                   "-pix_fmt", "yuv420p",
                   "-frames:v", str(n_out_expect),   # 按帧数封顶（比 -t 按秒精确）
                   "-movflags", "+faststart", out]
        enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=1 << 24)
        written = [0]

        def writer():
            try:
                while True:
                    fr = qout.get()
                    if fr is None:
                        break
                    enc.stdin.write(fr.tobytes())
                    written[0] += 1
            except Exception as e:                              # noqa: BLE001
                errors.append(f"encode: {type(e).__name__}: {e}")
                dead.set()
            finally:
                try:
                    enc.stdin.close()
                except Exception:                               # noqa: BLE001
                    pass

        threads = [
            threading.Thread(target=reader, daemon=True),
            threading.Thread(target=writer, daemon=True),
            threading.Thread(target=drain, args=(dec, "decode-stderr"), daemon=True),
            threading.Thread(target=drain, args=(enc, "encode-stderr"), daemon=True),
        ]
        for t in threads:
            t.start()

        # 预热（把 cudnn 算法选型的一次性开销挪出正式计时）
        if do_sr:
            self.sr.warmup(H0, W0)

        # ---- 主循环：微批 ----
        carry = None                 # 上一批最后一帧（供跨批插帧）
        carry_g0 = None              # 它的全局帧号
        g_read = 0                   # 下一批要读的源帧全局号
        produced = 0
        last_frame = None
        try:
            pg.stage = "sr" if do_sr else "fi"
            while True:
                batch = []
                while len(batch) < self.micro_batch:
                    try:
                        item = qin.get(timeout=0.5)
                    except queue.Empty:
                        # reader 已退出且队列已空 → 真的没有更多数据了
                        if reader_done.is_set() and qin.empty():
                            break
                        continue
                    if item is None:
                        break
                    batch.append(item)
                if not batch:
                    break
                n = len(batch)
                g0 = g_read
                src_arr = np.stack(batch)

                t0 = time.time()
                up = (self.sr.upscale(src_arr, cuts, shot_aware) if do_sr
                      else src_arr)
                if do_sr:
                    pg.marks["sr_s"] = round(pg.marks.get("sr_s", 0) + time.time() - t0, 1)

                if do_fi:
                    pg.stage = "fi"
                    if carry is None:
                        seq, seq_g0, lead = up, g0, 0
                    else:
                        seq = np.concatenate([carry[None], up], 0)
                        seq_g0, lead = carry_g0, 1
                    t1 = time.time()
                    out_seq = self.fi.interpolate(seq, cuts, shot_aware,
                                                  g0=seq_g0, total=n_src,
                                                  lead_pairs=lead)
                    pg.marks["fi_s"] = round(pg.marks.get("fi_s", 0) + time.time() - t1, 1)
                    for fr in out_seq:
                        _put(qout, fr)
                    produced += len(out_seq)
                    last_frame = out_seq[-1]
                    carry, carry_g0 = up[-1].copy(), g0 + n - 1
                else:
                    for fr in up:
                        _put(qout, fr)
                    produced += n
                    last_frame = up[-1]

                g_read += n
                # 进度放在**真正交出去之后**再记（否则下游卡住时页面会显示 100%）
                pg.done = min(pg.done + n, n_src)
                del batch, src_arr, up
        finally:
            # 插帧模式补最后一帧，凑满 2N
            if do_fi and last_frame is not None and produced < n_out_expect:
                try:
                    _put(qout, last_frame)
                    produced += 1
                except Exception:                               # noqa: BLE001
                    pass
            try:
                qout.put_nowait(None)
            except queue.Full:
                while not qout.empty():
                    try:
                        qout.get_nowait()
                    except queue.Empty:
                        break
                try:
                    qout.put_nowait(None)
                except queue.Full:
                    pass
            threads[1].join(timeout=600)
            threads[0].join(timeout=10)
            # 关掉 stdin 后编码器还要 flush 尾部、写 moov box —— 直接 kill 会把 mp4 截成 0 字节
            for p in (enc, dec):
                try:
                    p.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    self._log(f"进程 {p.pid} 超时未退出，强制结束")
                    p.kill()
                    p.wait()

        # ---- 结果校验：宁可报失败，也不交付打不开的文件 ----
        if enc.returncode not in (0, None):
            errors.append(f"ffmpeg 编码器退出码 {enc.returncode}")
        if written[0] == 0:
            errors.append("没有任何帧写入编码器")
        if not self._probe_ok(out):
            errors.append("产物无法解析（缺少 moov atom，说明编码没写完）")
        if errors:
            if os.path.exists(out):
                try:
                    os.remove(out)
                except OSError:
                    pass
            raise RuntimeError(f"流水线失败: {'; '.join(errors[-4:])}")

        pg.stage = "done"
        pg.done = pg.total
        pg.marks["total_s"] = round(time.time() - pg.t0, 1)
        return {
            "output": out,
            "size_mb": round(os.path.getsize(out) / 1048576, 1),
            "frames_in": n_src, "frames_out": written[0],
            "fps_src": fps_src, "fps_out": fps_out,
            "resolution": [W1, H1],
            "mode": ("sr+fi" if do_sr and do_fi else ("sr" if do_sr else "fi")),
            "shot_aware": bool(shot_aware and cuts),
            "n_cuts": len(cuts), "cuts": cuts,
            "realtime": realtime,
            "elapsed_s": round(time.time() - pg.t0, 1),
            "stages_s": pg.marks,
        }
