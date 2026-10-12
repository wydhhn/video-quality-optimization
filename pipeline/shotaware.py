"""分段感知的核心逻辑 —— 与模型无关的纯计算，可单独在 CPU 上测试。

为什么需要这一层
----------------
两个模型都会在**镜头切换处**出问题，分段结果正好能治：

* **超分（BasicVSR++）跨切点传播会"串色"**
  它的 forward 会先算相邻帧光流（SPyNet），再沿整段序列做双向时序传播。
  如果切点落在一段内部，SPyNet 会去算「镜头 A 末帧 ↔ 镜头 B 首帧」的光流 ——
  这两帧毫无关系，算出来是垃圾，然后这个垃圾光流被用来 warp 特征并累积传播，
  结果是切点附近拖影、串色。
  → 修法：**把切点作为强制分段边界**，让传播在切点处重置。

* **插帧（RIFE）跨切点插帧会出"鬼影"**
  它对每对相邻帧都生成一个中间帧。如果切点在第 i 和 i+1 帧之间，RIFE 会把
  两个不同场景融合成一帧 —— 典型鬼影。
  → 修法：**跨切点的帧对不调用 RIFE**，用重复帧代替。

帧数守恒
--------
2× 插帧：N 帧 → 1 + 2(N-1) = 2N-1 帧（再补一帧凑成 2N）。
跨切点的对有 C 个，每个少插一帧 → 必须用重复帧补回来，否则：
  · realtime 模式（帧率翻倍、原速播放）→ 视频变短
  · slowmo   模式（帧率不变、时长翻倍）→ 时长变短
本模块用 `plan_interp()` 给出掩码，调用方按掩码决定"插帧还是复制"，
**输出帧数始终是 2N**，与切点数量无关。
"""

from __future__ import annotations


# --------------------------------------------------------------------------
# 镜头划分
# --------------------------------------------------------------------------
def normalize_cuts(cut_frames, total: int) -> list[int]:
    """清洗切点：去重、排序、剔除越界值。"""
    return sorted({int(c) for c in cut_frames if 0 < int(c) < total})


def shot_bounds(cut_frames, total: int) -> list[tuple[int, int]]:
    """切点列表 → 每个镜头的 (start, end)。区间左闭右开。"""
    edges = [0] + normalize_cuts(cut_frames, total) + [total]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


# --------------------------------------------------------------------------
# 超分：分段计划
# --------------------------------------------------------------------------
def plan_chunks(total: int, cut_frames, chunk: int,
                overlap: int) -> list[dict]:
    """规划超分的分段。

    规则：
      1. **切点处强制断开** —— 传播在切点重置，不跨镜头
      2. 镜头内部按 `chunk` 继续切 —— 控制显存
      3. 每段的上下文帧（喂给模型的最左边几帧）**也不跨切点**

    返回 `[{"read": (lo, hi), "valid": (start, end)}, ...]`
      read  —— 实际喂进模型的帧区间 [lo, hi)，含上下文
      valid —— 该段负责输出的帧区间 [start, end)
    """
    cuts = normalize_cuts(cut_frames, total)
    out: list[dict] = []
    for (a, b) in shot_bounds(cuts, total):
        s = a
        while s < b:
            e = min(s + chunk, b)
            # 上下文最多往回取 overlap 帧，但不得越过本镜头的起点
            ctx = min(max(overlap, 0), s - a)
            out.append({"read": (s - ctx, e), "valid": (s, e)})
            s = e
    return out


def plan_chunks_naive(total: int, chunk: int,
                      overlap: int) -> list[dict]:
    """对照组：完全不分段感知，按固定 chunk 切（用于 A/B 对比）。

    切点可能落在某段内部 → 传播会跨界 → 用来证明分段感知确实有用。
    """
    out: list[dict] = []
    s = 0
    while s < total:
        e = min(s + chunk, total)
        ctx = min(max(overlap, 0), s)
        out.append({"read": (s - ctx, e), "valid": (s, e)})
        s = e
    return out


# --------------------------------------------------------------------------
# 插帧：帧对掩码
# --------------------------------------------------------------------------
def crossing_mask(g0: int, n_frames: int, cut_frames, total: int) -> list[bool]:
    """长度 `n_frames-1` 的掩码：True 表示该帧对**跨切点**，应跳过 RIFE。

    `g0` 是这段序列第一帧的**全局帧号**。这个参数不能省 ——
    流水线为了控制内存会用微批（micro_batch），帧对可能跨越微批边界，
    只有带上全局帧号才能正确判断。
    """
    cuts = set(normalize_cuts(cut_frames, total))
    return [(g0 + i + 1) in cuts for i in range(max(n_frames - 1, 0))]


def expand_with_interp(g0: int, n_frames: int, mids: list[int],
                       cut_frames, total: int) -> list[int]:
    """把「原帧 + 中间帧」展开成最终的帧索引序列。

    正常对：   [i, mid_i, i+1]
    跨切点对： [i, i,     i+1]   ← 用 i 的副本代替中间帧（1/120 秒的停顿，不可感知）

    返回长度为 `2*n_frames` 的索引序列（最后一帧再补一次，凑满 2N）。
    索引的约定：
      · `>= 0`  → 原序列中的第 i 帧
      · `== -1` → 需要调用 RIFE 生成（调用方按 `mids` 顺序取）
      · `== -2` → 复制前一帧（跨切点，不插帧）
    """
    mask = crossing_mask(g0, n_frames, cut_frames, total)
    out: list[int] = []
    for i in range(n_frames):
        out.append(i)
        if i < n_frames - 1:
            out.append(-2 if mask[i] else -1)
    # 末尾补一帧，保证帧数严格等于 2N
    out.append(n_frames - 1)
    return out


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------
def summarize(total: int, cut_frames, chunks: list[dict],
              cross_mask: list[bool] | None = None) -> dict:
    cuts = normalize_cuts(cut_frames, total)
    info = {
        "total_frames": total,
        "n_cuts": len(cuts),
        "n_shots": len(cuts) + 1,
        "cuts": cuts,
        "n_chunks": len(chunks),
        "chunk_lens": [v["valid"][1] - v["valid"][0] for v in chunks],
        # 有多少段是从切点开始的（= 传播重置的次数）
        "propagation_resets": sum(1 for v in chunks if v["valid"][0] in set(cuts)),
    }
    if cross_mask is not None:
        info["n_crossing_pairs"] = sum(1 for m in cross_mask if m)
        info["n_interp_pairs"] = sum(1 for m in cross_mask if not m)
    return info


# --------------------------------------------------------------------------
# 自测（纯 CPU，不需要模型）
# --------------------------------------------------------------------------
def _selftest():
    total = 30
    cuts = [10, 22]          # 镜头：[0,10) [10,22) [22,30)

    print("== shot_bounds ==")
    print("  ", shot_bounds(cuts, total))

    print("== plan_chunks(chunk=8, overlap=2) ==")
    for c in plan_chunks(total, cuts, 8, 2):
        print(f"   read={c['read']}  valid={c['valid']}  "
              f"长度={c['valid'][1] - c['valid'][0]}")

    print("== 对照：plan_chunks_naive ==")
    for c in plan_chunks_naive(total, 8, 2):
        print(f"   read={c['read']}  valid={c['valid']}")

    print("== crossing_mask（整段 0..29）==")
    m = crossing_mask(0, total, cuts, total)
    print("   跨切点的对:", [i for i, x in enumerate(m) if x],
          "（应为 9 和 21，因为切点在第 10 和第 22 帧）")

    print("== expand_with_interp ==")
    idx = expand_with_interp(0, 6, [None] * 5, [3], 6)
    print("   原 6 帧、切点在第 3 帧 →", idx, f"长度 {len(idx)}（应为 12）")

    print("== 帧数守恒检查 ==")
    for n in (1, 2, 5, 100):
        for cuts_ in ([], [1], [2, 5], list(range(1, n))):
            idx = expand_with_interp(0, n, [None] * (n - 1), cuts_, n)
            assert len(idx) == 2 * n, f"帧数不守恒: n={n} cuts={cuts_}"
    print("   n=1/2/5/100 各种切点组合下 输出帧数恒为 2N ✓")


if __name__ == "__main__":
    _selftest()
