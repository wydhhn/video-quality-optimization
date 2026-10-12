"""对照真值评估分段结果：哪些该切的切到了、哪些不该切的误切了。

用法：
  python evaluate_suite.py --gt test_suite_gt.json --scenes out_suite/scenes_shot.json
"""

import argparse
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--tol", type=float, default=0.6, help="切点匹配容差（秒）")
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8"))
    raw = json.load(open(args.scenes, encoding="utf-8"))

    # 兼容两种分段结果格式：
    #   本项目统一格式 {"engine":..., "segments":[{id,start,end,...}]}
    #   旧格式         [{scene_id,start,end,...}]
    if isinstance(raw, dict):
        segs = raw.get("segments", [])
    else:
        segs = raw
    if not segs:
        raise SystemExit(f"没有读到任何分段：{args.scenes}")

    detected = [round(float(s["end"]), 2) for s in segs[:-1]]
    expected = list(gt["expected_cuts"])
    tw = gt.get("transition_window")

    # 转场窗口单独处理：窗口内任意一个切点都算命中
    exp_hard = [e for e in expected if not (tw and tw["start"] <= e <= tw["end"])]
    exp_trans = [e for e in expected if tw and tw["start"] <= e <= tw["end"]]

    used = set()
    hit, miss = [], []
    for e in exp_hard:
        cand = [d for d in detected if d not in used and abs(d - e) <= args.tol]
        if cand:
            best = min(cand, key=lambda d: abs(d - e))
            used.add(best)
            hit.append((e, best))
        else:
            miss.append(e)

    trans_ok = None
    if exp_trans:
        in_win = [d for d in detected if d not in used
                  and tw["start"] - 0.2 <= d <= tw["end"] + 0.2]
        trans_ok = bool(in_win)
        if in_win:
            used.add(in_win[0])

    # 误切：落在「不该切」区间内的检出
    false_cuts = []
    for d in detected:
        if d in used:
            continue
        for z in gt["forbidden"]:
            if z["start"] <= d <= z["end"]:
                false_cuts.append((d, z["name"]))
                break

    # ---- 输出 ----
    print("=" * 74)
    print(f"预期切点 {len(expected)} 个 | 实际检出 {len(detected)} 个 | 容差 ±{args.tol}s")
    print("=" * 74)
    print(f"\n【该切的】命中 {len(hit)}/{len(exp_hard)}")
    for e, d in hit:
        print(f"  ✓ 预期 {e:>6.2f}s  →  检出 {d:>6.2f}s  （误差 {abs(d-e):.2f}s）")
    for e in miss:
        print(f"  ✗ 预期 {e:>6.2f}s  →  漏检")
    if trans_ok is not None:
        mark = "✓" if trans_ok else "✗"
        print(f"  {mark} 淡入淡出转场窗口 {tw['start']:.2f}–{tw['end']:.2f}s  →  "
              f"{'检出' if trans_ok else '漏检'}")

    print(f"\n【不该切的】误切 {len(false_cuts)} 个")
    if false_cuts:
        for d, name in false_cuts:
            print(f"  ✗ 在 {d:>6.2f}s 误切  （区间：{name}）")
    else:
        print("  ✓ 全部「不该切」区间都没有被误切")

    print(f"\n实际检出的全部切点：{[round(d,2) for d in detected]}")
    print("=" * 74)

    n_ok = len(hit) + (1 if trans_ok else 0)
    n_exp = len(exp_hard) + (1 if exp_trans else 0)
    print(f"小结：应切 {n_exp} 处，切中 {n_ok} 处；不应切的地方误切 {len(false_cuts)} 处")


if __name__ == "__main__":
    main()
