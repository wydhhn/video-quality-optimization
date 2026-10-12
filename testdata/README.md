# 测试素材

用于验证分段引擎的合成测试视频与真值标注。**全部是程序合成的**，不涉及任何真实素材的版权问题。

---

## 文件清单

| 文件 | 内容 | 真值 |
|---|---|---|
| `test_suite.mp4` | **算法测试集**：29.5 秒，10 个刻意构造的场景串成一段 | `test_suite_gt.json` |
| `test_video_real.mp4` | 36 秒，12 个镜头（3 张照片各 4 个裁切） | 12 个切点 |
| `test_video_pan.mp4` | 18 秒，3 段连续缓慢横摇 + 2 个硬切 | 3 个镜头 |
| `test_video_whip.mp4` | 2.4 秒，快速甩镜（极限情况） | 3 个镜头 |
| `test_video.mp4` | 早期的小测试片段 | — |
| `assets/` | 3 张源照片（生成上述视频用） | — |

---

## `test_suite.mp4` 的 10 个场景

这是最有价值的测试集，专门覆盖边界情况：

| # | 场景 | 时长 | 期望行为 |
|---|---|---|---|
| 1 | 静态镜头（基线） | 3s | 不该切 |
| 2 | 硬切 A→B | 3s | 段末应切 |
| 3 | **缓慢横摇**（5 秒摇过整幅画面） | 5s | **内部不该切** |
| 4 | **缓慢推近**（5 秒推近 1.4 倍） | 5s | **内部不该切** |
| 5 | 全景 C | 2s | — |
| 6 | **特写 C**（同一空间换机位） | 2s | **应切** |
| 7 | 静态 + **中间插 2 帧闪白** | 3s | **闪白处应切**（闪光视为真实场景切换） |
| 8 | **快速甩镜**（1.5 秒） | 1.5s | 内部不该切 |
| 9 | **淡入淡出转场** | 2s | 窗口内应有切点 |
| 10 | 静态镜头 | 3s | 不该切 |

**真值共 9 个预期切点 + 5 个「不该切」区间。**

> 闪光策略：闪光（相机补光、闪白转场）在成片里通常就是一次真实的场景切换，所以测试集把闪白位置算作**预期切点**，而不是误报。

---

## 用法

```bash
cd video-shot-seg

# 跑测试集
python seg.py --video testdata/test_suite.mp4 --slice --overlay

# 对照真值评估
python tools/evaluate.py --gt testdata/test_suite_gt.json \
                         --scenes out/segments_omnishotcut.json
```

---

## 重新生成测试素材

生成脚本在 `tools/` 下，需要从 `testdata/` 目录运行（脚本里用的是相对路径 `assets/scene_a.jpg`）：

```bash
cd video-shot-seg/testdata

python ../tools/make_test_suite.py --out test_suite.mp4        # 算法测试集
python ../tools/make_test_pan.py --out test_video_pan.mp4      # 缓慢横摇
python ../tools/make_test_pan.py --out test_video_whip.mp4 --pan-seconds 0.8  # 快速甩镜
python ../tools/make_test_video_real.py --out test_video_real.mp4  # 12 镜头
```

`tools/make_ai_test_video.py` 会从 HuggingFace 下载多条 Kling AI 片段拼成 1080p 长视频（需要网络）。
