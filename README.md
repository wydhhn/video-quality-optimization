# video-quality-optimization

**面向 AI 生成视频的画质优化工具链** —— 镜头切分 · 插帧 · 超分，三者共享一套「镜头感知」处理逻辑。

> Shot segmentation · Frame interpolation · Super-resolution for AI-generated video,
> with shot-aware processing that keeps transitions from bleeding across cuts.

---

## 一、这是什么

一套端到端的视频增强流水线，三个可独立使用的能力：

| 能力 | 模型 | 说明 |
|---|---|---|
| **镜头切分** | **OmniShotCut v1.5**（默认）/ TransNetV2 | 把视频按镜头边界切开，输出时间戳与帧号 |
| **插帧** | RIFE ×2 | 帧率翻倍（如 12fps → 24fps），时长不变 |
| **超分** | BasicVSR++ ×4 | 分辨率 4 倍（如 270×480 → 1080×1920） |

三个能力可以任意组合，入口是两个脚本：

```bash
python seg.py --video 输入.mp4                    # 只做镜头切分
python run.py --video 输入.mp4 --mode fi          # 插帧
python run.py --video 输入.mp4 --mode sr          # 超分
python run.py --video 输入.mp4 --mode sr+fi       # 超分 + 插帧
```

---

## 二、核心思路：为什么必须「镜头感知」

这是本项目相对于直接调用单个模型的最大区别。

**问题**：超分和插帧模型都假设输入是**一段连续的镜头**。如果直接喂整段视频，切点处会出事：

| 模型 | 跨切点会发生什么 | 修法 |
|---|---|---|
| **BasicVSR++** | 双向时序传播会去算「镜头 A 最后一帧」到「镜头 B 第一帧」的光流——这两帧毫无关系，算出的是垃圾；垃圾光流被用来 warp 特征并沿时间累积传播 → **切点附近拖影、串色** | 切点作为**强制分段边界**，让传播在切点重置 |
| **RIFE** | 跨切点的帧对会生成一个**把两个不同场景融合在一起的中间帧** → 典型鬼影 | 跨切点的帧对**跳过插帧**，用重复帧代替 |

**实测证据**（合成测试，切点处左镜头纯红、右镜头纯绿）：

| | R | G | B | 判定 |
|---|---|---|---|---|
| 源帧 9（红） | 250 | 15 | 15 | — |
| 源帧 10（绿） | 15 | 252 | 15 | — |
| **镜头感知**插出的中间帧 | **250** | **15** | 15 | ✅ 前一帧副本，干净 |
| **不分段**插出的中间帧 | **49** | **213** | 11 | ❌ 红绿混合 = **鬼影** |

**所以流程是**：

```
视频
 └─[1] 镜头切分（OmniShotCut）──→ 切点列表
      └─[2] 处理（超分 / 插帧）
           · 超分：切点强制分段，时序传播在切点重置
           · 插帧：跨切点的帧对跳过，用重复帧补位（保证帧数严格守恒 2N）
      └─[3] 输出
```

用 `--no-shot-aware` 可以关掉它做 A/B 对比。

---

## 三、快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**不需要** `mmcv` / `mmengine` / `transformers`——详见 `requirements.txt` 里的说明。

### 2. 下载模型权重（约 275 MB）

```bash
python tools/download_weights.py
```

脚本会自动下载 5 个权重到 `weights/`，并逐个校验字节数：

| 文件 | 大小 | 用途 |
|---|---|---|
| `OmniShotCut_ckpt.pth` | 201 MB | 镜头切分（默认） |
| `transnetv2-pytorch-weights.pth` | 29 MB | 镜头切分（备选） |
| `basicvsr_plusplus_reds4.pth` | 28 MB | 超分 |
| `spynet.pth` | 5.5 MB | 超分的对齐网络 |
| `flownet.pkl` | 11.6 MB | 插帧 |

### 3. 跑起来

```bash
# 镜头切分
python seg.py --video 输入.mp4 --slice --overlay

# 增强流水线
python run.py --video 输入.mp4 --mode sr+fi --out 输出.mp4
```

---

## 四、命令速查

### `seg.py` —— 镜头切分

```bash
python seg.py --video 输入.mp4                    # 默认 OmniShotCut
python seg.py --video 输入.mp4 --engine transnetv2 # 换轻量引擎
python seg.py --video 输入.mp4 --engine both      # 两个引擎都跑并对比
python seg.py --video 输入.mp4 --slice --overlay  # 导出切片 + 带标注的视频
```

### `run.py` —— 增强流水线

```bash
python run.py --video 输入.mp4 --mode sr          # 超分 ×4
python run.py --video 输入.mp4 --mode fi          # 插帧 ×2
python run.py --video 输入.mp4 --mode sr+fi       # 超分 + 插帧
python run.py --video 输入.mp4 --mode sr+fi --no-shot-aware   # 对照组
python run.py --video 输入.mp4 --mode fi --slowmo # 帧率不变、时长翻倍（便于逐帧看）
python run.py --video 输入.mp4 --mode fi --cuts-file 已有切点.json  # 跳过分段模型
```

---

## 五、整体链路

### 镜头切分：OmniShotCut（默认，当前 SOTA）

```
视频
 ├─[1] 解码 ── ffmpeg 解码并缩放到 96×128 RGB
 ├─[2] 滑窗 ── 窗口 100 帧，重叠 20 帧 → 步长 80
 ├─[3] 编码 ── ResNet18 backbone → 多尺度特征图
 ├─[4] 预测 ── Shot-Query Transformer（24 query / 6 层 decoder）
 │             每个 query 通过自注意力看整个窗口，输出三路：
 │               pred_shot_logits   → 镜头结束帧位置
 │               intra_clip_logits  → 镜头内部类型（转场/运镜）
 │               inter_clip_logits  → 镜头间关系（硬切/软转场）
 ├─[5] 解码 ── 每窗口的 query 输出 → 局部镜头区间
 ├─[6] 归并 ── 重叠区「对半劈开」划分有效区间，杜绝切点重复
 └─[7] 过滤 ── 最短镜头过滤（min_shot）→ 分段结果
```

### 增强流水线（`run.py`）

```
视频
 ├─[1] 镜头分段 ── 调 seg.py 的引擎，拿到切点
 └─[2] 处理（三线程流水：解码 → 推理 → 编码，有界队列 + 微批）
      ├── 超分：切点强制分段，分块内做双向时序传播
      └── 插帧：逐帧对插值，跨切点帧对跳过
 └─[3] 编码 ── libx264
```

---

## 六、关键超参

分成两类：**模型层**（模型自带，改不了）和**策略层**（工程可调）。

### 模型层

| 项 | OmniShotCut | TransNetV2 | 说明 |
|---|---|---|---|
| 输入分辨率 | **96×128** | 27×48 | 论文消融：64×96 / 96×128 / 128×160 中 **96×128 最优** |
| 窗口长度 | **100 帧（硬约束）** | 100 帧（训练设定） | OmniShotCut 的输出头维度、位置编码、reshape 全按 100 设计 |
| 时间感受野 | 整个窗口（全局注意力） | **±50 帧** | 架构决定 |
| 参数量 | 52.8 M | 7.6 M | |

> OmniShotCut 的消融显示 **128×160 反而比 96×128 差**——"提高分辨率"不是单调更好。

### 策略层

| 参数 | 默认 | 适用 | 作用 |
|---|---|---|---|
| `--engine` | **omnishotcut** | 通用 | 选分段引擎 |
| `--overlap` | **20** | OmniShotCut | 窗口重叠帧数。**不要用官方的 10**（见下节） |
| `--label-filter` | **all** | OmniShotCut | 按转场类型过滤：`all`（推荐）/ `non_padding` / `hard_only` |
| `--min-shot` | 0.4 秒 | 通用 | 最短镜头，过滤碎片 |
| `--chunk` / `--overlap` | 8 / 2 | 超分 | 超分的分段长度 / 重叠帧数 |
| `--step` / `--select` | 50 / center | TransNetV2 | 滑窗步长 / 每帧取哪次预测 |

---

## 七、一个值得记下的坑：OmniShotCut 会静默丢切点

这是本项目在接入 OmniShotCut 时发现的**官方默认配置缺陷**，两处偏离官方默认值都有实测依据。

### ① `overlap` 默认 20，不是官方的 10

官方 `overlap=10` 时，落在**窗口起点附近**的切点会被模型打成 `Padding` 标签，
再被官方的 `mode="clean_shot"` **静默丢弃**（不报警，只能靠数段数发现）。

实测一段 48.96 秒、每 4.08 秒一个硬切的视频（11 个真值切点）：

| overlap | 检出 | 结果 |
|---|---|---|
| **10（官方默认）** | 10 个切点 | **漏掉第 1 个** ✗ |
| **20（本项目默认）** | 11 个切点 | **全部命中** ✓ |

根因：该切点在窗口内的局部位置从 7 移到 17，标签由 `Padding` 恢复正常。

### ② `label_filter` 默认 `all`，不是官方的 `clean_shot`

官方 `clean_shot` 只保留 `General` 标签的区间，会丢掉
`Dissolve / Wipes / Push / Slide / Zoom / Fade / Doorway`——
但**这些都是真实的镜头边界**，其中 `Push / Slide / Zoom` 本身就是运镜。丢掉等于漏检。

三个视频的实测对照：

| 过滤策略 | 测试集（10 切点） | mailbox（11 切点） |
|---|---|---|
| `hard_only`（官方 clean_shot） | 10/10，最大误差 0.080s | **10/11，漏 1** ✗ |
| `non_padding` | 10/10，最大误差 0.000s | **10/11，漏 1** ✗ |
| **`all`（本项目默认）** | 10/10，最大误差 **0.000s** | **11/11** ✓ |

只靠 `min_shot` 去噪就够了——短命区间会被自然滤掉，真实边界全部保留。

> **教训**：`mode="clean_shot"` 是在**模型内部**就把非 General 区间丢掉的，
> 调用方连丢了多少都看不见。凡遇到「库内部帮你过滤」的设计，
> 都该怀疑它是否在静默吞掉真实结果。

---

## 八、实测数据

**环境：纯 CPU**（16 逻辑核，无独立显卡）。

### 镜头切分

| 视频 | 引擎 | 结果 |
|---|---|---|
| 算法测试集 640×360 / 29.5s（10 个真值切点） | **OmniShotCut** | **10/10 全中，误切 0**，误差 **0.00s** |
| 算法测试集 640×360 / 29.5s | TransNetV2 | 10/10 全中，误差 0.04s（1 帧） |
| 徒步视频 1280×720 / 45s（4 个真值切点） | **OmniShotCut** | 5 段，**4/4 全中** |
| mailbox 1080×1920 / 48.96s（11 个真值切点） | **OmniShotCut** | **11/11 全中**，误差 **≈0.00s（帧级精确）** |
| mailbox 1080×1920 / 48.96s | TransNetV2 | 11/11 全中，误差 0.042s（稳定早 1 帧） |

> mailbox 的真值用**逐帧像素差**独立验证：11 个切点处帧间差异 53~115，
> 正常帧间差异只有 2~12。

### 插帧（RIFE ×2）

在 49 秒视频上，582 帧 → 1164 帧，耗时 239 秒。

| 方法 | 仅插出帧的 PSNR |
|---|---|
| **RIFE 插帧** | **34.24 dB** |
| 基线：复制前一帧 | 30.66 dB |
| **增益** | **+3.58 dB** |

> ⚠️ 这个对照必须让基线和被测方法**走同一条压缩管线**。
> 用「干净原帧」当基线会得出 −0.54 dB 的错误结论。

### 超分（BasicVSR++ ×4）—— 结论要诚实

| 素材 | 输入 → 输出 | 双三次放大 | **BasicVSR++** | 差值 |
|---|---|---|---|---|
| AI 生成（mailbox） | 270×480 → 1080×1920 | **38.65** | 35.74 | **−2.90 dB** |
| 自然视频（徒步） | 320×180 → 1280×720 | **31.25** | 29.28 | **−1.97 dB** |

**PSNR/SSIM 上 BasicVSR++ 输给双三次**，但画面上它明显更锐——高频能量是真值的 2.6~4.4 倍
（它在**过锐**、合成纹理）。这是典型的**保真度 vs 感知质量**取舍：

- **PSNR 偏向双三次**：它忠实还原输入的模糊，不添加任何东西
- **感知清晰度偏向 BasicVSR++**：它合成高频细节，但合成的细节不可能与真值逐像素吻合

已排除的原因（都验证过）：输入压缩损失、退化模型不匹配、分块切断时序、亮度偏移、代码用错。

**建议**：想要真·超分收益，需要 ① GPU；② **干净的双三次下采样输入**（不是从压缩视频再降采样）；
③ 用感知指标（LPIPS/DISTS）而非 PSNR 评判。

### CPU 性能参考

| 任务 | 分辨率 | 速度 |
|---|---|---|
| RIFE 插帧 | 270×480 | ~400 ms/帧 |
| BasicVSR++ 超分 | 270×480 → 1080×1920 | ~11.2 s/帧（另加 6~7.5 分钟预热） |

> 超分的预热开销极大（`pipeline/sr.py` 的 `warmup()` 要跑一遍 cudnn 算法选型）。
> 短片段跑超分时，**预热比处理本身还久**。

---

## 九、项目结构

```
video-quality-optimization/
├── seg.py                      镜头切分入口
├── run.py                      增强流水线入口（sr / fi / sr+fi）
├── bootstrap.py                路径 / 设备 / ffmpeg 初始化
├── requirements.txt
│
├── engines/                    分段引擎
│   ├── omnishotcut.py            ★ 默认（当前 SOTA）
│   └── transnetv2.py             轻量备选
│
├── pipeline/                   增强流水线
│   ├── shotaware.py              ★ 镜头感知核心（切点对齐 + 帧号追踪）
│   ├── sr.py                    超分封装（BasicVSR++）
│   ├── fi.py                    插帧封装（RIFE）
│   └── flow.py                  编排（三线程 + 有界队列 + 微批）
│
├── vendor/                     第三方代码
│   ├── omnishotcut/             OmniShotCut 官方包（含本地补丁）
│   ├── vsr/                     BasicVSR++ + mmcv/mmengine 垫片
│   ├── rife/rife_ms/            RIFE
│   └── transnetv2_model.py      TransNetV2 网络定义
│
├── tools/
│   ├── download_weights.py      ★ 一键下载全部权重
│   ├── render_overlay.py        生成带分段标注的视频
│   ├── evaluate.py              对照真值评估
│   └── make_*.py                测试素材生成脚本
│
├── testdata/                   测试素材（含真值 json）
├── docs/                       详细报告
│   ├── 视频增强实测报告.md
│   └── 验证报告_分段感知流水线.md
└── weights/                    权重（gitignore，用脚本下载）
```

---

## 十、本地适配补丁

官方代码假设有 GPU 和完整 ffmpeg 工具链，本项目做了以下适配：

| 位置 | 问题 | 补丁 |
|---|---|---|
| `vendor/omnishotcut/engine.py` | 硬编码 `model.to("cuda")` | 自动选择设备，无显卡回落 CPU |
| `vendor/omnishotcut/datasets/utils.py` | 用 `ffmpeg.probe()` 取帧率，依赖 ffprobe | 改用 OpenCV |
| `vendor/vsr/compat.py` | 依赖 mmcv / mmengine | 注入轻量垫片，**无需安装** |
| `bootstrap.py` | PATH 上可能有坏掉的 ffmpeg shim | 优先用项目自带，且必须实际跑一次验证 |

---

## 十一、引用与许可

本项目的 `vendor/` 目录包含以下第三方代码，版权归原作者：

- **OmniShotCut** — UVA Computer Vision Lab，NeurIPS 2026
- **TransNetV2** — Tomáš Souček，arXiv:2008.04838
- **BasicVSR++ / SPyNet** — OpenMMLab，MIT License
- **RIFE** — Megvii Inc.，MIT License

本项目自身代码采用 **MIT License**，见 `LICENSE`。
