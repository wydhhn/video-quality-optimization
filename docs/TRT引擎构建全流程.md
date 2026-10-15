# TensorRT 引擎构建全流程

> 面向想给 BasicVSR++ / RIFE 做 TensorRT 加速的人，以及**接手继续优化**的 agent。
>
> 本文所有数字都是**实测**的，环境：RTX 5070 Ti（Blackwell sm_120，16 GB）、
> CUDA 12.8、TensorRT 10.x、PyTorch 2.9 + cu128。
>
> 硬件不同数字会变，但**踩过的坑和验证方法学是通用的**。

---

## 0. 先看结论

| 你要做的事 | 直接看 |
|---|---|
| 拿到能用的引擎 | `python tools/download_trt_engine.py` |
| 重新编译引擎 | [第 4 节](#4-正确的构建流程) |
| 结果为什么不对 | [第 3 节：三个坑](#3-三个必须知道的坑按代价排序) ← **先看这个，能省几小时** |
| 继续优化 | [第 7 节：还没做的](#7-还没做的与可继续优化的方向) |

**一句话**：TensorRT 在这套模型上**值得做**——
VSR 用 TF32 能到 **78.54 dB / 最大差 1/255**，比 PyTorch fp32 快 **1.69×**。
但**有三个非常隐蔽的坑**，任何一个没绕开都会让你得出"TRT 精度不行"的错误结论。

---

## 1. 为什么要做 TensorRT

`pipeline/sr.py` 的 PyTorch 版在 RTX 5070 Ti 上是 **441.9 ms/帧**（480×836 → ×4）。
一段 49 秒、582 帧的素材要跑 4 分多钟。

BasicVSR++ 的 forward 里有**上百个小 kernel**（4 条传播路径 × 每帧 15 个 64 通道 3×3 卷积），
kernel launch 开销占比很高。TensorRT 能把这些融合掉，再加上 TF32 的算力优势，
实测能到 **260.9 ms/帧**。

---

## 2. 三个模型的 TensorRT 可行性（先摸清边界）

| 模型 | 能否 TRT | 说明 |
|---|---|---|
| **BasicVSR++（VSR）** | ✅ 能，但**要绕路** | 用 `torchvision.ops.deform_conv2d`（带 mask 的可变形卷积），TRT 没有对应算子。**绕法见 3.3** |
| **RIFE（插帧）** | ✅ 能 | 只有 conv / PRelu / grid_sample / resize，导出顺畅 |
| **SpyNet（光流）** | ✅ 能 | 同 VSR 一起导出 |

> 试过且**否定**的路：ONNX Runtime CUDA EP（1646 ms/对，比 PyTorch 还慢 3.3×）；
> BasicVSR++ 的 `torch.compile`（Windows 没有 triton 轮子）；
> `jit.script`（报 `Could not cast NoneType to bool`）；
> `jit.trace`（内部 assert）。

---

## 3. 三个必须知道的坑（按代价排序）

### 3.1 ★★★ 坑一：导出 dummy 用了 `torch.zeros`，导出的图是残缺的

**这是最贵的一个坑，我一个人在上面耗了一整晚。**

`BasicVSRPlusPlusNet` 里有个**实例标志**：

```python
# check_if_mirror_extended()，在 forward 时调用
self.is_mirror_extended = False
...
if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:      # ← 全零输入时恒成立！
    self.is_mirror_extended = True              # ← 只置 True，永不重置
...
# compute_flow()
if self.is_mirror_extended:
    flows_forward = None                        # ← 直接跳过正向光流计算！
```

用 `torch.zeros(1, t, 3, h, w)` 做 dummy 导出 → 标志被置 True →
**导出的 ONNX 缺少正向光流分支**，是个结构残缺的网络。

**证据**（同一网络，只换 dummy，用 ONNX Runtime 跑导出的图对比 PyTorch）：

| dummy | `is_mirror_extended` | ORT vs PyTorch |
|---|---|---|
| `torch.zeros` | True | **55.89 dB** / 最大差 32 ❌ |
| **真实视频帧** | False | **98.64 dB** / 最大差 1 ✅ |

**这个坑的可怕之处**：现象是"TRT 精度差"，你会去怀疑精度、怀疑算子、怀疑 mixed precision
——而真正的原因是**图本身就不完整**。我在残缺图上做了一堆"精度手术"
（GridSample 拉 fp32、Resize 拉 fp32、499 个卷积拉 fp32、border→zeros 统一），
**全部无效，因为方向从一开始就错了**。换成真实帧 dummy 后，精度立刻从
31.29 dB 跳到 58.93 dB（fp16）/ 78.54 dB（TF32）。

**修法**（`tools/trt/export_vsr_onnx.py` 里已实现）：

```python
# ★ 绝不能用 torch.zeros
_s = decode_frames("某个真实视频.mp4", t)          # 解 t 帧真视频
_r = resize(torch.from_numpy(_s)..., (h, w))      # 缩放到导出尺寸
dummy = _r.float().div_(255.0)
```

**通用教训**：导出/编译模型前，先搜一遍模型里有没有
**数据相关的分支 / 实例标志**，然后确保 dummy 会把它引到**正确的那条路**。

**自检**（导出后必做，能省几小时）：

```bash
# 用 ORT 跑一遍导出的 ONNX，和 PyTorch 对比。应该 > 90 dB
python tools/trt/verify_export.py --onnx xxx.onnx --t 2 --size 836 480
```
如果只有 50 多 dB → 图是残缺的，别往下走。

---

### 3.2 ★★ 坑二：fp16 不能用 `net.half()`，要用 `autocast`

```python
net = net.half()                                    # ❌ 30.11 dB / 最大差 102
with torch.autocast("cuda", dtype=torch.float16):   # ✅ 66.98 dB / 最大差 1
    out = net(x)
```

`net.half()` 把**整个网络连权重带激活**粗暴转 fp16。
而 BasicVSR++ 里有对数值范围敏感的部分：
**SpyNet 光流的点积相关、时序递归的累加、可变形卷积的坐标与掩码** ——
整网 fp16 会把它们推到精度边缘。

`autocast` 让卷积走 fp16、**敏感算子自动留 fp32**，
用 0.27× 的速度换回 **36.7 dB** 精度。

> ⚠️ **本仓库的 `pipeline/sr.py` 目前用的就是 `net.half()`**（第 58 行附近）。
> 这是待修项 —— 见第 7 节。

**TensorRT 侧对应的是 `BuilderFlag.FP16`**，它是"整体降精度"，
效果接近 `net.half()` 而不是 `autocast`。所以**别指望 TRT 的 fp16 能维持精度**
（实测 58.93 dB / 最大差 21）。

---

### 3.3 ★ 坑三：TRT 不支持 deform_conv，也不支持 GridSample 的 border padding

#### (a) 可变形卷积

`torchvision.ops.deform_conv2d` 是自定义算子。ONNX 到 opset 28 都只有**不带 mask** 的
`DeformConv`，PyTorch 导出器也没为它注册 symbolic —— 实测直接报
`UnsupportedOperatorError: unrecognized namespace {op_name}`。
TRT 同样没有原生 DeformConv。

**绕法：拆解**（`tools/trt/deform_decompose.py`，已实现）。

带 mask 的可变形卷积本质是：

```
out(p) = Σ_k  w_k · x(p + p_k + Δp_k) · Δm_k
```

其中 `x(p + p_k + Δp_k)` 就是对 `x` 在偏移坐标处**双线性采样**。所以：

1. 把 N 个采样点**拼成一个 batch 维**，一次 `F.grid_sample` 全采完
2. 乘上 mask
3. 用一个 **1×1 卷积**做加权求和（权重 reshape 成 `(C_out, G*K*C_in/G, 1, 1)`）

**数值等价性有自测保证**（`tools/trt/vsr_decompose_test.py`）：

```
用例                        最大绝对误差      相对误差
BasicVSR++ 实配(G=16)      7.158e-06      3.49e-06    ✅
同配置+小offset(不出界)      1.025e-05      3.29e-06    ✅
单组对照(G=1)              4.053e-06      2.86e-06    ✅
多组+通道不等(G=4)          9.537e-07      8.49e-07    ✅
```

**性能上的注意**：
- **网格要纯广播构造**，不要 `torch.empty(...)` 再逐 tap 赋值 ——
  后者会把整个 `(K,H,W)` 张量**物化成常量**，导出后外部数据膨胀到 3.7 GB，
  直接超过 protobuf 的 2 GB 序列化上限（报 `Failed to serialize proto`）。
  改成广播后常量只剩 arange 小向量。
- **不要做"多段并行"**：把 n 个段塞进 batch 实测 n=2 慢 12%、n=4 慢 24%，
  显存还翻倍。算力本来就饱和。

#### (b) GridSample 的 padding_mode

`flow_warp(..., padding_mode='border')` 导出后 ONNX 里是
`GridSample(padding_mode='border')`，而 **TensorRT 只支持 zeros**，
写 border 会被**静默忽略**（改 ONNX 属性也没用，TRT 根本不读）。

后果：PyTorch 算 border、TRT 算 zeros，光流 warp 在边界算错 →
特征对齐错 → **误差经时序传播扩散到整幅画面**。

**绕法**（`tools/trt/flow_warp_trtsafe.py`，已实现）：

> 在 `align_corners=True` 下，**把归一化坐标钳到 `[-1,1]` 就严格等价于 border padding**。

实测最大差 **0.000e+00**。（也验证了 `align_corners=False` 下不成立。）

比"补边再裁"好得多 —— 后者会让中间张量变大、图更深，
TRT 构建时直接 `LLVM ERROR: out of memory`。

---

## 4. 正确的构建流程

### 4.0 环境

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install tensorrt pycuda onnx onnxruntime-gpu
```

> ★ **Blackwell（sm_120，RTX 50 系）必须用 cu128 及以上的轮子**，
> cu121 及更早会在 sm_120 上直接报 `no kernel image is available`。

### 4.1 导出 ONNX

```bash
python tools/trt/export_vsr_onnx.py --size 836 480 --t 8 --scale 4
```

脚本内部按顺序做三件事，**顺序不能错**：

1. 打 `deform` 拆解补丁（否则导出直接报 `UnsupportedOperatorError`）
2. **用真实视频帧做 dummy**（见坑一）
3. 导出 + 常量折叠

### 4.2 自检（**别跳过这步**）

```bash
python tools/trt/verify_export.py --onnx <导出的.onnx> --t 2 --size 836 480
```
应看到 PSNR > 90 dB。**如果只有 50 多 dB，说明图残缺，回去查 dummy。**

### 4.3 编译引擎

```bash
python tools/trt/build_trt_vsr.py --onnx <导出的.onnx> --mode tf32 --out vsr_tf32_t8_836x480_x4.trt
```

核心就一行配置：

```python
cfg = builder.create_builder_config()
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 10 << 30)
cfg.set_flag(trt.BuilderFlag.TF32)          # ★ 只开 TF32，不要设 FP16
```

编译约 4~5 分钟（31009 层）。

### 4.4 验证精度

```bash
python tools/trt/bench_trt_vsr.py --t 8 --size 836 480
```

应看到 PSNR ≈ 78 dB、最大差 ≈ 1/255。

---

## 5. 实测结果

**方法**：真实视频帧 480×836 → ×4，**两边都不分块、同帧数**，
参考是 PyTorch fp32。

| 实现 | ms/帧 | 相对 PyTorch fp32 | PSNR | **最大差** | >2/255 | 最大连通块 |
|---|---|---|---|---|---|---|
| PyTorch fp32（基准） | 441.9 | 1.00× | — | 0 | 0% | 0 |
| PyTorch fp16（**autocast**） | — | — | **99.00 dB** | **0.0** | 0% | 0 |
| **TensorRT TF32** ⭐ | **260.9** | **1.69×** | **78.54 dB** | **1.0** | **0.000%** | **0** |
| TensorRT fp32 | 262.8 | 1.41× | 72.22 dB | 1.0 | 0.000% | 0 |
| TensorRT fp16 | 93.3 | 3.97× | 58.93 dB | 21.0 | 0.166% | 1233 |
| TensorRT 混合(GS+Resize fp32) | 96.1 | 3.85× | 58.92 dB | 21.0 | 0.166% | 1202 |

**选 TF32 的理由**：精度最高（比 fp32 引擎还高），且比 PyTorch fp32 快 1.69×。
TF32 是 10 位尾数 + fp32 指数范围，对卷积的精度远好于 fp16，速度又比纯 fp32 快。

**"混合精度"方案没用的原因**：把 GridSample/Resize 拉回 fp32 后
精度纹丝不动（58.93 → 58.92）——说明残余误差来自 **fp16 卷积本身**，
不是这两个算子。所以想再压精度只能整网 fp32/TF32。

---

## 6. RIFE 的 TRT 现状

**结论：能做，但精度不达标，当前不建议用。**

| 实现 | ms/对 | 加速 | PSNR | 最大差 | >2/255 |
|---|---|---|---|---|---|
| PyTorch fp32（真值） | ~300 | — | — | 0 | 0% |
| **PyTorch fp16** | 162.6 | 0.83× | **62.72 dB** | **5.0** | 0.000% |
| TRT-fp16 | 83.4 | 1.95× | 46.35 dB | 57.0 | 8.57% |
| **TRT-warp+resize** | **85.8** | **1.90×** | **49.88 dB** | **32.0** | 0.30% |

**已经做过的工作**（`tools/trt/export_rife_onnx.py` 里都有）：

1. **ONNX 图手术做混合精度**：`layer.precision` 自 TRT 10.12 起已废弃（强设会 NaN）。
   正确路线是**整图转 fp16，再在 GridSample/Resize 前后显式插 Cast**，
   用 `NetworkDefinitionCreationFlag.STRONGLY_TYPED` 建网络且**不加 FP16 flag**。
   → 精度从 39.95 dB 提到 47.66 dB，**速度不变**。
   两个坑：① 插 Cast 时**不能直接改 `node.output[0]` 的名字**（下游会断链），
   要让算子输出到中间名、再用 Cast 转回原名；② 必须跳过可选的空输入
   （Resize 的 roi/scales/sizes 可能是 `""`）。
2. **TF32 试过**：TRT 实际建成了 fp32 引擎（56.8 MB、31.8 ms），无收益。
3. **`warp` 的 border padding**：RIFE 也用 `padding_mode='border'`，
   `flow_warp_trtsafe.py` 里的坐标钳位方案对它同样适用，**但 RIFE 没接这个补丁**
   （它的精度问题主要是 fp16 卷积，不是 padding）。

**为什么卡住**：TRT 的 fp16 卷积实现本身精度就不如 cuDNN 的 fp16
（cuDNN 用 fp32 累加，TRT 没有对应旋钮），而 RIFE 对 fp16 又特别敏感。
**当前的取舍是保住精度、用 PyTorch fp16**（62.72 dB / 最大差 5）。

**可继续优化的方向**：见 7.3。

---

## 7. 还没做的，与可继续优化的方向

### 7.1 ★ 修 `pipeline/sr.py` 的 fp16 用法

**当前 `sr.py` 用的是 `net.half()`**，按 3.2 的实测这会掉到 30 dB。
改成 `autocast` 可直接拿到 **66.98 dB**（代价是 1.85× → 1.53×，但精度换回 37 dB）。

这是一行级别的改动，收益很大。

### 7.2 ✓ TRT 引擎接进 `pipeline/sr.py`

已提供 `pipeline/sr_trt.py`（`SrTrtEngine`，接口与 `SrEngine` 完全一致）。
**还没做的**：

- `SrTrtEngine` 目前**忽略 `cut_frames` / `shot_aware`** ——
  要接镜头感知，得把切点先灌进 `shotaware.plan_chunks` 再喂引擎。
  引擎窗口是固定 `t` 帧，和 `plan_chunks` 的 `read`/`valid` 模型能对上，
  但需要一层适配（把 `read` 区间补/裁到恰好 `t` 帧）。
- `run.py` 里加一个 `--sr-backend {torch,trt}` 开关。

### 7.3 RIFE 的 TRT 优化

- 试 **`t`（每次插几对）**：现在 batch=1，RIFE 在 4x 分辨率下单对就要 85 ms，
  可能有批处理空间（注意 BasicVSR++ 那边的结论是"合批无收益"，但 RIFE 是浅层网络，情况不同）。
- 把 `flow_warp_trtsafe` 接到 RIFE 上（虽然推断影响不大，但值得实测确认）。
- 试 **INT8 量化**：RIFE 是浅层网络，INT8 的精度损失可能可接受。

### 7.4 其他可探索的

- **`t` 与 `overlap` 的重新规划**：当前 TRT 路径 `overlap=0`，
  而 PyTorch 路径是 `chunk=8, overlap=2`。两者**不等价**——
  TRT 版段间缺时序连续性。要做成等价，应按 **`t = chunk + overlap = 10`** 编译，
  并把适配器的 `chunk/overlap` 设成 8/2。
- **`cudnn.benchmark`**：`sr.py` 现在是 `True`。实测它在**稳态**下只快 ~2%，
  但每个新 shape 要付 **9~15 秒**算法试探，且在段长 4 上会**病态变慢**
  （预热 8 次后 12 次调用 5 分钟跑不完）。任务普遍不长时建议设 `False`。

---

## 8. 验证方法学（比任何单点结论都重要）

这一节是血泪换来的。**做性能/精度对比时，下面每一条都至少坑过我一次。**

### 8.1 比性能必须统一帧数
拿"240 帧的隔离跑"对比"整段素材的流水线跑"，会得出"流水线慢 4.88 倍"的假结论。

### 8.2 比精度必须统一分块 —— BasicVSR++ 不是因果的
**曾经以为**"BasicVSR++ 是因果的时序递归，尾部补帧不影响前面任何一帧"。
**实测证伪**：

```
8 帧  vs  8帧+尾部重复2帧（只比前 8 帧输出）
  最大差 27.0/255   平均 0.126/255   PSNR 58.22 dB
  第 0 帧 16.6、第 1 帧 14.3、第 2 帧 17.8 …… 每一帧都被改动
```

原因：有 `backward_1` / `backward_2` 两条**反向传播**路径（从最后一帧往前传），
尾部多出的帧会沿反向路径影响**前面每一帧**。

**推论**：
- 分块方式本身就能造成 **51~55 dB** 量级的差异（chunk8 vs c4 = 51.23 dB）
- 所有"补帧"策略都**不是无损的**
- **比精度时两边必须用同样的分块方式**，否则测的是分块差异，不是你要测的东西

### 8.3 看精度不能只看最大差
要同时看：**中位数 / 分位数**（判断是偏移还是噪声）、
**超阈像素占比**、以及**最大连通域大小**。

散点肉眼不可见，成片才是伪影。
（实测 fp16 的最大连通块有 1233 像素 → 那是看得见的块状伪影；
而 TF32 的最大连通块是 0。）

### 8.4 显存要看 `reserved`，不是 `allocated`
PyTorch 的缓存分配器会留住大块空闲块。
实测两个模型共存时 `allocated` 只有 9.31 GiB，但 **`reserved` 是 18.11 GiB**
—— 超过物理显存 → 驱动层换页 → **慢 4.8 倍且不报错**。
（症状：GPU 利用率 100% 但功耗只有 99W，满载应该 250W+。）

### 8.5 预热不计入计时，且预热输入不能用 `np.zeros`
`cudnn.benchmark=True` 下每个新 shape 要试探 9~15 秒；
`np.zeros` 会触发镜像判定（见坑一），耗时偏乐观。

### 8.6 测精度不能用白噪声
随机噪声在任意分辨率下都会让**所有**实现的精度崩掉（连长 3360×1920 的
PyTorch fp16 都只有 21.84 dB），完全反映不出真实差异。**必须用真实视频帧。**

---

## 9. 相关文件

```
tools/trt/
├── README.md                 使用说明
├── deform_decompose.py       可变形卷积拆解（数值等价，带自测）
├── flow_warp_trtsafe.py      border padding → 坐标钳位
├── export_vsr_onnx.py        导出 VSR ONNX（★ 含真实帧 dummy 修复）
├── export_rife_onnx.py       导出 RIFE ONNX + 混合精度图手术
├── trt_runner.py             引擎运行封装（★ 输入 dtype 必须读）
├── build_trt_vsr.py          编译 TF32 引擎
├── verify_export.py          导出自检（ORT vs PyTorch）
├── bench_trt_vsr.py          精度/速度基准
├── diag_error_map.py         误差空间分布（边框 vs 内部、连通域）
├── diag_vsr_fp16.py          定位 fp16 用法的诊断
└── vsr_decompose_test.py     拆解等价性自测

pipeline/
├── sr.py                     PyTorch 版（★ 待修 fp16 用法）
└── sr_trt.py                 TensorRT 版（接口一致，可互换）

docs/
└── TRT引擎构建全流程.md        本文

.claude/skills/trt-vsr-engine/SKILL.md   给 agent 的精简版
```

---

## 10. 一句话总结

**TensorRT 在这套模型上值得做，但有三个非常隐蔽的坑。**
其中**最贵的一个是导出 dummy** —— 它会让你的图悄悄残缺，
然后你会花一整天去优化一个根本不存在的问题。
**导出后立刻用 ORT 对一遍（应 > 90 dB），这一步能省下最多时间。**
