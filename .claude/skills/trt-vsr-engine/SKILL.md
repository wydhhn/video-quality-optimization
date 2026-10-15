---
name: trt-vsr-engine
description: >
  给 BasicVSR++ / RIFE 做 TensorRT 加速时使用。覆盖可变形卷积拆解、border padding 等价替换、
  混合精度图手术，以及三个会导致"精度莫名崩掉"的隐蔽陷阱（导出 dummy、fp16 用法、TRT 算子限制）。
  当用户提到 TensorRT / TRT / 引擎构建 / 加速推理 / 精度对不上 / deform_conv 导不出 /
  或想让 VSR、RIFE 跑得更快时，加载本 skill。
---

# TensorRT 引擎构建（VSR / RIFE）

## 什么时候用

- 要给 BasicVSR++、RIFE 做 TensorRT 加速
- TensorRT 结果和 PyTorch 对不上，在排查精度
- 导出 ONNX 报 `UnsupportedOperatorError` / 精度只有 30~55 dB
- 接手继续优化这套引擎

**完整版见 `docs/TRT引擎构建全流程.md`**（含所有实测数据和推导过程）。
本文件是精简版，只保留"不知道就会踩坑"的部分。

---

## ★ 三条铁律（违反任何一条，你会在错误的方向上耗一整天）

### 铁律 1：导出 dummy 必须用真实数据，绝不能用 `torch.zeros`

`BasicVSRPlusPlusNet` 里有个**实例标志**，全零输入会把它置 True，
导致导出时**跳过正向光流分支**——你拿到的是一个结构残缺的图：

```python
if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:   # ← 全零时恒成立
    self.is_mirror_extended = True            # ← 只置 True，永不重置
...
if self.is_mirror_extended:
    flows_forward = None                      # ← 跳过正向光流
```

| dummy | ORT vs PyTorch |
|---|---|
| `torch.zeros` | **55.89 dB** ❌ |
| 真实视频帧 | **98.64 dB** ✅ |

**这个坑的表现是"TRT 精度差"**，你会去怀疑精度/算子/混合精度，
而真正原因是**图本身不完整**。在此之上做的所有精度优化都是白费。

**做法**：解 t 帧真视频 → resize 到导出尺寸 → 当 dummy。

**自检**：导出后立刻用 ONNX Runtime 跑一遍和 PyTorch 对比，
**应 > 90 dB**。只有 50 多 dB 就回去查 dummy。

> 通用原则：导出前先搜一遍模型里有没有**数据相关分支 / 实例标志**。

### 铁律 2：fp16 用 `autocast`，不要 `net.half()`

```python
net = net.half()                                    # ❌ 30.11 dB / 最大差 102
with torch.autocast("cuda", dtype=torch.float16):   # ✅ 66.98 dB / 最大差 1
    out = net(x)
```

BasicVSR++ 里 **SpyNet 光流的点积相关、时序递归累加、可变形卷积坐标**对 fp16 极敏感。
`autocast` 让卷积走 fp16、敏感算子留 fp32，**用 0.27× 速度换回 36.7 dB**。

TensorRT 的 `BuilderFlag.FP16` 相当于 `net.half()` 而非 `autocast` ——
别指望它的 fp16 能维持精度。

### 铁律 3：对比实验必须统一条件

| 维度 | 要求 | 不遵守会怎样 |
|---|---|---|
| 帧数 | 两边跑同样帧数 | 得出"流水线慢 4.88 倍"的假结论 |
| **分块方式** | 两边用同样的 chunk/overlap | 测出的是分块差异，不是你要测的 |
| 素材 | **必须真实视频帧** | 白噪声下所有实现都崩（连 fp16 PyTorch 都只有 21.84 dB） |
| 预热 | 不计入计时，且不用 `np.zeros` | `np.zeros` 触发镜像判定，偏乐观 |

**BasicVSR++ 不是因果的**：它有 `backward_1`/`backward_2` 反向传播，
尾部补帧会改变**前面每一帧**（实测 8帧 vs 8帧+补2帧 → 第 0 帧也差 16.6/255）。
所以：

- 分块本身就能造成 **51~55 dB** 量级差异
- **任何"补帧"策略都不是无损的**
- 想无损 → 让 `chunk` 正好等于引擎的 `t`

---

## TRT 的两个算子限制，以及绕法

### `deform_conv2d` 不被支持 → 拆解

ONNX 到 opset 28 都没有带 mask 的 DeformConv，TRT 也没有。

**拆解原理**：带 mask 的可变形卷积 = 在偏移坐标处双线性采样 × mask × 加权求和。

1. N 个采样点拼成 batch 维，**一次 `grid_sample` 采完**
2. 乘 mask
3. **1×1 卷积**做加权求和（权重 reshape 成 `(C_out, G*K*C_in/G, 1, 1)`）

数值等价性有自测（相对误差 ~3e-6）。注意：
- 网格**纯广播构造**，别 `torch.empty` 逐 tap 赋值（会物化成常量，导出后 3.7 GB，
  超 protobuf 2 GB 上限）
- **别做多段并行**（n=2 慢 12%、n=4 慢 24%，显存还翻倍）

### GridSample 的 `border` padding 被忽略 → 坐标钳位

TRT 的 GridSample **只支持 zeros**，ONNX 写 border 会被静默忽略（改属性也没用）。

**绕法**：`align_corners=True` 下，**把归一化坐标钳到 `[-1,1]` 严格等价于 border**
（实测最大差 0.000e+00；`align_corners=False` 下不成立）。

比"补边再裁"好：后者会让图更大更深，TRT 构建直接 `LLVM ERROR: out of memory`。

---

## 预期结果（RTX 5070 Ti，480×836 → ×4，对 PyTorch fp32 真值）

| 实现 | ms/帧 | 加速 | PSNR | 最大差 | 建议 |
|---|---|---|---|---|---|
| PyTorch fp32 | 441.9 | 1.00× | — | 0 | 基准 |
| PyTorch fp16（autocast） | — | — | 99.00 dB | 0.0 | 精度首选 |
| **TRT TF32** | **260.9** | **1.69×** | **78.54 dB** | **1.0** | ⭐ 推荐 |
| TRT fp32 | 262.8 | 1.41× | 72.22 dB | 1.0 | 可 |
| TRT fp16 | 93.3 | 3.97× | 58.93 dB | 21.0 | 精度不够 |

**选 TF32 的理由**：10 位尾数 + fp32 指数范围，对卷积精度远好于 fp16、速度又比 fp32 快。
**混合精度（把 GridSample/Resize 拉回 fp32）无效** —— 残余误差来自 fp16 卷积本身。

环境注意：**Blackwell（sm_120，RTX 50 系）必须用 cu128+ 的 PyTorch 轮子**。

---

## RIFE 的现状（别重复踩）

**能做，但精度不达标，当前建议用 PyTorch fp16。**

| 实现 | ms/对 | 加速 | PSNR | 最大差 |
|---|---|---|---|---|
| PyTorch fp16 | 162.6 | 0.83× | **62.72 dB** | **5.0** |
| TRT-warp+resize | 85.8 | **1.90×** | 49.88 dB | 32.0 |

**已做的工作**：
- **混合精度图手术**：`layer.precision` 自 TRT 10.12 已废弃（强设 NaN）。
  正确做法是整图转 fp16 + 在 GridSample/Resize 前后**显式插 Cast** +
  `STRONGLY_TYPED` 建网且不加 FP16 flag → 39.95 → **47.66 dB，速度不变**。
  两个坑：① 插 Cast **不能直接改 `node.output[0]` 名字**（下游断链），
  要输出到中间名再用 Cast 转回原名；② **跳过可选空输入**（Resize 的 roi/scales 可能是 `""`）。
- TF32 试过：TRT 建成了 fp32 引擎，无收益。

**卡住的原因**：TRT 的 fp16 卷积精度不如 cuDNN（cuDNN 用 fp32 累加，TRT 无对应旋钮），
而 RIFE 对 fp16 又特别敏感。**取舍是保精度**。

**可继续做的**：试 batch>1（RIFE 是浅层网，与 VSR 结论可能不同）；
把坐标钳位接到 RIFE 的 warp；试 INT8。

---

## 待办（接手可从这里开始）

1. **修 `pipeline/sr.py` 的 `net.half()`** → 改 autocast（一行，+37 dB）
2. **把 `SrTrtEngine` 接上镜头感知** —— 目前忽略 `cut_frames`，
   需要把切点灌进 `shotaware.plan_chunks` 再喂引擎
3. `run.py` 加 `--sr-backend {torch,trt}` 开关
4. RIFE 的 TRT 继续优化（见上）
5. **`t` 与 `overlap` 对齐**：TRT 路径现在 `overlap=0`，PyTorch 路径 `chunk=8,overlap=2`，
   两者**不等价**。要等价应按 **`t = chunk + overlap = 10`** 编译

---

## 常见误判 → 先查什么

| 现象 | 大概率原因 |
|---|---|
| 精度只有 30~35 dB | **导出 dummy 用了 zeros** → 图残缺（先查这个！） |
| 精度 55~59 dB | fp16 卷积本身的精度上限，正常 |
| 精度对不上但都很高 | **分块方式不一致**（边界处时序断了） |
| `UnsupportedOperatorError` | 没打 deform 拆解补丁 |
| ONNX 超过 2 GB 序列化失败 | 网格用 `torch.empty` 物化成了常量 |
| TRT 构建 `LLVM ERROR: out of memory` | 图太大，检查是否做了"补边再裁" |
| 精度突然变成 5 dB、最大差 545 | 喂进去的张量 dtype 和引擎输入不符（读到未初始化内存） |
| 慢 4~5 倍但不报错 | **显存 reserved 超了物理显存**，驱动在换页 |
