# tools/trt — TensorRT 引擎构建链

给 BasicVSR++ / RIFE 做 TensorRT 加速的完整工具集。

> **完整技术文档见 [`docs/TRT引擎构建全流程.md`](../../docs/TRT引擎构建全流程.md)**，
> 里面有所有实测数据和推导过程。本文件只是使用说明。
>
> 如果你是 agent，先读
> [`.claude/skills/trt-vsr-engine/SKILL.md`](../../.claude/skills/trt-vsr-engine/SKILL.md)
> —— 那是精简版，只保留"不知道就会踩坑"的部分。

---

## 快速开始

### 拿现成的引擎（推荐）

```bash
python tools/download_trt_engine.py
```

引擎（135 MB）走 GitHub Release 发布 —— 因为 **GitHub 单文件上限 100 MB**，
git 推不上去。这和 `tools/download_weights.py` 是同一个套路。

### 自己编译

```bash
# 1) 导出 ONNX（内部会打 deform 拆解补丁 + 用真实帧做 dummy）
python tools/trt/export_vsr_onnx.py --size 836 480 --t 8 --scale 4

# 2) ★ 自检：用 ONNX Runtime 对比 PyTorch，应 > 90 dB
python tools/trt/verify_export.py --onnx <导出的.onnx> --t 2 --size 836 480

# 3) 编译引擎（约 4~5 分钟）
python tools/trt/build_trt_vsr.py --onnx <导出的.onnx> --mode tf32 \
    --out trt5/vsr_tf32_t8_836x480_x4.trt

# 4) 验证精度（应 ≈ 78 dB / 最大差 ≈ 1）
python tools/trt/bench_trt_vsr.py --engine trt5/vsr_tf32_t8_836x480_x4.trt \
    --t 8 --size 836 480
```

**第 2 步别跳过。** 它 30 秒，能直接告诉你图对不对；
跳过它你可能在错误方向上耗一整天。

---

## 脚本清单

| 文件 | 作用 |
|---|---|
| **`export_vsr_onnx.py`** | 导出 VSR 的 ONNX。含 **★ 真实帧 dummy 修复**（见文档 3.1） |
| **`deform_decompose.py`** | 可变形卷积拆解为 `grid_sample + mask + 1×1conv`，带数值等价自测 |
| **`flow_warp_trtsafe.py`** | border padding → 坐标钳位（TRT 只支持 zeros） |
| **`build_trt_vsr.py`** | 编译引擎，`--mode tf32/fp32/fp16` |
| **`verify_export.py`** | 导出自检：ORT vs PyTorch |
| **`bench_trt_vsr.py`** | 精度/速度基准 |
| `trt_runner.py` | 引擎运行封装。★ 输入 dtype 必须读，不能猜 |
| `export_rife_onnx.py` | RIFE 的 ONNX 导出 + 混合精度图手术 |
| `diag_error_map.py` | 误差空间分布（边框 vs 内部、连通域大小） |
| `diag_vsr_fp16.py` | 定位"fp16 用错方式"的诊断 |
| `vsr_decompose_test.py` | 拆解等价性自测（纯 CPU，不需要 GPU） |

---

## 三个坑（详见文档第 3 节）

1. **导出 dummy 不能用 `torch.zeros`** → 触发 `is_mirror_extended`，
   导出的图**缺正向光流分支**，精度只剩 31 dB。**这是最贵的一个。**
2. **fp16 不能用 `net.half()`**，要用 `torch.autocast`（差 37 dB）
3. **TRT 不支持 `deform_conv2d`，也不支持 GridSample 的 border padding**
   —— 两个都有绕法，见 `deform_decompose.py` 和 `flow_warp_trtsafe.py`

---

## 环境要求

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install tensorrt pycuda onnx onnxruntime-gpu
```

> ★ **Blackwell（sm_120，RTX 50 系）必须用 cu128+ 的轮子**，
> cu121 及更早在 sm_120 上会直接报 `no kernel image is available`。

---

## 引擎命名约定

```
vsr_{mode}_t{t}_{h}x{w}_x4.trt
     │      │
     │      └─ 每次前向吃几帧（编译时写死，改了就重编）
     └─ tf32 / fp32 / fp16

例：vsr_tf32_t8_836x480_x4.trt
    → TF32，一次吃 8 帧，输入 480×836，输出 ×4
```

**`t` 必须和 `pipeline/sr_trt.py` 的 `chunk + overlap` 对上。**
Python 侧 `SrTrtEngine(chunk=8)` 会自动找 `t=8` 的引擎。
