# webapp — 网页版增强服务

把 `run.py` 的命令行流水线包成一个网页服务：
**上传 → 异步任务 → 实时进度 → 下载**。

---

## 快速开始

```bash
pip install -r requirements.txt          # 含 fastapi / uvicorn / python-multipart
python webapp/serve.py --host 0.0.0.0 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。

也可以：

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000   # 需先 cd webapp/
```

---

## 它和 `run.py` 的关系

| | `run.py`（命令行） | `webapp/`（网页） |
|---|---|---|
| 输入 | 本地文件路径 | 浏览器上传（或指定服务器路径） |
| 输出 | 写到指定路径 | 任务完成后下载 |
| 进度 | 终端输出 | 网页实时进度条 |
| 并发 | 一次一个 | 任务队列，可查历史 |
| 适用 | 批处理、脚本化 | 交互式、给别人用 |

**核心逻辑是同一套**：`webapp/pipeline.py` 就是命令行那条链路的服务化封装，
同样走 `VSR → RIFE` 的串联，同样支持镜头感知。

---

## 目录

```
webapp/
├── serve.py           启动入口（读参数 → uvicorn）
├── app.py             FastAPI 应用：REST API + 静态页
├── pipeline.py        ★ 流水线：VSR → RIFE 串联 + 有界缓冲 + 微批
├── jobs.py            异步任务管理（提交/进度/取消/产物）
├── config.py          ★ 路径配置（双布局自适应，见下）
├── autoconfig.py      按输出分辨率自动选后端与参数，防止显存超限换页
├── public.py          公网隧道（localhost.run），用于把服务暴露出去
├── mp4meta.py         解析 MP4 的 colr box 取色彩元数据（环境无 ffprobe 时用）
├── engine/
│   ├── vsr_engine.py     BasicVSR++（PyTorch）
│   └── rife_engine.py    RIFE（torch / onnx / trt 三条后端）
└── static/index.html  网页界面
```

### `config.py` 的双布局自适应

同一份代码在两种目录结构下都能跑：

| 布局 | 模型源码 | 权重 |
|---|---|---|
| **A. 本仓库** | `vendor/` | `weights/` |
| **B. 原项目**（FrameInterpolation） | `超分模型/` `插帧模型/` | 各自目录下 |

判定方式是**看目录存不存在**，优先用本仓库布局。
想强制指定根目录，设环境变量 `FI_BASE`。

排查"为什么找不到模型"：

```bash
python webapp/config.py      # 打印路径判定结果
```

---

## 和 `tools/trt/` 的配合

`engine/` 里的后端是**可替换**的。装好 TensorRT 引擎后可以换成 TRT 版：

```python
from pipeline.sr_trt import SrTrtEngine    # 接口与 SrEngine 完全一致
sr = SrTrtEngine(chunk=8)                  # 自动找 t=8 的 TF32 引擎
```

实测 TF32 版：**78.54 dB / 最大差 1/255**，比 PyTorch fp32 快 **1.69×**。
详见 [`docs/TRT引擎构建全流程.md`](../docs/TRT引擎构建全流程.md)。

> ⚠️ **待修**：`engine/vsr_engine.py` 里用的是 `net.half()`，
> 按实测这会掉到 30 dB。应改成 `torch.autocast`（能到 66.98 dB）。

---

## ⚠️ 首次使用前

1. **下载权重**：`python tools/download_weights.py`
2. **（可选）下载 TRT 引擎**：`python tools/download_trt_engine.py`
3. 检查 `python webapp/config.py` 里的权重路径是否都是 ✅

---

## 显存注意

这套流水线在 16 GB 卡上是**贴着上限**跑的。两个模型共存时
`torch.cuda.memory_reserved()` 可能超过物理显存 → 驱动层换页 →
**慢 4.8 倍且不报错**（症状：GPU 利用率 100% 但功耗只有 99W）。

代码里已做的应对：
- `autoconfig.py` 按输出分辨率自动选参数，超预算就降级
- 每个微批跑完 `torch.cuda.empty_cache()` 释放缓存块

**看显存要看 `reserved`，不是 `allocated`** —— 缓存分配器会留住大块空闲块。
