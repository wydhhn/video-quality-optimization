"""TensorRT 引擎的最小运行封装 —— 与具体模型无关。

为什么要自己封一层（而不是直接用 `tensorrt` 原生 API）：
  1. **输入 dtype 必须读出来，不能猜**。ONNX 是 fp32 导出的，引擎输入就期待 fp32；
     如果按"引擎是 fp16 的"想当然喂 fp16 张量，TRT 会按 fp32 去解引用 fp16 缓冲区，
     读到**未初始化内存** —— 表现是 PSNR 5.5 dB、最大差 545（值域才 0~1），
     看着像"精度崩了"，其实是绑错了内存。`get_tensor_dtype()` 必读。
  2. **显存要自己管**。TRT 的 device buffer 不在 PyTorch 的缓存分配器里，
     必须显式释放，否则和 PyTorch 抢显存（本机 16 GB 卡上这点很致命）。
  3. 接口统一成"张量进、张量出"，方便替换。

用法：
    r = TrtRunner("xxx.trt")
    y = r(x)            # x 的 dtype 必须与 r.in_dtype 一致，shape 必须与引擎一致
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

_DTYPE_MAP: dict[Any, Any] = {}


def _dtype_map():
    """trt.DataType → torch.dtype 的映射（延迟构造，避免 import 时就拉起 trt）。"""
    if not _DTYPE_MAP:
        import tensorrt as trt
        import torch
        _DTYPE_MAP.update({
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        })
    return _DTYPE_MAP


class TrtRunner:
    """一个定形引擎的执行封装。输入输出都走 GPU 张量，不经过 numpy。"""

    def __init__(self, engine_path: str, verbose: bool = True):
        import tensorrt as trt
        import torch
        self.torch = torch
        self.trt = trt
        self.path = engine_path
        self.verbose = verbose

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            ser = f.read()
        # ★ Runtime 必须**存成实例属性**，不能让它是局部变量。
        #   engine 的生命周期依附于创建它的 Runtime —— Runtime 被 GC 掉之后，
        #   engine 就成了悬空引用，运行时表现为
        #       CUDA error: an illegal memory access was encountered
        #       Internal Error: MyelinCheckException ... l2cm_layer_internal.cpp
        #   （排查了很久：同一个 .trt 在别的封装里跑得好好的，唯独这里崩，
        #     唯一差别就是那边把 Runtime 留在了表达式里没被立刻回收。）
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(ser)
        if self.engine is None:
            raise RuntimeError(f"引擎反序列化失败: {engine_path}")
        self.ctx = self.engine.create_execution_context()

        names = [self.engine.get_tensor_name(i)
                 for i in range(self.engine.num_io_tensors)]
        self.in_names = [n for n in names
                         if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.out_names = [n for n in names
                          if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.in_name = self.in_names[0]
        self.out_name = self.out_names[0]

        # ★ 输入 dtype 必须读，不能假设（见模块 docstring）
        dm = _dtype_map()
        self.in_dtype = dm.get(self.engine.get_tensor_dtype(self.in_name),
                               torch.float32)
        self.out_dtype = dm.get(self.engine.get_tensor_dtype(self.out_name),
                                torch.float32)
        self.in_shape = tuple(self.engine.get_tensor_shape(self.in_name))

    # ------------------------------------------------------------------ #
    def __call__(self, x):
        torch = self.torch
        # ★ 必须拦下 CPU 张量。
        #   把 CPU 张量的 data_ptr() 传给 TRT，它会当成显存地址去解引用 ——
        #   报出来的是 `CUDA error 700 launching __myl_SlicReshMove kernel` 或
        #   `MyelinCheckException: l2cm_layer_internal.cpp ... CHECK(false)`，
        #   完全指不到"张量还在内存条上"这个真正的原因。
        #   （踩过：pipeline/sr_trt.py 里漏了一个 .to(device)。）
        if not x.is_cuda:
            raise ValueError(
                f"输入张量在 {x.device}，TensorRT 引擎只能吃 CUDA 张量。\n"
                f"  多半是忘了 `.to('cuda')` —— CPU 指针被当成显存地址会触发\n"
                f"  CUDA error 700 / MyelinCheckException，报错信息完全指不到根因。")
        if x.dtype != self.in_dtype:
            x = x.to(self.in_dtype)
        if tuple(x.shape) != self.in_shape:
            # 定形引擎：形状不符会静默算错或直接崩，这里直接拦下来
            raise ValueError(
                f"引擎要求输入形状 {self.in_shape}，收到 {tuple(x.shape)}。\n"
                f"  （TensorRT 引擎是固定形状的，换尺寸/换帧数都要重新编译）")

        x = x.contiguous()
        self.ctx.set_input_shape(self.in_name, self.in_shape)
        out_shape = tuple(self.ctx.get_tensor_shape(self.out_name))
        out = torch.empty(out_shape, dtype=self.out_dtype, device=x.device)

        self.ctx.set_tensor_address(self.in_name, x.data_ptr())
        self.ctx.set_tensor_address(self.out_name, out.data_ptr())
        # ★ 每次都重新取当前 stream，**不要缓存这个裸句柄**。
        #   缓存过一版，结果运行时报 `CUDA error: an illegal memory access`
        #   （还伴随 MyelinCheckException）。句柄在上下文/stream 变化后会失效，
        #   而失效的句柄传给 execute_async_v3 是未定义行为。
        #   参考实现（build_trt_rife.py 的 TrtRife）也是每次都取的。
        ok = self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("execute_async_v3 返回失败")
        return out

    def __del__(self):
        # TRT 的显存不在 PyTorch 缓存分配器里，要显式放
        try:
            del self.ctx
            del self.engine
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass
