"""把官方代码依赖的 `mmcv.ops` 变形卷积换成 torchvision 原生实现。

为什么不直接装 mmcv：
  mmcv 的 wheel 不含 ops，源码安装要编译 CUDA 算子（还挑 torch 版本），
  在无编译环境的 Windows 上基本装不上。而 `mmcv-lite`（纯 Python 版）
  只提供 `mmcv.cnn` 等，没有 ops。

替代方案：
  **`torchvision.ops.deform_conv2d` 是 PyTorch 原生的可变形卷积实现**，
  与 mmcv 的 `modulated_deform_conv2d` 语义一致：
      input   (N, C, H, W)
      offset  (N, deform_groups*2*kh*kw, out_h, out_w)
      mask    (N, deform_groups*1*kh*kw, out_h, out_w)
      weight  (out_c, in_c//groups, kh, kw)
  参数顺序不同（mmcv 把 mask 放在第 3 位），下面按 mmcv 的签名封装。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.ops


def _pair(v):
    if isinstance(v, (tuple, list)):
        return (int(v[0]), int(v[1]))
    return (int(v), int(v))


def modulated_deform_conv2d(input, offset, mask, weight, bias=None,
                            stride=1, padding=0, dilation=1,
                            groups=1, deform_groups=1):
    """签名与 mmcv.ops.modulated_deform_conv2d 一致。"""
    return torchvision.ops.deform_conv2d(
        input, offset, weight, bias,
        stride=_pair(stride), padding=_pair(padding), dilation=_pair(dilation),
        mask=mask)


class ModulatedDeformConv2d(nn.Module):
    """mmcv.ops.ModulatedDeformConv2d 的最小等价实现（只保留被用到的部分）。

    官方 `SecondOrderDeformableAlignment` 继承它并覆写 forward，
    实际依赖的成员是：weight / bias / stride / padding / dilation /
    groups / deform_groups / out_channels。
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deform_groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        kh, kw = self.kernel_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kh, kw))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, offset, mask):
        return modulated_deform_conv2d(
            x, offset, mask, self.weight, self.bias,
            self.stride, self.padding, self.dilation,
            self.groups, self.deform_groups)


# ---------------------------------------------------------------------------
# 以下是对 mmengine / mmcv.cnn 的最小垫片。
#
# 官方代码除了 mmcv.ops 还依赖这几个：
#   mmengine.model.BaseModule                      → 只是个基类
#   mmengine.model.weight_init.constant_init       → 常量初始化
#   mmengine.MMLogger / print_log                  → 日志
#   mmcv.cnn.ConvModule                            → conv(+norm)(+act) 组合块
# 都是纯 Python 的简单东西，没必要为此引入两个大依赖（还会挑 torch 版本）。
# ---------------------------------------------------------------------------


class BaseModule(nn.Module):
    """mmengine.model.BaseModule 的轻量替代（只保留推理需要的部分）。"""

    def __init__(self, init_cfg=None, **kwargs):
        super().__init__()
        self.init_cfg = init_cfg
        self._is_init = False

    def init_weights(self):
        """官方实现会按 init_cfg 做初始化；推理场景权重由 checkpoint 覆盖，无需处理。"""
        self._is_init = True
        return self

    def _is_init_state(self):
        return self._is_init


def constant_init(module, val, bias=0):
    """把 module 的 weight/bias 填成常数（对应 mmengine 的同名函数）。"""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class MMLogger:
    """只提供 `get_current_instance()` 的极简 logger。"""

    _instance = None

    @classmethod
    def get_current_instance(cls):
        if cls._instance is None:
            import logging
            lg = logging.getLogger("vsr")
            if not lg.handlers:
                lg.addHandler(logging.StreamHandler())
            lg.setLevel(logging.INFO)
            cls._instance = lg
        return cls._instance


def print_log(msg, logger=None, level=20, **kwargs):
    """对应 mmengine.print_log：有 logger 就用它，否则直接打印。"""
    if logger is not None and hasattr(logger, "log"):
        logger.log(level, msg)
    else:
        print(msg, flush=True)


_ACT = {"ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "GELU": nn.GELU,
        "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh, "ELU": nn.ELU,
        "SiLU": nn.SiLU, "Swish": nn.SiLU}


class ConvModule(nn.Module):
    """mmcv.cnn.ConvModule 的轻量替代。

    **属性命名必须和 mmcv 完全一致**（`self.conv` / `self.bn` / `self.activate`），
    否则 checkpoint 的 key 会对不上：mmcv 里 `self.conv` 直接是 `nn.Conv2d`
    （key 为 `...conv.weight`），如果把 conv+norm+act 打包成 `nn.Sequential`
    就会变成 `...conv.0.weight`，加载权重时全部报「多余 / 缺失」。
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias="auto",
                 conv_cfg=None, norm_cfg=None, act_cfg=dict(type="ReLU"),
                 inplace=True, order=("conv", "norm", "act"), **kwargs):
        super().__init__()
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias
        self.order = order

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bool(bias))
        if self.with_norm:
            ntype = (norm_cfg or {}).get("type", "BN")
            if ntype in ("BN", "BN2d", "BatchNorm2d"):
                self.bn = nn.BatchNorm2d(out_channels)
            elif ntype in ("GN", "GroupNorm"):
                self.bn = nn.GroupNorm(norm_cfg.get("num_groups", 32), out_channels)
            else:
                raise NotImplementedError(f"未支持的 norm 类型: {ntype}")
        if self.with_activation:
            atype = act_cfg.get("type", "ReLU") if isinstance(act_cfg, dict) else act_cfg
            cls = _ACT.get(atype, nn.ReLU)
            kw = {k: v for k, v in (act_cfg or {}).items() if k != "type"}
            if "inplace" not in kw and cls in (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SiLU):
                kw["inplace"] = inplace
            self.activate = cls(**kw)

    def forward(self, x):
        for op in self.order:
            if op == "conv":
                x = self.conv(x)
            elif op == "norm" and self.with_norm:
                x = self.bn(x)
            elif op == "act" and self.with_activation:
                x = self.activate(x)
        return x


def install():
    """把所有垫片注入 sys.modules，让官方源码的 import 直接可用。"""
    import sys
    import types

    # --- mmcv ---
    mmcv_mod = types.ModuleType("mmcv")
    mmcv_ops = types.ModuleType("mmcv.ops")
    mmcv_ops.ModulatedDeformConv2d = ModulatedDeformConv2d
    mmcv_ops.modulated_deform_conv2d = modulated_deform_conv2d
    mmcv_cnn = types.ModuleType("mmcv.cnn")
    mmcv_cnn.ConvModule = ConvModule
    mmcv_mod.ops = mmcv_ops
    mmcv_mod.cnn = mmcv_cnn
    sys.modules["mmcv"] = mmcv_mod
    sys.modules["mmcv.ops"] = mmcv_ops
    sys.modules["mmcv.cnn"] = mmcv_cnn

    # --- mmengine ---
    me_mod = types.ModuleType("mmengine")
    me_mod.MMLogger = MMLogger
    me_mod.print_log = print_log
    me_model = types.ModuleType("mmengine.model")
    me_model.BaseModule = BaseModule
    me_wi = types.ModuleType("mmengine.model.weight_init")
    me_wi.constant_init = constant_init
    me_model.weight_init = me_wi
    me_mod.model = me_model
    sys.modules["mmengine"] = me_mod
    sys.modules["mmengine.model"] = me_model
    sys.modules["mmengine.model.weight_init"] = me_wi
