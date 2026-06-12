# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Custom modules for the yolo11-mobileone-gsconv architecture."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv, Detect
from ultralytics.utils.torch_utils import fuse_conv_and_bn

__all__ = ("GSConv", "MobileOne", "RepDetect", "SimSPPF", "VoVGSCSP")


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """Shuffle channels to mix grouped features."""
    if groups <= 1 or x.shape[1] % groups:
        return x
    b, c, h, w = x.shape
    x = x.reshape(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.reshape(b, c, h, w)


class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation block used by MobileOne."""

    def __init__(self, c: int, reduction: int = 16):
        super().__init__()
        c_ = max(c // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c_, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class MobileOne(nn.Module):
    """Re-parameterizable MobileOne block."""

    def __init__(self, c1: int, c2: int, use_se: bool = False):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.use_se = use_se

        self.rbr_dense = nn.Sequential(nn.Conv2d(c1, c2, 3, 1, 1, bias=False), nn.BatchNorm2d(c2))
        self.rbr_1x1 = nn.Sequential(nn.Conv2d(c1, c2, 1, 1, 0, bias=False), nn.BatchNorm2d(c2))
        self.rbr_identity = nn.BatchNorm2d(c1) if c1 == c2 else None
        self.se = SqueezeExcite(c2) if use_se else nn.Identity()
        self.act = copy.deepcopy(Conv.default_act) if isinstance(Conv.default_act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            return self.act(self.se(self.reparam_conv(x)))

        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.act(self.se(out))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.se(self.reparam_conv(x)))

    @torch.no_grad()
    def fuse(self) -> None:
        """Fuse the branch convolutions into a single conv layer."""
        if hasattr(self, "reparam_conv"):
            return

        conv3x3 = fuse_conv_and_bn(self.rbr_dense[0], self.rbr_dense[1])
        conv1x1 = fuse_conv_and_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        kernel1x1 = F.pad(conv1x1.weight.data, [1, 1, 1, 1])
        bias1x1 = conv1x1.bias.data
        kernelid, biasid = self._fuse_identity()

        reparam = nn.Conv2d(self.c1, self.c2, 3, 1, 1, bias=True).requires_grad_(False)
        reparam.weight.data.copy_(conv3x3.weight.data + kernel1x1 + kernelid)
        reparam.bias.data.copy_(conv3x3.bias.data + bias1x1 + biasid)
        self.reparam_conv = reparam

        del self.rbr_dense
        del self.rbr_1x1
        if self.rbr_identity is not None:
            del self.rbr_identity

    def _fuse_identity(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse the identity branch batch norm into a 3x3 conv kernel."""
        if self.rbr_identity is None:
            return 0, 0

        bn = self.rbr_identity
        kernel = torch.zeros((self.c2, self.c1, 3, 3), device=bn.weight.device, dtype=bn.weight.dtype)
        for i in range(self.c2):
            kernel[i, i, 1, 1] = 1.0
        std = (bn.running_var + bn.eps).sqrt()
        scale = (bn.weight / std).reshape(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * bn.weight / std
        return kernel * scale, bias


class GSConv(nn.Module):
    """GSConv module with channel shuffle."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        c_ = max(c2 // 2, 1)
        c3 = c2 - c_
        self.cv1 = Conv(c1, c_, 1, s)
        self.cv2 = Conv(c_, c3, k, 1, g=math.gcd(c_, c3), act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        return channel_shuffle(torch.cat((x1, x2), 1), 2)


class VoVGSCSP(nn.Module):
    """VoV-GSCSP block built from GSConv."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        c_ = max(int(c2 * e), 1)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*(GSConv(c_, c_) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SimSPPF(nn.Module):
    """Simplified SPPF block used by the custom backbone."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = max(c1 // 2, 1)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y = [x]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class RepDetect(Detect):
    """Detection head alias for custom YAML architectures."""

    pass
