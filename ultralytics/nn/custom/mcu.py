# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Custom MCU-oriented modules for yolo11-mcu."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv, Detect

__all__ = ("BiFPNAdd", "DecoupledDetect", "ECA", "MBConv")


class ECA(nn.Module):
    """Efficient Channel Attention."""

    def __init__(self, c1: int, k_size: int = 3):
        super().__init__()
        k_size = k_size if k_size % 2 == 1 else k_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


class MBConv(nn.Module):
    """Light mobile inverted bottleneck block."""

    def __init__(self, c1: int, c2: int, expand_ratio: float = 2.0, shortcut: bool = True):
        super().__init__()
        hidden = max(int(c1 * expand_ratio), 1)
        self.expand = Conv(c1, hidden, 1, 1)
        self.dwconv = Conv(hidden, hidden, 3, 1, g=hidden)
        self.project = Conv(hidden, c2, 1, 1, act=False)
        self.act = copy_act()
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.dwconv(self.expand(x)))
        return self.act(y + x) if self.add else self.act(y)


class BiFPNAdd(nn.Module):
    """Weighted BiFPN-style fusion for two same-shape tensors."""

    def __init__(self, c2: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.refine = Conv(c2, c2, 1, 1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        assert len(x) == 2, f"BiFPNAdd expects exactly 2 inputs, got {len(x)}"
        w = F.relu(self.weight)
        w = w / (w.sum() + self.eps)
        y = w[0] * x[0] + w[1] * x[1]
        return self.refine(y)


class DecoupledDetect(Detect):
    """MCU-friendly detection head name for a decoupled box/classification head."""

    pass


def copy_act() -> nn.Module:
    """Create the default activation used by Conv."""
    act = Conv.default_act
    return act if not isinstance(act, nn.Module) else type(act)(**_module_kwargs(act))


def _module_kwargs(module: nn.Module) -> dict[str, object]:
    """Best-effort reconstruction of module kwargs for shallow copies."""
    if hasattr(module, "inplace"):
        return {"inplace": getattr(module, "inplace")}
    return {}
