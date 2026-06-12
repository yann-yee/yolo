# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math

import torch
import torch.nn as nn


class WiseIoULoss(nn.Module):
    """Wise-IoU and SIoU family losses for bounding box regression."""

    def __init__(
        self,
        ltype: str = "WIoU",
        monotonous: bool | None = False,
        momentum: float = 1e-2,
        alpha: float = 1.7,
        delta: float = 2.7,
        theta: float = 4.0,
    ):
        super().__init__()
        normalized = ltype.replace("-", "").replace("_", "").lower()
        if normalized == "wiou":
            self.ltype = "wiou"
        elif normalized == "siou":
            self.ltype = "siou"
        else:
            raise ValueError(f"Unsupported Wise-IoU loss type: {ltype}")

        self.monotonous = monotonous
        self.momentum = momentum
        self.alpha = alpha
        self.delta = delta
        self.theta = theta
        self.register_buffer("iou_mean", torch.tensor(1.0))

    @staticmethod
    def _boxes_iou(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return IoU and geometric helpers for xyxy boxes."""
        lt = torch.maximum(pred[..., :2], target[..., :2])
        rb = torch.minimum(pred[..., 2:], target[..., 2:])
        inter_wh = (rb - lt).clamp_min(0)
        inter = inter_wh[..., 0] * inter_wh[..., 1]

        pred_wh = (pred[..., 2:] - pred[..., :2]).clamp_min(0)
        target_wh = (target[..., 2:] - target[..., :2]).clamp_min(0)
        union = pred_wh[..., 0] * pred_wh[..., 1] + target_wh[..., 0] * target_wh[..., 1] - inter + 1e-9
        iou = inter / union

        enclose_lt = torch.minimum(pred[..., :2], target[..., :2])
        enclose_rb = torch.maximum(pred[..., 2:], target[..., 2:])
        enclose_wh = (enclose_rb - enclose_lt).clamp_min(1e-9)
        center_pred = (pred[..., :2] + pred[..., 2:]) / 2
        center_target = (target[..., :2] + target[..., 2:]) / 2
        center_delta = center_pred - center_target
        return iou, pred_wh, target_wh, enclose_wh, center_delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor, ret_iou: bool = False) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor
    ]:
        """Compute Wise-IoU or SIoU loss for xyxy boxes."""
        iou, pred_wh, target_wh, enclose_wh, center_delta = self._boxes_iou(pred, target)

        if self.ltype == "wiou":
            if self.training:
                self.iou_mean.mul_(1 - self.momentum)
                self.iou_mean.add_(self.momentum * iou.detach().mean())

            beta = (iou.detach() / self.iou_mean.clamp_min(1e-9)).clamp_min(1e-9)
            if self.monotonous is True:
                weight = beta.sqrt()
            elif self.monotonous is False:
                alpha = torch.as_tensor(self.alpha, device=pred.device, dtype=pred.dtype)
                delta = torch.as_tensor(self.delta, device=pred.device, dtype=pred.dtype)
                weight = beta / (delta * torch.pow(alpha, beta - delta))
            else:
                weight = 1.0
            loss = (1.0 - iou) * weight

        else:
            sigma = torch.sqrt(center_delta.square().sum(-1) + 1e-9)
            sin_alpha = torch.minimum(center_delta.abs()[..., 0], center_delta.abs()[..., 1]) / sigma
            sin_alpha = sin_alpha.clamp(0, 1 - 1e-9)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            gamma = 2.0 - angle_cost

            rho_x = center_delta[..., 0].square() / enclose_wh[..., 0].square().clamp_min(1e-9)
            rho_y = center_delta[..., 1].square() / enclose_wh[..., 1].square().clamp_min(1e-9)
            distance_cost = 2.0 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)

            w_ratio = torch.abs(pred_wh[..., 0] - target_wh[..., 0]) / target_wh[..., 0].clamp_min(1e-9)
            h_ratio = torch.abs(pred_wh[..., 1] - target_wh[..., 1]) / target_wh[..., 1].clamp_min(1e-9)
            shape_cost = torch.pow(1.0 - torch.exp(-w_ratio), self.theta) + torch.pow(
                1.0 - torch.exp(-h_ratio), self.theta
            )
            loss = (1.0 - iou) + 0.5 * (distance_cost + shape_cost)

        return (loss, iou) if ret_iou else loss
