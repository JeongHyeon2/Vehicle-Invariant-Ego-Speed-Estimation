"""Final 48x86 C=3 EgoSpeed-SmartROI model."""

from __future__ import annotations

import math

import torch
from torch import nn

from egospeed.models.backbone import (
    AdaptiveFeatureTransformer,
    ContextualMotionAnalysis,
    DynamicIntegrationGate,
    MotionFeatureExtraction,
    SpatialFeatureExtraction,
)


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.coefficient * gradient, None


class GradientReversalLayer(nn.Module):
    def __init__(self, coefficient: float = 1.0) -> None:
        super().__init__()
        self.lambda_grl = float(coefficient)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_grl)


class ResBlock3D(nn.Module):
    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class SpatialDisentanglementModuleResCNN(nn.Module):
    def __init__(self, channels: int = 256, num_blocks: int = 1) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        self.speed_branch = nn.Sequential(*[ResBlock3D(channels) for _ in range(num_blocks)])
        self.vehicle_branch = nn.Sequential(
            *[ResBlock3D(channels) for _ in range(num_blocks)]
        )
        self.speed_dim = channels
        self.vehicle_dim = channels
        self.num_blocks = num_blocks


class SpeedHead(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 512) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class VehicleHead(nn.Module):
    def __init__(self, in_dim: int, num_vehicles: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_vehicles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class EgoSpeedSmartROI(nn.Module):
    """FlexiNet + SmartROI + vehicle-adversarial disentanglement.

    Args:
        num_vehicles: Number of source vehicle domains in the current fold.
        input_channels: Must be 3: grayscale and raw optical-flow rates (u, v).
        latent_dim: Must be 512 for compatibility with the released checkpoints.
        grl_lambda: Gradient reversal coefficient. The adversarial loss weight is
            scheduled separately during training.
        num_blocks: Number of residual 3D blocks per disentanglement branch.
    """

    def __init__(
        self,
        num_vehicles: int = 4,
        input_channels: int = 3,
        latent_dim: int = 512,
        grl_lambda: float = 1.0,
        num_blocks: int = 1,
    ) -> None:
        super().__init__()
        if input_channels != 3:
            raise ValueError("The final model requires C=3: grayscale + raw flow (u, v)")
        if latent_dim != 512:
            raise ValueError("The final model uses latent_dim=512 (256 per branch)")

        self.cma_block = ContextualMotionAnalysis(input_channels, 64)
        self.aft_block = AdaptiveFeatureTransformer(64, 128)
        self.sfe_module = SpatialFeatureExtraction(128, 256)
        self.mfe_module = MotionFeatureExtraction(128, 192, 256)
        self.dig = DynamicIntegrationGate(output_channels=256)

        self.disent = SpatialDisentanglementModuleResCNN(256, num_blocks=num_blocks)
        self.speed_head = SpeedHead(256, 512)
        self.vehicle_head = VehicleHead(256, num_vehicles, 256)
        self.vehicle_adv_head = VehicleHead(256, num_vehicles, 256)
        self.grl = GradientReversalLayer(grl_lambda)
        self.num_vehicles = num_vehicles
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks

    def extract_backbone(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 5 or clip.shape[2] != 3:
            raise ValueError(f"Expected clip shape (B,T,3,H,W), got {tuple(clip.shape)}")
        x = clip.permute(0, 2, 1, 3, 4)
        cma = self.cma_block(x)
        aft = self.aft_block(cma, cma)
        spatial = self.sfe_module(aft)
        motion = self.mfe_module(aft)
        return self.dig(spatial, motion)

    @staticmethod
    def apply_smartroi(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 4:
            raise ValueError(f"Expected mask shape (B,T,H,W), got {tuple(mask.shape)}")
        return feature * mask.unsqueeze(1).to(device=feature.device, dtype=feature.dtype)

    def forward(self, clip: torch.Tensor, smartroi_mask: torch.Tensor):
        feature = self.extract_backbone(clip)
        speed_feature = self.disent.speed_branch(self.apply_smartroi(feature, smartroi_mask))
        vehicle_feature = self.disent.vehicle_branch(feature)
        z_speed = speed_feature.mean(dim=(-3, -2, -1))
        z_vehicle = vehicle_feature.mean(dim=(-3, -2, -1))
        return {
            "speed_pred": self.speed_head(z_speed),
            "vehicle_pred_private": self.vehicle_head(z_vehicle),
            "vehicle_pred_adv": self.vehicle_adv_head(self.grl(z_speed)),
            "z_speed": z_speed,
            "z_vehicle": z_vehicle,
        }

    @torch.no_grad()
    def predict_speed(self, clip: torch.Tensor, smartroi_mask: torch.Tensor) -> torch.Tensor:
        self.eval()
        feature = self.extract_backbone(clip)
        speed_feature = self.disent.speed_branch(self.apply_smartroi(feature, smartroi_mask))
        z_speed = speed_feature.mean(dim=(-3, -2, -1))
        return self.speed_head(z_speed)


def dann_weight(current_iteration: int, warmup_iterations: int, max_weight: float) -> float:
    """DANN logistic schedule used by the final experiments."""
    if warmup_iterations <= 0:
        return float(max_weight)
    progress = min(1.0, max(0.0, current_iteration / warmup_iterations))
    coefficient = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
    return float(coefficient * max_weight)
