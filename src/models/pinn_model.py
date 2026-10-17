"""
Physics-Informed Neural Network (PINN) for B-Mode Ultrasound Muscle Architecture.
Predicts: Pennation Angle (pa_deg), Fascicle Length (fl_mm), Muscle Thickness (mt_mm).
Enforces anatomical geometric constraint: MT ≈ FL * sin(PA)
"""

import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Residual convolutional block with depthwise-separable options for ultrasound feature extraction."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act2(x + res)


class UltraMusclePINN(nn.Module):
    """
    Multi-Scale Physics-Informed Neural Network for Ultrasound Metrology.
    Accepts 3-channel or 1-channel B-mode ultrasound frames.
    Outputs: [pa_deg, fl_mm, mt_mm]
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Multi-scale stages
        c = base_channels
        self.stage1 = nn.Sequential(ConvBlock(c, c), ConvBlock(c, c))
        self.stage2 = nn.Sequential(ConvBlock(c, c * 2, stride=2), ConvBlock(c * 2, c * 2))
        self.stage3 = nn.Sequential(ConvBlock(c * 2, c * 4, stride=2), ConvBlock(c * 4, c * 4))
        self.stage4 = nn.Sequential(ConvBlock(c * 4, c * 8, stride=2), ConvBlock(c * 8, c * 8))
        
        feature_dim = c * 8
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Shared latent representation
        self.fc_shared = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Task-specific regression heads
        # Head 1: Pennation Angle (degrees, typically 10° - 40°)
        self.head_pa = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Head 2: Fascicle Length (mm, typically 40mm - 140mm)
        self.head_fl = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Head 3: Muscle Thickness (mm, typically 10mm - 50mm)
        self.head_mt = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.pool(x).flatten(1)
        feat = self.fc_shared(x)
        
        pa = F.relu(self.head_pa(feat)) + 5.0   # Enforce positive angle (minimum ~5 deg)
        fl = F.relu(self.head_fl(feat)) + 10.0  # Enforce positive length (minimum ~10 mm)
        mt = F.relu(self.head_mt(feat)) + 3.0   # Enforce positive thickness (minimum ~3 mm)
        
        return torch.cat([pa, fl, mt], dim=1)


class PhysicsInformedUMUDLoss(nn.Module):
    """
    Combined loss function for the UMUD competition:
    1. Normalized Mean Absolute Error (NMAE) across pa_deg, fl_mm, mt_mm
    2. Physics-Informed Anatomical Regularization: |sin(theta_PA) - MT / FL|
    """
    def __init__(
        self,
        tol_pa: float = 5.0,
        tol_fl: float = 10.0,
        tol_mt: float = 2.0,
        lambda_pinn: float = 0.3
    ):
        super().__init__()
        self.tol_pa = tol_pa
        self.tol_fl = tol_fl
        self.tol_mt = tol_mt
        self.lambda_pinn = lambda_pinn

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Predictions
        pred_pa = pred[:, 0]
        pred_fl = pred[:, 1]
        pred_mt = pred[:, 2]
        
        # Ground Truth
        true_pa = target[:, 0]
        true_fl = target[:, 1]
        true_mt = target[:, 2]
        
        # 1. Normalized L1 error (competition metric proxy)
        err_pa = torch.abs(pred_pa - true_pa) / self.tol_pa
        err_fl = torch.abs(pred_fl - true_fl) / self.tol_fl
        err_mt = torch.abs(pred_mt - true_mt) / self.tol_mt
        nmae_loss = (err_pa + err_fl + err_mt).mean() / 3.0
        
        # 2. Physics-Informed Anatomical Consistency Loss
        # In skeletal muscle architecture: sin(theta) ≈ MT / FL
        # theta in radians:
        rad_pa = pred_pa * (math.pi / 180.0)
        sin_pa = torch.sin(rad_pa)
        ratio_mt_fl = pred_mt / (pred_fl + 1e-6)
        
        pinn_loss = torch.abs(sin_pa - ratio_mt_fl).mean()
        
        total_loss = nmae_loss + (self.lambda_pinn * pinn_loss)
        
        metrics = {
            "loss_total": total_loss.item(),
            "loss_nmae": nmae_loss.item(),
            "loss_pinn": pinn_loss.item(),
            "mae_pa": torch.abs(pred_pa - true_pa).mean().item(),
            "mae_fl": torch.abs(pred_fl - true_fl).mean().item(),
            "mae_mt": torch.abs(pred_mt - true_mt).mean().item(),
        }
        return total_loss, metrics
