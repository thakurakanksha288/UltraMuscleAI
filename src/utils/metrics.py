"""
Official evaluation metrics for the UMUD Ultrasound Architecture Challenge.
Calculates Normalized Mean Absolute Error (UMUD Score), individual parameter errors,
and biomechanical agreement metrics.
"""

from typing import Dict, Union, Tuple, Optional
import numpy as np
import pandas as pd
import torch


# Default clinical tolerance scales for normalization (degrees and mm)
DEFAULT_TOLERANCES: Dict[str, float] = {
    "pa_deg": 5.0,  # Degrees
    "fl_mm": 10.0,  # Millimeters
    "mt_mm": 2.0,   # Millimeters
}

# Official Kaggle Competition Dynamic Range Normalization
LEADERBOARD_RANGES: Dict[str, float] = {
    "pa_deg": 40.0,   # Test range: 5.0 to 45.0 degrees
    "fl_mm": 170.0,   # Test range: 30.0 to 200.0 mm
    "mt_mm": 40.0,    # Test range: 10.0 to 50.0 mm
}


def compute_leaderboard_score(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """
    Computes the official Kaggle Leaderboard Normalized Score:
        Score = (1/3) * ( MAE_PA / 40.0 + MAE_FL / 170.0 + MAE_MT / 40.0 )
    """
    return compute_umud_score(y_true, y_pred, tolerances=LEADERBOARD_RANGES)


def compute_umud_score(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    tolerances: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Computes the official normalized UMUD Score across the three parameters:
        Index 0: pa_deg (Pennation Angle in degrees)
        Index 1: fl_mm  (Fascicle Length in mm)
        Index 2: mt_mm  (Muscle Thickness in mm)
        
    Formula:
        UMUD_Score = Mean( (1/3) * sum( |pred_j - true_j| / Tol_j ) )
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    tols = tolerances or DEFAULT_TOLERANCES
    t_pa = tols["pa_deg"]
    t_fl = tols["fl_mm"]
    t_mt = tols["mt_mm"]

    # Absolute errors
    abs_pa = np.abs(y_pred[:, 0] - y_true[:, 0])
    abs_fl = np.abs(y_pred[:, 1] - y_true[:, 1])
    abs_mt = np.abs(y_pred[:, 2] - y_true[:, 2])

    # Mean absolute errors
    mae_pa = float(np.mean(abs_pa))
    mae_fl = float(np.mean(abs_fl))
    mae_mt = float(np.mean(abs_mt))

    # Normalized errors
    norm_pa = mae_pa / t_pa
    norm_fl = mae_fl / t_fl
    norm_mt = mae_mt / t_mt

    umud_score = float((norm_pa + norm_fl + norm_mt) / 3.0)

    return {
        "umud_score": umud_score,
        "mae_pa_deg": mae_pa,
        "mae_fl_mm": mae_fl,
        "mae_mt_mm": mae_mt,
        "norm_pa": norm_pa,
        "norm_fl": norm_fl,
        "norm_mt": norm_mt,
    }


def compute_segmentation_dice_iou(
    y_pred_mask: np.ndarray,
    y_true_mask: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """Computes Dice Coefficient and Intersection over Union (IoU) for binary masks."""
    pred_bin = (y_pred_mask > 0.5).astype(np.float32)
    true_bin = (y_true_mask > 0.5).astype(np.float32)

    intersection = np.sum(pred_bin * true_bin)
    union = np.sum(pred_bin) + np.sum(true_bin) - intersection
    iou = float((intersection + eps) / (union + eps))
    dice = float((2.0 * intersection + eps) / (np.sum(pred_bin) + np.sum(true_bin) + eps))
    return dice, iou
