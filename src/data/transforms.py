"""
Ultrasound-specific augmentation and preprocessing transforms for UltraMuscleAI.
Designed to handle B-mode speckle noise, acoustic dropouts, probe compression,
and angle-preserving geometric augmentations.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_ultrasound_transforms(
    img_size: Tuple[int, int] = (512, 512),
    is_train: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Creates Albumentations transform pipeline tailored for B-mode musculoskeletal ultrasound.
    
    Args:
        img_size: Target (height, width) for model inputs.
        is_train: Whether to include training data augmentations.
        mean: Normalization channel means.
        std: Normalization channel standard deviations.
    
    Returns:
        A.Compose pipeline.
    """
    if is_train:
        return A.Compose(
            [
                A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR),
                # 1. Speckle noise & contrast enhancement
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.7),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.2,
                    p=0.6
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                
                # 2. Ultrasound acoustic artifacts simulation
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5, p=0.5),
                        A.MedianBlur(blur_limit=5, p=0.5),
                        A.GaussianBlur(blur_limit=5, p=0.5),
                    ],
                    p=0.4
                ),
                # Simulates probe contact acoustic dropouts / shadowing
                A.CoarseDropout(
                    max_holes=6,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.3
                ),
                
                # 3. Mild biomechanical tissue deformation (probe pressure)
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=20,
                    p=0.3
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.06,
                    scale_limit=0.08,
                    rotate_limit=8, # Limit rotation to avoid invalidating pennation angles
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
                
                # 4. Normalization and Tensor conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def apply_angle_preserving_horizontal_flip(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    pa_deg: Optional[float] = None,
    fl_mm: Optional[float] = None,
    mt_mm: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Optional[float]]]:
    """
    Applies horizontal reflection while mathematically inverting the fascicle pennation angle.
    When an ultrasound image is flipped horizontally:
        - Image width reflects: x' = W - 1 - x
        - Fascicle slope inverts: theta_new = 180 - theta_old (or -theta depending on reference axis)
        - Fascicle Length and Muscle Thickness remain invariant under reflection.
    """
    flipped_img = cv2.flip(image, 1)
    flipped_mask = cv2.flip(mask, 1) if mask is not None else None
    
    transformed_targets = {
        "pa_deg": (180.0 - pa_deg) if pa_deg is not None else None,
        "fl_mm": fl_mm,
        "mt_mm": mt_mm,
    }
    return flipped_img, flipped_mask, transformed_targets
