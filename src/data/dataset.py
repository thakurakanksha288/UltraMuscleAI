"""
PyTorch Datasets for the UMUD Ultrasound Architecture Project.
Supports segmentation, multi-task regression, and video burst sequence loading.
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2


class UMUDSegmentationDataset(Dataset):
    """
    Dataset for aponeurosis and fascicle segmentation.
    Loads grayscale/RGB B-mode ultrasound images and corresponding mask annotations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_col: str = "image_path",
        mask_col: str = "mask_path",
        transform: Optional[Callable] = None,
        is_multiclass: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.mask_col = mask_col
        self.transform = transform
        self.is_multiclass = is_multiclass

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = str(row[self.image_col])
        mask_path = str(row[self.mask_col])

        # Load image (B-mode ultrasound is 1-channel, load as 3-channel for pretrained backbones)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Ultrasound image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Binarize or preserve class values
        if not self.is_multiclass:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.int64)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            if not self.is_multiclass and not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            elif self.is_multiclass and not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.long)
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            if not self.is_multiclass:
                mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            else:
                mask = torch.tensor(mask, dtype=torch.long)

        return {
            "image": image,
            "mask": mask,
            "image_id": str(row.get("image_id", Path(img_path).stem)),
        }


class UMUDRegressionDataset(Dataset):
    """
    Dataset for direct and physics-informed multi-task regression.
    Outputs: [pa_deg, fl_mm, mt_mm]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_col: str = "image_path",
        target_cols: List[str] = ["pa_deg", "fl_mm", "mt_mm"],
        mask_col: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.target_cols = target_cols
        self.mask_col = mask_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = str(row[self.image_col])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Ultrasound image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = None
        if self.mask_col and self.mask_col in row and pd.notna(row[self.mask_col]):
            mask_path = str(row[self.mask_col])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        targets = torch.tensor(
            [float(row[c]) if c in row and pd.notna(row[c]) else 0.0 for c in self.target_cols],
            dtype=torch.float32,
        )

        item = {
            "image": image,
            "targets": targets,
            "image_id": str(row.get("image_id", Path(img_path).stem)),
        }
        if mask is not None:
            item["mask"] = mask
        return item


class UMUDTestDataset(Dataset):
    """
    Inference dataset for test images and consecutive video bursts.
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        transform: Optional[Callable] = None,
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Test image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        return {
            "image": image,
            "image_id": img_path.stem,
            "orig_size": (orig_h, orig_w),
            "file_path": str(img_path),
        }
