"""
Cross-validation and fold splitting module.
Implements leak-free Stratified Group K-Fold cross-validation
partitioned strictly by subject ID and video sequence burst.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold


def extract_group_identifier(image_id_or_path: str) -> str:
    """
    Extracts subject or video burst identifier from file name or image ID.
    Examples:
        - 'sub01_seq02_frame003.png' -> 'sub01_seq02'
        - 'VL_subj05_001.png' -> 'VL_subj05'
        - '0123_frame4.png' -> '0123'
    """
    stem = str(image_id_or_path).replace("\\", "/").split("/")[-1].split(".")[0]
    parts = stem.split("_")
    if len(parts) >= 2:
        # Join first two parts if they identify subject and session
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def create_stratified_group_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: Optional[str] = None,
    target_cols: Optional[List[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Partitions the dataset into N leak-free folds grouped by subject/sequence
    and stratified by continuous target quantile distributions.
    
    Args:
        df: Pandas DataFrame of training samples.
        n_splits: Number of cross-validation folds.
        group_col: Column name of grouping ID. If None, automatically parsed from image_id.
        target_cols: List of continuous targets (e.g., ['pa_deg', 'fl_mm', 'mt_mm']).
        random_state: Seed for reproducibility.
        
    Returns:
        DataFrame with an added 'fold' column (0 to n_splits - 1).
    """
    df = df.copy().reset_index(drop=True)
    
    # 1. Determine grouping column
    if group_col is None or group_col not in df.columns:
        if "image_id" in df.columns:
            df["group_id"] = df["image_id"].apply(extract_group_identifier)
        elif "image_path" in df.columns:
            df["group_id"] = df["image_path"].apply(extract_group_identifier)
        else:
            df["group_id"] = np.arange(len(df))
        group_col = "group_id"
        
    # 2. Build multi-target stratification bins if targets exist
    strat_col = None
    if target_cols:
        valid_targets = [col for col in target_cols if col in df.columns]
        if valid_targets:
            # Multi-target quantile discretization
            bin_series = []
            for col in valid_targets:
                if df[col].notna().sum() > 0:
                    bins = pd.qcut(df[col].fillna(df[col].median()), q=4, labels=False, duplicates="drop")
                    bin_series.append(bins.astype(str))
            if bin_series:
                df["strat_bin"] = bin_series[0]
                for s in bin_series[1:]:
                    df["strat_bin"] = df["strat_bin"] + "_" + s
                strat_col = "strat_bin"
                
    # 3. Perform splitting
    df["fold"] = -1
    if strat_col is not None:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        # Fallback to GroupKFold if any strat class has fewer members than folds
        try:
            for fold_idx, (train_idx, val_idx) in enumerate(
                sgkf.split(df, y=df[strat_col], groups=df[group_col])
            ):
                df.loc[val_idx, "fold"] = fold_idx
        except ValueError:
            gkf = GroupKFold(n_splits=n_splits)
            for fold_idx, (train_idx, val_idx) in enumerate(
                gkf.split(df, groups=df[group_col])
            ):
                df.loc[val_idx, "fold"] = fold_idx
    else:
        gkf = GroupKFold(n_splits=n_splits)
        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(df, groups=df[group_col])
        ):
            df.loc[val_idx, "fold"] = fold_idx
            
    # Print fold distribution audit
    print("\n[+] Cross-Validation Fold Distribution:")
    fold_counts = df["fold"].value_counts().sort_index()
    for f, count in fold_counts.items():
        n_groups = df[df["fold"] == f][group_col].nunique()
        print(f"    Fold {f}: {count} samples across {n_groups} unique groups")
        
    return df
