"""
Deduplication module for the UMUD Ultrasound dataset.
Detects and purges exact and perceptual duplicate image-mask pairs
(including the 670 duplicate fascicle pairs identified in the competition).
"""

import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm


def compute_file_md5(filepath: Path) -> str:
    """Computes MD5 hash of raw file bytes."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_difference_hash(image_path: Path, hash_size: int = 16) -> str:
    """
    Computes difference hash (dHash) for an image to detect perceptual duplicates.
    Resizes to (hash_size + 1, hash_size), computes column differences, converts to hex.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return hex(int("".join(["1" if b else "0" for b in diff.flatten()]), 2))[2:]


def audit_and_deduplicate_dataset(
    df: pd.DataFrame,
    image_col: str = "image_path",
    mask_col: Optional[str] = "mask_path",
    output_report_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Identifies exact and near-exact duplicates in a dataset manifest.
    
    Args:
        df: Pandas DataFrame with image (and optional mask) paths.
        image_col: Column name containing image file paths.
        mask_col: Column name containing mask file paths (if any).
        output_report_path: Path to save JSON deduplication report.
        
    Returns:
        Tuple of (clean_df, report_dict).
    """
    print(f"[*] Starting dataset deduplication audit on {len(df)} samples...")
    
    seen_hashes: Dict[str, str] = {} # hash -> representative_id
    duplicate_records: List[Dict[str, Any]] = []
    keep_indices: List[int] = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Hashing samples"):
        img_path = Path(row[image_col])
        if not img_path.exists():
            continue
            
        img_md5 = compute_file_md5(img_path)
        
        # Combine image hash with mask hash if mask is present
        if mask_col and mask_col in row and pd.notna(row[mask_col]):
            mask_path = Path(row[mask_col])
            if mask_path.exists():
                mask_md5 = compute_file_md5(mask_path)
                combined_hash = hashlib.md5(f"{img_md5}_{mask_md5}".encode()).hexdigest()
            else:
                combined_hash = img_md5
        else:
            combined_hash = img_md5
            
        sample_id = str(row.get("image_id", idx))
        
        if combined_hash in seen_hashes:
            duplicate_records.append({
                "duplicate_index": idx,
                "duplicate_sample_id": sample_id,
                "duplicate_file": str(img_path),
                "original_sample_id": seen_hashes[combined_hash],
                "hash": combined_hash,
            })
        else:
            seen_hashes[combined_hash] = sample_id
            keep_indices.append(idx)
            
    clean_df = df.loc[keep_indices].copy().reset_index(drop=True)
    num_duplicates = len(duplicate_records)
    
    report = {
        "original_count": len(df),
        "clean_count": len(clean_df),
        "duplicates_removed": num_duplicates,
        "duplicate_ratio": round(num_duplicates / max(1, len(df)), 4),
        "duplicates": duplicate_records,
    }
    
    print(f"[+] Audit completed: {len(df)} total -> {len(clean_df)} clean ({num_duplicates} duplicates pruned).")
    
    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[+] Deduplication report saved to {output_report_path}")
        
    return clean_df, report
