"""
Master data preprocessing and ingestion pipeline for UltraMuscleAI.
Orchestrates raw data discovery, synthetic demonstration generation,
deduplication audit, scale calibration, and stratified group fold splitting.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import cv2

from src.data.deduplicate import audit_and_deduplicate_dataset
from src.data.splits import create_stratified_group_folds
from src.metrology.scale_detector import UltrasoundScaleDetector


def generate_synthetic_ultrasound_benchmark(
    data_dir: Path,
    n_samples: int = 50,
    include_duplicates: int = 6,
) -> pd.DataFrame:
    """
    Generates realistic synthetic B-mode musculoskeletal ultrasound images,
    aponeurosis masks, and fascicle lines to verify the entire pipeline end-to-end
    before downloading the complete competition archive.
    """
    print(f"[*] Generating {n_samples} benchmark ultrasound samples at: {data_dir}")
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    records = []
    h, w = 512, 512

    for i in range(n_samples):
        subj_id = f"subj_{(i // 5):02d}"
        seq_id = f"seq_{(i % 5):02d}"
        img_id = f"{subj_id}_{seq_id}_frame_{i:03d}"

        # 1. Base speckle noise
        base = np.random.rayleigh(scale=35, size=(h, w)).astype(np.float32)

        # 2. Draw superficial aponeurosis (top bright band)
        sup_y = int(np.random.normal(90, 8))
        sup_thick = np.random.randint(8, 14)
        cv2.line(base, (0, sup_y), (w, sup_y), 210, sup_thick)

        # 3. Draw deep aponeurosis (bottom bright band)
        deep_y = int(np.random.normal(410, 10))
        deep_thick = np.random.randint(10, 16)
        cv2.line(base, (0, deep_y), (w, deep_y), 220, deep_thick)

        # 4. Draw fascicles (hyperechoic oblique lines)
        pa_deg = float(np.random.uniform(14.0, 26.0))
        rad = np.radians(pa_deg)
        mt_px = float(deep_y - sup_y)
        # Analytical fascicle length in pixels: FL = MT / sin(theta)
        fl_px = float(mt_px / np.sin(rad))

        # Add lateral probe calibration ticks (every 50px = 5mm -> 10 px/mm)
        px_per_mm = 10.0
        for tick_y in range(50, h - 50, 50):
            cv2.line(base, (w - 15, tick_y), (w - 2, tick_y), 240, 2)

        # Draw fascicle lines
        mask = np.zeros((h, w), dtype=np.uint8)
        # Class 1: superficial aponeurosis, Class 2: deep aponeurosis
        mask[sup_y - sup_thick // 2 : sup_y + sup_thick // 2, :] = 1
        mask[deep_y - deep_thick // 2 : deep_y + deep_thick // 2, :] = 2

        # Draw 5 parallel fascicles
        for f_offset in range(50, w - 50, 80):
            start_x = f_offset
            start_y = deep_y
            end_x = int(start_x + mt_px / np.tan(rad))
            end_y = sup_y
            cv2.line(base, (start_x, start_y), (end_x, end_y), 180, 2)
            cv2.line(mask, (start_x, start_y), (end_x, end_y), 3, 2)

        # Speckle blur & clipping
        base = cv2.GaussianBlur(base, (3, 3), 0.8)
        base = np.clip(base, 0, 255).astype(np.uint8)

        img_path = images_dir / f"{img_id}.png"
        mask_path = masks_dir / f"{img_id}_mask.png"

        cv2.imwrite(str(img_path), base)
        cv2.imwrite(str(mask_path), mask)

        mt_mm = round(mt_px / px_per_mm, 2)
        fl_mm = round(fl_px / px_per_mm, 2)

        records.append({
            "image_id": img_id,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "pa_deg": round(pa_deg, 2),
            "fl_mm": fl_mm,
            "mt_mm": mt_mm,
            "subject_id": subj_id,
            "sequence_id": seq_id,
        })

    # Intentionally inject duplicates to test deduplication logic
    for dup_i in range(include_duplicates):
        src = records[dup_i]
        dup_id = f"dup_{src['image_id']}"
        records.append({
            "image_id": dup_id,
            "image_path": src["image_path"],
            "mask_path": src["mask_path"],
            "pa_deg": src["pa_deg"],
            "fl_mm": src["fl_mm"],
            "mt_mm": src["mt_mm"],
            "subject_id": src["subject_id"],
            "sequence_id": src["sequence_id"],
        })

    df = pd.DataFrame(records)
    csv_path = data_dir / "train_manifest.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Created {len(df)} samples ({include_duplicates} duplicates) -> {csv_path}")
    return df


def discover_raw_dataset(raw_dir: Path) -> Optional[pd.DataFrame]:
    """Scans raw_dir for competition images, masks, and labels."""
    print(f"[*] Scanning for dataset in {raw_dir}...")
    
    # 1. Check for standard competition CSVs
    candidates = ["train.csv", "train_manifest.csv", "labels.csv", "annotations.csv"]
    for c in candidates:
        csv_file = raw_dir / c
        if csv_file.exists():
            print(f"[+] Found existing dataset manifest: {csv_file}")
            return pd.read_csv(csv_file)

    # 2. Check for image directories
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_files = [p for p in raw_dir.rglob("*") if p.suffix.lower() in image_exts]

    if not image_files:
        return None

    print(f"[+] Discovered {len(image_files)} image files in {raw_dir}")
    records = []
    for img_p in image_files:
        if "mask" in img_p.stem.lower():
            continue
        mask_candidates = [
            img_p.parent / f"{img_p.stem}_mask{img_p.suffix}",
            raw_dir / "masks" / f"{img_p.stem}_mask{img_p.suffix}",
            raw_dir / "masks" / f"{img_p.name}",
        ]
        mask_path = next((str(m) for m in mask_candidates if m.exists()), None)
        records.append({
            "image_id": img_p.stem,
            "image_path": str(img_p),
            "mask_path": mask_path,
        })
    return pd.DataFrame(records)


def run_preprocessing_pipeline(
    raw_dir: Path = Path("D:/UltraMuscleAI/data/raw"),
    processed_dir: Path = Path("D:/UltraMuscleAI/data/processed"),
    n_splits: int = 5,
    auto_generate_sample: bool = True,
) -> Dict[str, Any]:
    """
    Main preprocessing driver.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" ULTRA-MUSCLE-AI: DATA PREPROCESSING & AUDITING ENGINE ")
    print("=" * 65)

    df = discover_raw_dataset(raw_dir)

    if df is None or len(df) == 0:
        if auto_generate_sample:
            print("\n[!] No dataset found in data/raw.")
            print("[*] Generating high-fidelity synthetic benchmark dataset for verification...")
            df = generate_synthetic_ultrasound_benchmark(raw_dir, n_samples=50, include_duplicates=6)
        else:
            raise FileNotFoundError(
                f"No images or manifests found in {raw_dir}. "
                "Please place competition data in data/raw or pass --create-sample-data"
            )

    # 1. Audit and deduplicate dataset
    report_path = processed_dir / "deduplication_report.json"
    clean_df, dedup_report = audit_and_deduplicate_dataset(
        df=df,
        image_col="image_path",
        mask_col="mask_path" if "mask_path" in df.columns else None,
        output_report_path=report_path,
    )

    # 2. Automated probe scale calibration check
    scale_detector = UltrasoundScaleDetector()
    scale_results = []
    print("\n[*] Calibrating ultrasound depth and scale on image samples...")
    for _, row in clean_df.head(20).iterrows():
        img = cv2.imread(str(row["image_path"]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            calib = scale_detector.detect_scale_from_image(img)
            scale_results.append(calib)

    mean_px_per_mm = float(np.mean([s["px_per_mm"] for s in scale_results])) if scale_results else 10.0
    scale_meta = {
        "mean_px_per_mm": round(mean_px_per_mm, 3),
        "mean_mm_per_px": round(1.0 / mean_px_per_mm, 5),
        "calibrated_samples": len(scale_results),
    }
    scale_path = processed_dir / "scale_calibration.json"
    with open(scale_path, "w") as f:
        json.dump(scale_meta, f, indent=2)
    print(f"[+] Scale calibration saved to {scale_path} (Average: {mean_px_per_mm:.2f} px/mm)")

    # 3. Create leak-free stratified group folds
    folds_df = create_stratified_group_folds(
        df=clean_df,
        n_splits=n_splits,
        target_cols=["pa_deg", "fl_mm", "mt_mm"],
    )

    # 4. Save clean processed manifests
    train_clean_path = processed_dir / "train_clean.csv"
    folds_path = processed_dir / "folds.csv"

    clean_df.to_csv(train_clean_path, index=False)
    folds_df.to_csv(folds_path, index=False)

    print("\n" + "=" * 65)
    print(" PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY ")
    print("=" * 65)
    print(f" - Clean dataset manifest: {train_clean_path}")
    print(f" - 5-Fold stratified splits: {folds_path}")
    print(f" - Deduplication report:    {report_path}")
    print(f" - Probe scale metadata:   {scale_path}")
    print("=" * 65)

    return {
        "clean_samples": len(clean_df),
        "duplicates_removed": dedup_report["duplicates_removed"],
        "n_folds": n_splits,
        "scale_meta": scale_meta,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UltraMuscleAI Data Preprocessing Pipeline")
    parser.add_argument("--raw-dir", type=str, default="D:/UltraMuscleAI/data/raw")
    parser.add_argument("--processed-dir", type=str, default="D:/UltraMuscleAI/data/processed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--create-sample-data", action="store_true", default=True)

    args = parser.parse_args()
    run_preprocessing_pipeline(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        n_splits=args.folds,
        auto_generate_sample=args.create_sample_data,
    )
