"""
Synthetic Ultrasound Benchmark Generator for UltraMuscleAI offline testing.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2


def generate_synthetic_ultrasound_benchmark(
    data_dir: Path,
    n_samples: int = 50,
    include_duplicates: int = 6,
) -> pd.DataFrame:
    """
    Generates synthetic B-mode musculoskeletal ultrasound images,
    aponeurosis masks, and fascicle lines for offline testing.
    """
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

        # Base speckle noise
        base = np.random.rayleigh(scale=35, size=(h, w)).astype(np.float32)

        # Superficial aponeurosis
        sup_y = int(np.random.normal(90, 8))
        sup_thick = np.random.randint(8, 14)
        cv2.line(base, (0, sup_y), (w, sup_y), 210, sup_thick)

        # Deep aponeurosis
        deep_y = int(np.random.normal(410, 10))
        deep_thick = np.random.randint(10, 16)
        cv2.line(base, (0, deep_y), (w, deep_y), 220, deep_thick)

        # Fascicle parameters
        pa_deg = float(np.random.uniform(14.0, 26.0))
        rad = np.radians(pa_deg)
        mt_px = float(deep_y - sup_y)
        fl_px = float(mt_px / np.sin(rad))

        px_per_mm = 10.0
        for tick_y in range(50, h - 50, 50):
            cv2.line(base, (w - 15, tick_y), (w - 2, tick_y), 240, 2)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[sup_y - sup_thick // 2 : sup_y + sup_thick // 2, :] = 1
        mask[deep_y - deep_thick // 2 : deep_y + deep_thick // 2, :] = 2

        for f_offset in range(50, w - 50, 80):
            start_x = f_offset
            start_y = deep_y
            end_x = int(start_x + mt_px / np.tan(rad))
            end_y = sup_y
            cv2.line(base, (start_x, start_y), (end_x, end_y), 180, 2)
            cv2.line(mask, (start_x, start_y), (end_x, end_y), 3, 2)

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
