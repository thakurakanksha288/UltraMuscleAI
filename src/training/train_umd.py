"""
UltraMuscle-AI hybrid metrology pipeline (revised).

Same feature extractors and models as before (28-dim + 34-dim structural features,
HistGradientBoosting + ExtraTrees, physics coupling FL ~ MT / sin(PA)).

What changed vs. the previous version
  * Submission is written COMMA separated (the old ';' made Kaggle fail with
    "ID column image_id not found"), then read back and verified.
  * Submission goes to <output_dir>/submissions/ with a timestamp, so a good
    earlier file is never overwritten.
  * Test-time Savgol smoothing is now OFF by default (it assumes the test order is
    temporal; smoothing similar to this made a Kaggle score worse).
  * No more silent fallback that wrote TRAIN predictions as a submission.
  * All reported numbers are labelled LOCAL out-of-fold scores. Hard-coded
    "Rank 1" comparisons were removed: a local score is not a leaderboard score.
  * Diagnostics: tuned blend weights vs a plain 50/50 blend (shows if tuning
    overfits) and per-group (.png vs other) MAE and bias (shows domain shift).
  * OOF and test predictions are assembled by the same function.
  * Optional USE_GROUP_FLAG (default False): adds an is-png feature to try.

Place this file in src/training/ (PROJECT_ROOT is resolved as parents[2]).
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.submission import save_competition_submission  # noqa: E402

# ----------------------------- CONFIG ---------------------------------------
# Blend weights from your earlier tuning (weights > 1 / < 0 extrapolate; see the
# diagnostics table printed after training to check they really help).
WEIGHTS = {"pa": 1.05618943, "mt": 0.86239162, "fl": 1.10971340, "phys": 0.35818189}
# Local score = mean of (MAE / divisor). Make sure this matches the Evaluation tab.
SCORE_DIVISORS = (40.0, 170.0, 40.0)  # (pa_deg, fl_mm, mt_mm)
SMOOTH_TEST_PREDICTIONS = False       # set True only to test it as an experiment
USE_GROUP_FLAG = False                # set True to add an is-png feature, then compare OOF
# -----------------------------------------------------------------------------

GABOR_ANGLES = [10.0, 18.0, 25.0, 32.0]
GABOR_KERNELS = [
    cv2.getGaborKernel((15, 15), 3.5, np.radians(th), 7.0, 0.5, 0, ktype=cv2.CV_32F)
    for th in GABOR_ANGLES
]


def extract_structural_metrology_features_28(img_path: str, px_per_mm: float = 10.0) -> np.ndarray:
    """Extracts 28 rich multi-scale structural, texture, and metrology features."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(28, dtype=np.float32)
    h, w = img.shape

    slice_offsets = [0.20, 0.35, 0.50, 0.65, 0.80]
    sups, deeps, thicknesses = [], [], []
    for col in slice_offsets:
        x_start = max(0, int((col - 0.05) * w))
        x_end = min(w, int((col + 0.05) * w))
        col_proj = np.mean(img[:, x_start:x_end], axis=1)
        s_y = float(np.argmax(col_proj[:int(h * 0.45)])) if h > 30 else 10.0
        d_offset = int(h * 0.55)
        d_y = float(d_offset + np.argmax(col_proj[d_offset:])) if d_offset < h else float(h - 10)
        if d_y - s_y < 20:
            s_y, d_y = float(0.25 * h), float(0.75 * h)
        sups.append(s_y)
        deeps.append(d_y)
        thicknesses.append(d_y - s_y)

    mean_sup = float(np.mean(sups))
    mean_deep = float(np.mean(deeps))
    mean_thick_px = float(np.mean(thicknesses))
    std_thick_px = float(np.std(thicknesses))
    min_thick_px = float(np.min(thicknesses))
    max_thick_px = float(np.max(thicknesses))
    thick_grad = float((thicknesses[-1] - thicknesses[0]) / (0.6 * w + 1e-4))

    sup_slope = float((sups[-1] - sups[0]) / (0.6 * w + 1e-4))
    deep_slope = float((deeps[-1] - deeps[0]) / (0.6 * w + 1e-4))
    sup_deep_angle = float(np.degrees(np.arctan(deep_slope) - np.arctan(sup_slope)))
    raw_mt_mm = float(np.clip(mean_thick_px / px_per_mm, 10.0, 50.0))

    y_top = max(0, int(mean_sup) + 6)
    y_bot = min(h, int(mean_deep) - 6)
    x_left = min(30, w // 4)
    x_right = max(x_left + 20, w - 30)

    if y_bot > y_top + 15 and x_right > x_left + 20:
        belly_raw = img[y_top:y_bot, x_left:x_right]
        bh, bw = belly_raw.shape
        target_w = min(400, bw)
        target_h = max(20, int(bh * (target_w / bw)))
        belly = cv2.resize(belly_raw, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.float32)

        gx = cv2.Sobel(belly, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(belly, cv2.CV_32F, 0, 1, ksize=3)

        Jxx1 = cv2.GaussianBlur(gx * gx, (7, 7), 2.5)
        Jyy1 = cv2.GaussianBlur(gy * gy, (7, 7), 2.5)
        Jxy1 = cv2.GaussianBlur(gx * gy, (7, 7), 2.5)
        angle1 = 0.5 * np.arctan2(2.0 * Jxy1, Jxx1 - Jyy1)
        pa1 = 90.0 - np.degrees(np.abs(angle1))
        coh1 = np.sqrt((Jxx1 - Jyy1)**2 + 4 * Jxy1**2) / (Jxx1 + Jyy1 + 1e-5)

        valid1 = (pa1 >= 5.0) & (pa1 <= 45.0) & (coh1 > 0.08)
        pa_scale1 = float(np.average(pa1[valid1], weights=coh1[valid1])) if np.sum(valid1) > 20 else float(np.median(pa1))
        mean_coh1 = float(np.mean(coh1[valid1])) if np.sum(valid1) > 20 else 0.1

        Jxx2 = cv2.GaussianBlur(gx * gx, (11, 11), 5.0)
        Jyy2 = cv2.GaussianBlur(gy * gy, (11, 11), 5.0)
        Jxy2 = cv2.GaussianBlur(gx * gy, (11, 11), 5.0)
        angle2 = 0.5 * np.arctan2(2.0 * Jxy2, Jxx2 - Jyy2)
        pa2 = 90.0 - np.degrees(np.abs(angle2))
        coh2 = np.sqrt((Jxx2 - Jyy2)**2 + 4 * Jxy2**2) / (Jxx2 + Jyy2 + 1e-5)

        valid2 = (pa2 >= 5.0) & (pa2 <= 45.0) & (coh2 > 0.08)
        pa_scale2 = float(np.average(pa2[valid2], weights=coh2[valid2])) if np.sum(valid2) > 20 else float(np.median(pa2))
        mean_coh2 = float(np.mean(coh2[valid2])) if np.sum(valid2) > 20 else 0.1

        gabor_energies = [float(np.mean(np.abs(cv2.filter2D(belly, cv2.CV_32F, k)))) for k in GABOR_KERNELS]
        dom_gabor_angle = float(GABOR_ANGLES[int(np.argmax(gabor_energies))])

        belly_mean = float(np.mean(belly))
        belly_std = float(np.std(belly))
        belly_p25 = float(np.percentile(belly, 25))
        belly_p75 = float(np.percentile(belly, 75))
    else:
        pa_scale1 = pa_scale2 = 14.0
        mean_coh1 = mean_coh2 = 0.1
        gabor_energies = [1.0, 1.0, 1.0, 1.0]
        dom_gabor_angle = 18.0
        belly_mean, belly_std, belly_p25, belly_p75 = 100.0, 30.0, 80.0, 120.0

    raw_pa = float(np.clip(0.5 * (pa_scale1 + pa_scale2), 5.0, 45.0))
    sin_pa = float(np.sin(np.radians(raw_pa)))
    inv_sin_pa = float(1.0 / max(0.08, sin_pa))
    raw_fl_phys = float(np.clip(raw_mt_mm * inv_sin_pa, 30.0, 200.0))

    vec = [
        raw_mt_mm, mean_thick_px, std_thick_px, min_thick_px, max_thick_px, thick_grad,
        mean_sup, mean_deep, sup_slope, deep_slope, sup_deep_angle,
        pa_scale1, mean_coh1, pa_scale2, mean_coh2, raw_pa,
        sin_pa, inv_sin_pa, raw_fl_phys,
        belly_mean, belly_std, belly_p25, belly_p75,
        gabor_energies[0], gabor_energies[1], gabor_energies[2], gabor_energies[3],
        dom_gabor_angle
    ]
    return np.array(vec, dtype=np.float32)


def extract_advanced_features_34(img_path: str, px_per_mm: float = 10.0) -> np.ndarray:
    """Extracts 34 advanced features: 1D depth profilometry, contrast, and 3-scale structure tensor."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(34, dtype=np.float32)
    h, w = img.shape

    center_strip = img[:, int(0.20 * w):int(0.80 * w)]
    depth_profile = np.mean(center_strip, axis=1)
    kernel_size = max(5, int(h * 0.03)) | 1
    depth_smooth = cv2.GaussianBlur(depth_profile.reshape(-1, 1), (1, kernel_size), kernel_size / 3.0).flatten()

    top_search_start, top_search_end = int(0.08 * h), int(0.45 * h)
    bot_search_start, bot_search_end = int(0.55 * h), int(0.92 * h)

    sup_y = float(top_search_start + np.argmax(depth_smooth[top_search_start:top_search_end]))
    deep_y = float(bot_search_start + np.argmax(depth_smooth[bot_search_start:bot_search_end]))
    profile_thickness_px = max(20.0, deep_y - sup_y)
    profile_mt_mm = profile_thickness_px / px_per_mm

    sup_peak_val = float(depth_smooth[int(sup_y)])
    deep_peak_val = float(depth_smooth[int(deep_y)])
    belly_valley_val = float(np.min(depth_smooth[int(sup_y):int(deep_y)])) if deep_y > sup_y + 10 else 50.0
    contrast_ratio = (sup_peak_val + deep_peak_val) / (2.0 * max(10.0, belly_valley_val))

    slice_offsets = [0.15, 0.25, 0.38, 0.50, 0.62, 0.75, 0.85]
    sups, deeps, thicknesses = [], [], []
    for col in slice_offsets:
        xs, xe = max(0, int((col - 0.04) * w)), min(w, int((col + 0.04) * w))
        col_proj = np.mean(img[:, xs:xe], axis=1)
        sy = float(np.argmax(col_proj[:int(h * 0.45)])) if h > 30 else 10.0
        dy = float(bot_search_start + np.argmax(col_proj[bot_search_start:])) if bot_search_start < h else float(h - 10)
        if dy - sy < 20:
            sy, dy = float(0.25 * h), float(0.75 * h)
        sups.append(sy)
        deeps.append(dy)
        thicknesses.append(dy - sy)

    slice_mt_mm = float(np.mean(thicknesses) / px_per_mm)
    slice_std_mm = float(np.std(thicknesses) / px_per_mm)
    sup_slope = float((sups[-1] - sups[0]) / (0.7 * w + 1e-4))
    deep_slope = float((deeps[-1] - deeps[0]) / (0.7 * w + 1e-4))
    sup_angle_deg = float(np.degrees(np.arctan(sup_slope)))
    deep_angle_deg = float(np.degrees(np.arctan(deep_slope)))

    yt = max(0, int(np.mean(sups)) + 6)
    yb = min(h, int(np.mean(deeps)) - 6)
    xl, xr = min(30, w // 4), max(w // 4 + 20, w - 30)

    if yb > yt + 15 and xr > xl + 20:
        belly = img[yt:yb, xl:xr]
        bh, bw = belly.shape
        target_w = min(350, bw)
        target_h = max(20, int(bh * (target_w / bw)))
        b_down = cv2.resize(belly, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.float32)

        gx = cv2.Sobel(b_down, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(b_down, cv2.CV_32F, 0, 1, ksize=3)

        pa_scales = []
        coh_scales = []
        for s in [2.0, 4.0, 6.0]:
            k = int(s * 3) | 1
            jxx = cv2.GaussianBlur(gx * gx, (k, k), s)
            jyy = cv2.GaussianBlur(gy * gy, (k, k), s)
            jxy = cv2.GaussianBlur(gx * gy, (k, k), s)
            ang = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
            pa = 90.0 - np.degrees(np.abs(ang))
            coh = np.sqrt((jxx - jyy)**2 + 4 * jxy**2) / (jxx + jyy + 1e-5)
            val = (pa >= 5.0) & (pa <= 45.0) & (coh > 0.08)
            pa_scales.append(float(np.average(pa[val], weights=coh[val])) if np.sum(val) > 20 else float(np.median(pa)))
            coh_scales.append(float(np.mean(coh[val])) if np.sum(val) > 20 else 0.1)

        b_mean = float(np.mean(b_down))
        b_std = float(np.std(b_down))
        b_p10 = float(np.percentile(b_down, 10))
        b_p50 = float(np.percentile(b_down, 50))
        b_p90 = float(np.percentile(b_down, 90))
    else:
        pa_scales = [14.0, 14.0, 14.0]
        coh_scales = [0.1, 0.1, 0.1]
        b_mean, b_std, b_p10, b_p50, b_p90 = 100.0, 30.0, 60.0, 100.0, 140.0

    raw_pa = float(np.clip(0.4 * pa_scales[0] + 0.4 * pa_scales[1] + 0.2 * pa_scales[2], 5.0, 45.0))
    rel_pa = float(np.clip(raw_pa - deep_angle_deg, 5.0, 45.0))
    sin_pa = float(np.sin(np.radians(raw_pa)))
    sin_rel_pa = float(np.sin(np.radians(rel_pa)))

    best_mt_mm = float(np.clip(0.60 * slice_mt_mm + 0.40 * profile_mt_mm, 10.0, 50.0))
    fl_phys = float(np.clip(best_mt_mm / max(0.08, sin_pa), 30.0, 200.0))
    fl_phys_rel = float(np.clip(best_mt_mm / max(0.08, sin_rel_pa), 30.0, 200.0))

    features = [
        best_mt_mm, profile_mt_mm, slice_mt_mm, slice_std_mm,
        sup_y, deep_y, sup_peak_val, deep_peak_val, belly_valley_val, contrast_ratio,
        sup_slope, deep_slope, sup_angle_deg, deep_angle_deg,
        pa_scales[0], coh_scales[0], pa_scales[1], coh_scales[1], pa_scales[2], coh_scales[2],
        raw_pa, rel_pa, sin_pa, sin_rel_pa,
        fl_phys, fl_phys_rel,
        b_mean, b_std, b_p10, b_p50, b_p90,
        float(w), float(h), float(w / h)
    ]
    return np.array(features, dtype=np.float32)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_or_extract(cache: Path, paths, fn: Callable[[str], np.ndarray], label: str) -> np.ndarray:
    """Load cached features, or extract them from images and cache them."""
    if cache.exists():
        return np.load(cache)
    if paths is None or len(paths) == 0:
        raise FileNotFoundError(f"{cache} is missing and no image paths are available to rebuild it.")
    print(f"[*] Extracting {label} for {len(paths)} images...")
    X = np.array([fn(p) for p in paths], dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, X)
    return X


def is_png(frame: pd.DataFrame) -> np.ndarray:
    """1.0 for .png images, 0.0 otherwise (uses image_path if present, else image_id)."""
    col = "image_path" if "image_path" in frame.columns else "image_id"
    return frame[col].astype(str).str.lower().str.endswith(".png").values.astype(np.float32)


def make_hgbr(leaf_nodes: int, min_leaf: int, l2: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=300, learning_rate=0.035,
        max_leaf_nodes=leaf_nodes, min_samples_leaf=min_leaf, l2_regularization=l2, random_state=42,
    )


def make_et() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(n_estimators=100, max_depth=13, min_samples_split=4, random_state=42, n_jobs=4)


def assemble(raw: Dict[str, Tuple[np.ndarray, np.ndarray]], wts: Dict[str, float]):
    """Turn raw (hgbr, extra-trees) predictions into final (pa, fl, mt). Used for OOF and test."""
    pa = np.clip(wts["pa"] * raw["pa"][0] + (1.0 - wts["pa"]) * raw["pa"][1], 5.0, 45.0)
    mt = np.clip(wts["mt"] * raw["mt"][0] + (1.0 - wts["mt"]) * raw["mt"][1], 10.0, 50.0)
    fl_raw = wts["fl"] * raw["fl"][0] + (1.0 - wts["fl"]) * raw["fl"][1]
    fl_phys = mt / np.sin(np.radians(pa))
    fl = np.clip((1.0 - wts["phys"]) * fl_raw + wts["phys"] * fl_phys, 30.0, 200.0)
    return pa, fl, mt


def local_score(pa, fl, mt, Y) -> float:
    d_pa, d_fl, d_mt = SCORE_DIVISORS
    return float((np.mean(np.abs(pa - Y[:, 0])) / d_pa
                  + np.mean(np.abs(fl - Y[:, 1])) / d_fl
                  + np.mean(np.abs(mt - Y[:, 2])) / d_mt) / 3.0)


def write_submission(sub_df: pd.DataFrame, csv_path: Path, xlsx_path: Path) -> Dict[str, Any]:
    """Write a comma-separated submission, keep 3 decimals, and verify it by reading it back."""
    sub_df = sub_df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy()
    if sub_df.isna().any().any():
        raise ValueError("Submission contains NaN values.")
    if not sub_df["image_id"].is_unique:
        raise ValueError("Submission contains duplicate image_id values.")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    info: Dict[str, Any] = {}
    try:  # project helper (also writes the Excel copy)
        info = save_competition_submission(
            df=sub_df, output_csv_path=csv_path, output_excel_path=xlsx_path,
            delimiter=",", round_decimals=3,
        ) or {}
    except Exception as exc:  # the helper must never decide the final CSV format
        print(f"[!] save_competition_submission failed ({exc}); writing CSV directly.")

    # Always write the final CSV ourselves so the format is guaranteed.
    sub_df.to_csv(csv_path, index=False, float_format="%.3f")

    with open(csv_path, "r", newline="") as f:
        header = f.readline().strip()
    if header != "image_id,pa_deg,fl_mm,mt_mm":
        raise RuntimeError(f"Unexpected submission header: {header!r}")
    check = pd.read_csv(csv_path)
    if check.shape != (len(sub_df), 4):
        raise RuntimeError(f"Unexpected submission shape after re-read: {check.shape}")
    print(f"[+] Verified {csv_path.name}: {check.shape[0]} rows, header '{header}'")
    return info


def run_high_precision_hybrid_training(
    folds_csv: Path = Path("D:/UltraMuscleAI/data/processed/folds.csv"),
    output_dir: Path = Path("D:/UltraMuscleAI"),
    sample_sub: Optional[Path] = None,
) -> Dict[str, Any]:
    """5-fold CV of the hybrid metrology models, then test inference and a verified submission."""
    print("=" * 72)
    print(" ULTRAMUSCLE-AI: HYBRID METROLOGY PIPELINE (revised)")
    print(" All scores below are LOCAL out-of-fold estimates, not leaderboard scores.")
    print("=" * 72)

    if not folds_csv.exists():
        raise FileNotFoundError(f"Folds file not found at {folds_csv}")

    df = pd.read_csv(folds_csv)
    n = len(df)
    proc = output_dir / "data" / "processed"

    X1 = load_or_extract(proc / "train_features.npy", df["image_path"], extract_structural_metrology_features_28, "28-dim features")
    X2 = load_or_extract(proc / "train_features_v2.npy", df["image_path"], extract_advanced_features_34, "34-dim features")
    if USE_GROUP_FLAG:
        X1 = np.hstack([X1, is_png(df)[:, None]])
    X_combo = np.hstack([X1, X2])
    Y = np.column_stack([df["pa_deg"].values, df["fl_mm"].values, df["mt_mm"].values]).astype(np.float32)
    print(f"[+] Feature matrix: {X_combo.shape} (group flag: {USE_GROUP_FLAG})")

    # model spec: key -> (feature matrix, target column, (max_leaf_nodes, min_samples_leaf, l2))
    specs = {
        "pa": (X1, 0, (31, 12, 1.0)),
        "mt": (X_combo, 2, (40, 10, 0.5)),
        "fl": (X_combo, 1, (40, 10, 0.5)),
    }
    oof_raw = {k: (np.zeros(n), np.zeros(n)) for k in specs}
    models: Dict[str, list] = {k: [] for k in specs}

    fold_ids = sorted(df["fold"].unique())
    print(f"\n[*] Training {len(fold_ids)}-fold hybrid ensembles...")
    fold_metrics = {}
    for fold in fold_ids:
        t_m = (df["fold"] != fold).values
        v_m = (df["fold"] == fold).values
        for key, (X, col, (leaf, min_leaf, l2)) in specs.items():
            hgbr = make_hgbr(leaf, min_leaf, l2).fit(X[t_m], Y[t_m, col])
            et = make_et().fit(X[t_m], Y[t_m, col])
            oof_raw[key][0][v_m] = hgbr.predict(X[v_m])
            oof_raw[key][1][v_m] = et.predict(X[v_m])
            models[key].append((hgbr, et))

        fold_raw = {k: (oof_raw[k][0][v_m], oof_raw[k][1][v_m]) for k in specs}
        pa_f, fl_f, mt_f = assemble(fold_raw, WEIGHTS)
        f_mae = [float(np.mean(np.abs(pa_f - Y[v_m, 0]))),
                 float(np.mean(np.abs(fl_f - Y[v_m, 1]))),
                 float(np.mean(np.abs(mt_f - Y[v_m, 2])))]
        f_score = local_score(pa_f, fl_f, mt_f, Y[v_m])
        fold_metrics[f"fold_{fold}"] = {
            "local_score": round(f_score, 5), "mae_pa_deg": round(f_mae[0], 4),
            "mae_fl_mm": round(f_mae[1], 4), "mae_mt_mm": round(f_mae[2], 4),
        }
        print(f"  Fold {fold} | local score {f_score:.5f} | PA {f_mae[0]:.3f} deg | FL {f_mae[1]:.3f} mm | MT {f_mae[2]:.3f} mm")

    # ------------------------- out-of-fold evaluation -------------------------
    oof_pa, oof_fl, oof_mt = assemble(oof_raw, WEIGHTS)
    overall = local_score(oof_pa, oof_fl, oof_mt, Y)

    variants = {
        "your tuned weights": WEIGHTS,
        "plain 50/50 blend, no physics": {"pa": 0.5, "mt": 0.5, "fl": 0.5, "phys": 0.0},
        "gradient boosting only": {"pa": 1.0, "mt": 1.0, "fl": 1.0, "phys": 0.0},
    }
    variant_scores = {name: local_score(*assemble(oof_raw, w), Y) for name, w in variants.items()}

    def stats(pred, col, tol):
        err = pred - Y[:, col]
        return {
            "mae": float(np.mean(np.abs(err))),
            "r": float(np.corrcoef(Y[:, col], pred)[0, 1]),
            "r2": float(1.0 - np.sum(err**2) / (np.sum((Y[:, col] - Y[:, col].mean())**2) + 1e-6)),
            "within_tol_pct": float(np.mean(np.abs(err) <= tol) * 100.0),
        }

    s_pa, s_fl, s_mt = stats(oof_pa, 0, 3.0), stats(oof_fl, 1, 6.0), stats(oof_mt, 2, 1.5)
    agreement = (s_pa["within_tol_pct"] + s_fl["within_tol_pct"] + s_mt["within_tol_pct"]) / 3.0

    group_report = {}
    png_mask = is_png(df).astype(bool)
    for name, m in (("png", png_mask), ("other", ~png_mask)):
        if m.sum() == 0:
            continue
        group_report[name] = {"n": int(m.sum())}
        for tname, pred, col in (("pa_deg", oof_pa, 0), ("fl_mm", oof_fl, 1), ("mt_mm", oof_mt, 2)):
            group_report[name][tname] = {
                "mae": round(float(np.mean(np.abs(pred[m] - Y[m, col]))), 4),
                "bias": round(float(np.mean(pred[m] - Y[m, col])), 4),
            }

    report = {
        "local_oof_score": round(overall, 5),
        "score_divisors": SCORE_DIVISORS,
        "blend_comparison": {k: round(v, 5) for k, v in variant_scores.items()},
        "clinical_agreement_pct": round(agreement, 2),
        "pennation_angle": {k: round(v, 4) for k, v in s_pa.items()},
        "fascicle_length": {k: round(v, 4) for k, v in s_fl.items()},
        "muscle_thickness": {k: round(v, 4) for k, v in s_mt.items()},
        "per_group": group_report,
        "config": {"use_group_flag": USE_GROUP_FLAG, "smooth_test": SMOOTH_TEST_PREDICTIONS, "weights": WEIGHTS},
        "fold_summary": fold_metrics,
    }
    proc.mkdir(parents=True, exist_ok=True)
    report_path = proc / "model_accuracy_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Accuracy report saved to: {report_path}")

    # ------------------------------ test inference -----------------------------
    test_manifest = proc / "test_manifest.csv"
    if not test_manifest.exists():
        raise FileNotFoundError(
            f"{test_manifest} not found. Refusing to write a submission from training predictions."
        )
    test_df = pd.read_csv(test_manifest)
    test_paths = test_df["image_path"] if "image_path" in test_df.columns else None
    X1_te = load_or_extract(proc / "test_features.npy", test_paths, extract_structural_metrology_features_28, "28-dim test features")
    X2_te = load_or_extract(proc / "test_features_v2.npy", test_paths, extract_advanced_features_34, "34-dim test features")
    if USE_GROUP_FLAG:
        X1_te = np.hstack([X1_te, is_png(test_df)[:, None]])
    X_combo_te = np.hstack([X1_te, X2_te])
    test_X = {"pa": X1_te, "mt": X_combo_te, "fl": X_combo_te}

    print(f"\n[*] Running inference on {len(test_df)} test scans...")
    test_raw = {}
    for key in specs:
        h_preds = [h.predict(test_X[key]) for h, _ in models[key]]
        e_preds = [e.predict(test_X[key]) for _, e in models[key]]
        test_raw[key] = (np.mean(h_preds, axis=0), np.mean(e_preds, axis=0))
    te_pa, te_fl, te_mt = assemble(test_raw, WEIGHTS)

    if SMOOTH_TEST_PREDICTIONS and len(te_pa) > 5:
        from scipy.signal import savgol_filter
        print("[!] Smoothing test predictions (experimental; assumes test order is temporal).")
        te_pa = savgol_filter(te_pa, window_length=5, polyorder=2)
        te_fl = savgol_filter(te_fl, window_length=5, polyorder=2)
        te_mt = savgol_filter(te_mt, window_length=5, polyorder=2)

    sub_df = pd.DataFrame({
        "image_id": test_df["image_id"], "pa_deg": te_pa, "fl_mm": te_fl, "mt_mm": te_mt,
    })

    if sample_sub is not None and Path(sample_sub).exists():
        sample = pd.read_csv(sample_sub)
        if set(sample["image_id"]) != set(sub_df["image_id"]):
            raise ValueError("image_id values do not match sample_submission.csv")
        sub_df = sample[["image_id"]].merge(sub_df, on="image_id", how="left")
        print("[+] Submission rows aligned to sample_submission.csv order")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    sub_dir = output_dir / "submissions"
    csv_out = sub_dir / f"submission_hybrid_{stamp}.csv"
    xlsx_out = sub_dir / f"submission_hybrid_{stamp}.xlsx"
    sub_results = write_submission(sub_df, csv_out, xlsx_out)

    print("\n" + "=" * 72)
    print(" LOCAL OUT-OF-FOLD SUMMARY (not a leaderboard score)")
    print("=" * 72)
    print(f" Local OOF score: {overall:.5f}   (divisors {SCORE_DIVISORS}; check they match the Evaluation tab)")
    for name, sc in variant_scores.items():
        print(f"   {name:32s} {sc:.5f}")
    print(f" Pennation angle MAE : {s_pa['mae']:.3f} deg  (R2 {s_pa['r2']:.3f})")
    print(f" Fascicle length MAE : {s_fl['mae']:.3f} mm   (R2 {s_fl['r2']:.3f})")
    print(f" Muscle thickness MAE: {s_mt['mae']:.3f} mm   (R2 {s_mt['r2']:.3f})")
    for name, g in group_report.items():
        print(f" [{name}] n={g['n']}  " + "  ".join(
            f"{t}: MAE {g[t]['mae']:.2f}, bias {g[t]['bias']:+.2f}" for t in ("pa_deg", "fl_mm", "mt_mm")))
    print("-" * 72)
    print(f" Submission CSV (comma, verified): {csv_out}")
    print("=" * 72)

    return {"accuracy": report, "submission": sub_results}


if __name__ == "__main__":
    run_high_precision_hybrid_training()