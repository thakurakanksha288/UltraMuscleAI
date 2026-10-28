r"""
UMUD Challenge - local VS Code script (CNN on the real images, no hand-made features).

RUN (PowerShell, from the folder that contains this file):
    python -m pip install torch torchvision scikit-learn pandas pillow
    python umud_train_local.py --quick                      # 2-minute test: 1 fold, 2 epochs
    python umud_train_local.py                              # full run
    python umud_train_local.py "D:\path\to\competition_data"  # if your data is elsewhere

It looks for the labels csv, the images and sample_submission.csv by itself (and unzips the
competition .zip if the images are still zipped), then prints what it found.
Output: submissions\submission_cnn_<timestamp>.csv  (comma separated, verified).

Why this instead of the earlier pipelines
  * Learns from the images (pretrained ResNet34) instead of hand-made pixel features that assume
    a fixed 10 pixels per mm for every scanner.
  * Also feeds the original image size and file type (.tif / .png) to the network, because these
    differ between scanners and change the pixel-to-mm scale.
  * Blocked folds (neighbouring frames stay together) so the local score is not inflated.
  * Trains directly on the competition metric (MAE_pa/18 + MAE_fl/36 + MAE_mt/9) and prints that
    exact score locally, next to a median-guess baseline, plus per-image-type error and bias.
  * Without an NVIDIA GPU it still runs, but slowly: use --quick to test, then lower EPOCHS.
"""
import os
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.model_selection import GroupKFold, KFold
from torchvision import transforms as T

# ----------------------------- CONFIG ---------------------------------------
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
ON_KAGGLE = os.path.isdir("/kaggle/input")
INPUT_ROOT = ARGS[0] if ARGS else ("/kaggle/input" if ON_KAGGLE else r"D:\UltraMuscleAI\data\raw")
OUT_DIR = "/kaggle/working" if ON_KAGGLE else "submissions"
TARGETS = ["pa_deg", "fl_mm", "mt_mm"]
# Competition metric: Score = MAE_pa/18 + MAE_fl/36 + MAE_mt/9  (lower is better)
TOL = np.array([18.0, 36.0, 9.0], dtype=np.float32)
QUICK = "--quick" in sys.argv   # 1 fold x 2 epochs, just to check that everything runs
SIZE = 320           # image size fed to the network (use 224 if you have no GPU)
FOLDS = 5
EPOCHS = 20
BATCH = 16
LR = 3e-4
SEED = 42
NUM_WORKERS = 0 if os.name == "nt" else 2
# -----------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
ID_CANDIDATES = ["image_id", "id", "filename", "file_name", "file", "image", "image_name", "img", "path"]
GROUP_HINTS = ("subject", "patient", "seq", "video", "group", "participant")


# ----------------------------------------------------------------------------
# Finding the data
# ----------------------------------------------------------------------------
def read_csv_flex(path, **kw):
    try:
        return pd.read_csv(path, sep=None, engine="python", **kw)
    except Exception:
        return pd.read_csv(path, **kw)


def scan(root):
    csvs, images = [], []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            ext = Path(fn).suffix.lower()
            if ext == ".csv":
                csvs.append(p)
            elif ext in IMG_EXT:
                images.append(p)
    return sorted(csvs), sorted(images)


def unzip_if_needed(root):
    """If the competition data is still a .zip, extract it next to the zip."""
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".zip"):
                z = os.path.join(dp, fn)
                target = os.path.join(dp, Path(fn).stem + "_unzipped")
                if not os.path.isdir(target):
                    print(f"[*] Unzipping {z} (this can take a few minutes)...")
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(target)


def path_score(path, want):
    p = path.lower().replace("\\", "/")
    s = 0
    if any(k in p for k in ("mask", "label", "seg", "annot", "overlay")):
        s -= 10
    if want in p:
        s += 5
    other = "train" if want == "test" else "test"
    if other in p:
        s -= 5
    return s


class ImageIndex:
    def __init__(self, images):
        self.by_name, self.by_stem = {}, {}
        for p in images:
            name = os.path.basename(p).lower()
            self.by_name.setdefault(name, []).append(p)
            self.by_stem.setdefault(Path(name).stem, []).append(p)

    def find(self, image_id, want):
        key = str(image_id).lower().replace("\\", "/").split("/")[-1]
        cands = self.by_name.get(key) or self.by_stem.get(Path(key).stem, [])
        if not cands:
            return None
        return max(cands, key=lambda c: (path_score(c, want), -len(c)))


def pick_id_col(df):
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]


def discover(root):
    csvs, images = scan(root)
    if not images:
        unzip_if_needed(root)
        csvs, images = scan(root)
    print(f"[*] Found {len(csvs)} csv files and {len(images)} images under {root}")
    if not images:
        sys.exit(f"No images found under {root}. Pass the data folder: python umud_train_local.py \"D:\\path\\to\\data\"")
    index = ImageIndex(images)

    train_csvs, sample_csv = [], None
    for p in csvs:
        try:
            head = read_csv_flex(p, nrows=5)
        except Exception:
            continue
        if all(t in head.columns for t in TARGETS):
            low = os.path.basename(p).lower()
            if "sample" in low or "submission" in low:
                sample_csv = sample_csv or p
            else:
                train_csvs.append(p)
    if not train_csvs:
        print("\nCould not find a csv with columns", TARGETS)
        for p in csvs[:30]:
            try:
                print("  ", p, "->", list(read_csv_flex(p, nrows=2).columns))
            except Exception:
                pass
        sys.exit("Paste the lines above in the chat and I will adapt the script.")
    train_csv = max(train_csvs, key=lambda p: ("train" in p.lower(), os.path.getsize(p)))
    print(f"[+] Training labels: {train_csv}")

    tr = read_csv_flex(train_csv)
    id_col = pick_id_col(tr)
    tr = tr.rename(columns={id_col: "image_id"}).drop_duplicates("image_id").reset_index(drop=True)
    tr["path"] = [index.find(i, "train") for i in tr["image_id"]]
    missing = int(tr["path"].isna().sum())
    if missing:
        print(f"[!] {missing} of {len(tr)} training ids have no image file; dropping them.")
    tr = tr.dropna(subset=["path"] + TARGETS).reset_index(drop=True)
    if len(tr) < 20:
        sys.exit("Too few labelled training images could be matched to files. Paste the first rows of the labels csv.")

    if sample_csv:
        print(f"[+] Sample submission: {sample_csv}")
        te = read_csv_flex(sample_csv)
        te = te.rename(columns={pick_id_col(te): "image_id"})[["image_id"]]
        te["path"] = [index.find(i, "test") for i in te["image_id"]]
        if te["path"].isna().any():
            sys.exit(f"{int(te['path'].isna().sum())} test ids from the sample submission have no image file.")
    else:
        print("[!] No sample submission found; using every image whose path contains 'test'.")
        paths = [p for p in images if "test" in p.lower() and path_score(p, "test") >= 0]
        te = pd.DataFrame({"image_id": [os.path.basename(p) for p in paths], "path": paths})
        te = te.sort_values("image_id").reset_index(drop=True)
    print(f"[+] {len(tr)} training images, {len(te)} test images")
    return tr, te


# ----------------------------------------------------------------------------
# Loading images
# ----------------------------------------------------------------------------
def px_per_cm(h, w, is_png):
    """Approximate pixels per cm by scanner type (table taken from the public UMUD notebook; unverified)."""
    if (h, w) == (644, 1088):
        return 126.1
    if h == 853:
        return 166.8
    if 505 <= h <= 520:
        return 77.6
    if is_png:
        return 148.2
    return 135.333


def load_gray(path):
    img = Image.open(path)
    if img.mode.startswith("I") or img.mode == "F":  # 16-bit / float tif
        a = np.asarray(img).astype(np.float32)
        a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6) * 255.0
        img = Image.fromarray(a.astype(np.uint8))
    w, h = img.size
    arr = np.asarray(img.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)
    is_png = 1.0 if str(path).lower().endswith(".png") else 0.0
    meta = np.array([np.log(w) / 8.0, np.log(h) / 8.0, (w / h) / 2.0, is_png,
                     px_per_cm(h, w, bool(is_png)) / 150.0], dtype=np.float32)
    return arr, meta


def preload(paths, label):
    print(f"[*] Loading {len(paths)} {label} images...")
    out = [load_gray(p) for p in paths]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def make_folds(tr, n_splits):
    gcols = [c for c in tr.columns if any(h in c.lower() for h in GROUP_HINTS)]
    if gcols:
        print(f"[+] Folds grouped by column '{gcols[0]}'")
        return list(GroupKFold(n_splits).split(tr, groups=tr[gcols[0]].values))
    print("[+] No subject/sequence column found: using blocked folds (neighbouring rows stay together).")
    return list(KFold(n_splits, shuffle=False).split(tr))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class DS(torch.utils.data.Dataset):
    def __init__(self, imgs, meta, y=None, train=False):
        self.imgs, self.meta, self.y, self.train = imgs, meta, y, train
        self.affine = T.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)) if train else None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = torch.from_numpy(self.imgs[i]).float().div_(255.0).unsqueeze(0)
        if self.train:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])  # a mirrored image keeps the same angle and lengths
            x = (x * (1.0 + (random.random() - 0.5) * 0.4) + (random.random() - 0.5) * 0.2).clamp_(0, 1)
            x = self.affine(x)
        x = (x.expand(3, -1, -1) - MEAN) / STD
        m = torch.from_numpy(self.meta[i])
        if self.y is None:
            return x, m
        return x, m, torch.tensor(self.y[i], dtype=torch.float32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            backbone = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
        except Exception as exc:
            print(f"[!] Could not load pretrained weights ({exc}). Turn Internet On for much better results.")
            backbone = torchvision.models.resnet34(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(512 + 5, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, len(TARGETS)))

    def forward(self, x, meta):
        return self.head(torch.cat([self.backbone(x), meta], dim=1))


@torch.no_grad()
def predict(model, imgs, meta, tta=True):
    model.eval()
    dl = torch.utils.data.DataLoader(DS(imgs, meta), batch_size=BATCH * 2, num_workers=NUM_WORKERS)
    out = []
    for x, m in dl:
        x, m = x.to(DEVICE), m.to(DEVICE)
        p = model(x, m)
        if tta:
            p = (p + model(torch.flip(x, dims=[3]), m)) / 2
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def train_fold(imgs, meta, y_std, tr_idx, va_idx, epochs, loss_w):
    model = Net().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(DS(imgs[tr_idx], meta[tr_idx], y_std[tr_idx], train=True), batch_size=BATCH,
                                     shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * len(dl))
    use_amp = DEVICE == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    w = torch.tensor(loss_w, dtype=torch.float32, device=DEVICE)  # sd/tolerance: weights errors like the metric
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        for x, m, y in dl:
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                err = F.smooth_l1_loss(model(x, m).float(), y, beta=0.05, reduction="none")
                loss = (err * w).sum(dim=1).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        pv = predict(model, imgs[va_idx], meta[va_idx], tta=False)
        mae = float(np.abs(pv - y_std[va_idx]).mean())
        if mae < best:
            best, best_state = mae, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"    epoch {ep + 1:02d}/{epochs}  val MAE (standardized) {mae:.4f}")
    model.load_state_dict(best_state)
    return model


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def write_submission(ids, preds, out_path):
    sub = pd.DataFrame({"image_id": ids})
    for j, t in enumerate(TARGETS):
        sub[t] = preds[:, j]
    if sub.isna().any().any() or not sub["image_id"].is_unique:
        raise ValueError("Submission has NaN values or duplicate ids.")
    sub.to_csv(out_path, index=False, float_format="%.3f")
    with open(out_path, "r", newline="") as f:
        header = f.readline().strip()
    check = pd.read_csv(out_path)
    if header != "image_id," + ",".join(TARGETS) or check.shape != (len(sub), 1 + len(TARGETS)):
        raise RuntimeError(f"Submission check failed: header={header!r}, shape={check.shape}")
    print(f"[+] Wrote and verified {out_path}: {check.shape[0]} rows, header '{header}'")


def comp_score(pred, y):
    """Competition metric on raw units."""
    return float((np.abs(pred - y).mean(0) / TOL).sum())


def report(tr, Y, oof, have):
    """Local out-of-fold report using the competition metric. Baseline = the training medians for every image."""
    y, p = Y[have], oof[have]
    base_pred = np.tile(np.median(Y, axis=0), (len(y), 1))
    print("\n" + "=" * 70)
    print(" LOCAL OUT-OF-FOLD REPORT (n=%d)   Score = MAE_pa/18 + MAE_fl/36 + MAE_mt/9" % have.sum())
    print("=" * 70)
    for j, t in enumerate(TARGETS):
        print(f" {t:6s} MAE {np.abs(p[:, j] - y[:, j]).mean():8.3f}   (median-guess baseline {np.abs(base_pred[:, j] - y[:, j]).mean():.3f})")
    print(f" LOCAL SCORE: model {comp_score(p, y):.4f}   vs   median-guess baseline {comp_score(base_pred, y):.4f}")
    print(" (public leaderboard reference points: about 0.45 baseline, 0.374 public notebook, 0.206 top)")
    is_png = tr["path"].str.lower().str.endswith(".png").values[have]
    for name, m in (("png", is_png), ("other", ~is_png)):
        if m.sum() == 0:
            continue
        row = "  ".join(f"{t}: MAE {np.abs(p[m, j] - y[m, j]).mean():.2f}, bias {np.mean(p[m, j] - y[m, j]):+.2f}"
                        for j, t in enumerate(TARGETS))
        print(f" [{name}] n={int(m.sum())}  score {comp_score(p[m], y[m]):.4f}  {row}")
    print("=" * 70)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}   QUICK={QUICK}   data: {INPUT_ROOT}")
    if DEVICE == "cpu" and not QUICK:
        print("[!] No GPU found: a full run can take many hours. Try --quick first, then lower EPOCHS/FOLDS/SIZE.")
    tr, te = discover(INPUT_ROOT)

    imgs, meta = preload(tr["path"].tolist(), "training")
    imgs_te, meta_te = preload(te["path"].tolist(), "test")
    Y = tr[TARGETS].values.astype(np.float32)
    mu, sd = Y.mean(0), Y.std(0)
    y_std = (Y - mu) / sd

    folds = make_folds(tr, FOLDS)
    epochs = 2 if QUICK else EPOCHS
    if QUICK:
        folds = folds[:1]

    oof = np.zeros_like(Y)
    have = np.zeros(len(Y), dtype=bool)
    test_pred = np.zeros((len(te), len(TARGETS)))
    for k, (tr_idx, va_idx) in enumerate(folds):
        print(f"\nFold {k + 1}/{len(folds)}  (train {len(tr_idx)}, val {len(va_idx)})")
        model = train_fold(imgs, meta, y_std, tr_idx, va_idx, epochs, sd / TOL)
        oof[va_idx] = predict(model, imgs[va_idx], meta[va_idx]) * sd + mu
        have[va_idx] = True
        test_pred += (predict(model, imgs_te, meta_te) * sd + mu) / len(folds)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    report(tr, Y, oof, have)
    test_pred = np.clip(test_pred, Y.min(0), Y.max(0))

    is_png_te = te["path"].str.lower().str.endswith(".png").values
    for name, m in (("png", is_png_te), ("other", ~is_png_te)):
        if m.sum():
            print(f" test predictions [{name}] mean: " + ", ".join(
                f"{t} {test_pred[m, j].mean():.1f}" for j, t in enumerate(TARGETS)))
    print(" training label mean: " + ", ".join(f"{t} {Y[:, j].mean():.1f}" for j, t in enumerate(TARGETS)))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT_DIR, exist_ok=True)
    name = "submission.csv" if ON_KAGGLE else f"submission_cnn_{stamp}.csv"
    out_path = os.path.join(OUT_DIR, name)
    write_submission(te["image_id"].tolist(), test_pred, out_path)
    print(f"Done. Submit {out_path} on Kaggle.")


if __name__ == "__main__":
    main()