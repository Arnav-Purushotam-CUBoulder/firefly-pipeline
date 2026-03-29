#!/usr/bin/env python3
from __future__ import annotations

"""
Stage 2 (Trainer)
-----------------

Firefly/background 10×10 CNN — now with a selectable ResNet backbone.
Everything else (data handling, loss, sampler, live training chart, checkpoints)
is unchanged from the original script, except we now feed RGB into the model.
"""

# ── USER SETTINGS ───────────────────────────────────────────────
DATA_DIR        = '/Volumes/DL Project SSD/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/dataset/v2/final dataset'
BEST_MODEL_PATH = '/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/models and other data/pyrallis gopro models resnet18/resnet18_pyrallis_gopro_best_model v3.pt'
EPOCHS, BATCH_SIZE, LR = 100, 128, 3e-4
NUM_WORKERS = 2

# choose your backbone: 'resnet18' | 'resnet34' | 'resnet50' | 'resnet101' | 'resnet152'
RESNET_MODEL   = 'resnet18'
# ────────────────────────────────────────────────────────────────

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F
from collections import Counter
from tqdm import tqdm

# ── dataset sanitizer ─────────────────────────────────────────
def sanitize_dataset_images(
    data_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
    exts: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    mode: str = "quarantine",  # "quarantine" | "delete"
    quarantine_dir: str | Path | None = None,
    verify_with_pil: bool = True,
    report_max: int = 20,
) -> dict:
    """
    Best-effort sanitize to avoid DataLoader crashes due to corrupt image files.

    We treat 0-byte images as corrupt, and optionally PIL-verify remaining images.
    Corrupt files are either moved out of the dataset tree (quarantine) or deleted.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    mode_norm = (mode or "").strip().lower()
    if mode_norm not in {"quarantine", "delete"}:
        raise ValueError(f"sanitize mode must be 'quarantine' or 'delete' (got {mode!r})")

    qroot: Path | None = None
    if mode_norm == "quarantine":
        qroot = Path(quarantine_dir).expanduser().resolve() if quarantine_dir else (data_dir / "__corrupt__")
        qroot.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Pillow (PIL) is required for dataset sanitization: {e}") from e

    preview: list[dict] = []
    per_split: dict[str, dict[str, int]] = {}
    total_bad = 0

    for split in splits:
        split_root = data_dir / split
        if not split_root.exists():
            per_split[split] = {"scanned": 0, "bad": 0, "actioned": 0}
            continue

        # Pre-count for a real ETA.
        total = 0
        for root, _, files in os.walk(split_root):
            for f in files:
                if Path(f).suffix.lower() in exts:
                    total += 1

        scanned = bad = actioned = 0
        pbar = (
            tqdm(total=total, desc=f"[sanitize] {split}", unit="img", dynamic_ncols=True)
            if total > 0
            else None
        )
        for root, _, files in os.walk(split_root):
            for f in files:
                if Path(f).suffix.lower() not in exts:
                    continue
                p = Path(root) / f
                scanned += 1

                reason: str | None = None
                try:
                    if p.stat().st_size == 0:
                        reason = "zero_bytes"
                    elif verify_with_pil:
                        # verify() catches many truncations; load() catches decode issues.
                        with Image.open(p) as im:  # type: ignore[attr-defined]
                            im.verify()
                        with Image.open(p) as im:  # type: ignore[attr-defined]
                            im.load()
                except Exception as e:
                    reason = f"{type(e).__name__}: {e}"

                if reason is not None:
                    bad += 1
                    total_bad += 1
                    if len(preview) < int(report_max):
                        preview.append({"path": str(p), "split": split, "reason": reason})

                    try:
                        if mode_norm == "delete":
                            p.unlink(missing_ok=True)
                            actioned += 1
                        else:
                            assert qroot is not None
                            rel = p.relative_to(split_root)
                            dst = qroot / split / rel
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            if dst.exists():
                                dst = dst.with_name(f"{dst.stem}__dup{total_bad}{dst.suffix}")
                            shutil.move(str(p), str(dst))
                            actioned += 1
                    except Exception:
                        # Non-fatal: we'll still try training, but it may crash if corrupt files remain.
                        pass

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(bad=int(bad))

        if pbar is not None:
            pbar.close()

        per_split[split] = {"scanned": int(scanned), "bad": int(bad), "actioned": int(actioned)}

    if total_bad > 0:
        print(
            "[sanitize] Found corrupt images:",
            f"bad={total_bad} mode={mode_norm} verify_with_pil={bool(verify_with_pil)}"
            + (f" quarantine_dir={qroot}" if qroot is not None else ""),
        )
        for r in preview:
            print(f"  - {r['path']} ({r['split']} | {r['reason']})")
        if total_bad > len(preview):
            print(f"  ... and {total_bad - len(preview)} more")

    return {
        "enabled": True,
        "mode": mode_norm,
        "verify_with_pil": bool(verify_with_pil),
        "quarantine_dir": str(qroot) if qroot is not None else None,
        "per_split": per_split,
        "bad_total": int(total_bad),
        "bad_preview": preview,
    }

# ── helper: rotate by 0/90/180/270 ─────────────────────────────
class RandomRotate90:
    def __call__(self, img): return F.rotate(img, random.choice([0, 90, 180, 270]))

# ── helper: build a 3-channel, 2-class ResNet ──────────────────
def build_resnet(name: str, num_classes: int = 2) -> nn.Module:
    resnet_fns = {
        'resnet18':  models.resnet18,
        'resnet34':  models.resnet34,
        'resnet50':  models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }
    if name not in resnet_fns:
        raise ValueError(f"Unknown ResNet model '{name}'. Choose from {list(resnet_fns)}")
    net = resnet_fns[name](weights=None)                      # no pre-training
    # final fc: match our two classes
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    nn.init.xavier_uniform_(net.fc.weight)
    nn.init.zeros_(net.fc.bias)
    return net

# ── programmatic API (used by the integrated orchestrator) ──────────────────
def train_resnet_classifier(
    *,
    data_dir: str | Path,
    best_model_path: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    resnet_model: str,
    seed: int = 1337,
    no_gui: bool = True,
    metrics_out: str | Path | None = None,
    sanitize_dataset: bool = True,
    sanitize_mode: str = "quarantine",
    sanitize_quarantine_dir: str | Path | None = None,
    sanitize_verify_with_pil: bool = True,
    sanitize_report_max: int = 20,
) -> dict:
    """
    Train a firefly/background ResNet classifier on a dataset folder that looks like:
      <data_dir>/{train,val,test}/{firefly,background}/*.png

    Returns a JSON-serializable metrics dict.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    best_model_path = Path(best_model_path).expanduser().resolve()
    if not str(best_model_path):
        raise ValueError("best_model_path is required")

    epochs = int(epochs)
    batch_size = int(batch_size)
    lr = float(lr)
    num_workers = int(num_workers)
    resnet_model = str(resnet_model)
    seed = int(seed)

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    plt = None
    if not no_gui:
        import matplotlib  # local import

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt  # type: ignore[no-redef]

    print(f"Dataset root: {data_dir}")

    DEVICE = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {DEVICE}")

    sanitizer_report: dict = {}
    if sanitize_dataset:
        sanitizer_report = sanitize_dataset_images(
            data_dir,
            mode=str(sanitize_mode),
            quarantine_dir=sanitize_quarantine_dir,
            verify_with_pil=bool(sanitize_verify_with_pil),
            report_max=int(sanitize_report_max),
        )

    # ─ transforms (now keep color)
    train_tfm = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotate90(),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ]
    )
    plain_tfm = transforms.Compose([transforms.ToTensor()])

    # ─ datasets
    train_ds = datasets.ImageFolder(os.path.join(str(data_dir), "train"), train_tfm)
    val_ds = datasets.ImageFolder(os.path.join(str(data_dir), "val"), plain_tfm)
    test_ds = datasets.ImageFolder(os.path.join(str(data_dir), "test"), plain_tfm)

    # This pipeline expects exactly two classes: firefly/background.
    if len(train_ds.classes) != 2:
        raise RuntimeError(f"Expected 2 classes under train/. Found: {train_ds.classes}")

    # ─ handle class imbalance
    counts = Counter(train_ds.targets)
    n_classes = len(train_ds.classes)
    total = sum(counts.values())
    class_weights_list = []
    missing = []
    for c in range(n_classes):
        n_c = int(counts.get(c, 0))
        if n_c <= 0:
            class_weights_list.append(0.0)
            missing.append(train_ds.classes[c])
        else:
            class_weights_list.append(total / n_c)
    if missing:
        print(f"WARNING: Missing classes in train split (no samples): {missing}")
        print("Training will proceed, but metrics/models may be meaningless without negatives/positives.")
    class_weights = torch.tensor(class_weights_list, dtype=torch.float)
    sample_weights = [class_weights[t] for t in train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    # ─ loaders
    train_dl = DataLoader(train_ds, batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ─ model / loss / optimiser
    model = build_resnet(resnet_model).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_model_path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    best_epoch = -1

    # ─ live plot
    if plt is not None:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ln_tr, = ax.plot([], [], label="train loss")
        ln_val, = ax.plot([], [], label="val loss")
        ln_acc, = ax.plot([], [], label="val acc")
        ax.legend()

    def update_plot(ep, tr, vl, acc, eta):
        if plt is None:
            return
        for ln, y in zip((ln_tr, ln_val, ln_acc), (tr, vl, acc)):
            ln.set_data(list(ln.get_xdata()) + [ep], list(ln.get_ydata()) + [y])
        ax.relim()
        ax.autoscale_view()
        fig.suptitle(f"Global ETA: {eta}", fontsize=10)
        plt.pause(0.001)

    # ─ epoch helper
    def run_epoch(loader, train=True, desc=""):
        model.train() if train else model.eval()
        tot_loss = tot_corr = tot = 0
        for x, y in tqdm(loader, desc=desc, ncols=100, leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                out = model(x)
                loss = criterion(out, y)
                if train:
                    loss.backward()
                    optimizer.step()
            tot_loss += loss.item() * x.size(0)
            tot_corr += (out.argmax(1) == y).sum().item()
            tot += x.size(0)
        return tot_loss / tot, tot_corr / tot

    # ─ training loop
    start, ep_times = time.time(), []
    ep_pbar = tqdm(range(1, epochs + 1), desc="[stage2] epochs", unit="epoch", dynamic_ncols=True)
    for ep in ep_pbar:
        t0 = time.time()
        tr_loss, _ = run_epoch(train_dl, True, f"train {ep:02d}")
        val_loss, val_acc = run_epoch(val_dl, False, f"val   {ep:02d}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            torch.save(
                {"epoch": ep, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "val_acc": val_acc},
                str(best_model_path),
            )
            print(f"  ✅ saved new best (val_acc={val_acc:.2%}) → {best_model_path}")

        ep_times.append(time.time() - t0)
        eta = time.strftime("%H:%M:%S", time.gmtime(sum(ep_times) / len(ep_times) * (epochs - ep)))
        ep_pbar.set_postfix(val_acc=f"{val_acc:.2%}", best=f"{best_val_acc:.2%}", eta=str(eta))
        print(
            f"Epoch {ep:02d} | train_loss {tr_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.2%} | "
            f"epoch {time.strftime('%H:%M:%S', time.gmtime(ep_times[-1]))} | ETA {eta}"
        )
        update_plot(ep, tr_loss, val_loss, val_acc, eta)

    # ─ reload best & evaluate on TEST ───────────────────────────
    ckpt = torch.load(str(best_model_path), map_location=DEVICE)
    best_model = build_resnet(resnet_model).to(DEVICE)
    best_model.load_state_dict(ckpt["model"])
    best_model.eval()
    test_loss = corr = tot = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = best_model(x)
            test_loss += criterion(out, y).item() * x.size(0)
            corr += (out.argmax(1) == y).sum().item()
            tot += x.size(0)
    test_loss = (test_loss / tot) if tot else float("nan")
    test_acc = (corr / tot) if tot else float("nan")
    print(f"\nBest checkpoint epoch {ckpt.get('epoch', best_epoch)} (val_acc={ckpt.get('val_acc', best_val_acc):.2%})")
    print(f"TEST | loss {test_loss:.4f} | acc {test_acc:.2%}")
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}")

    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(data_dir),
        "best_model_path": str(best_model_path),
        "resnet_model": resnet_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "num_workers": num_workers,
        "seed": seed,
        "best_epoch": int(ckpt.get("epoch", best_epoch)),
        "best_val_acc": float(ckpt.get("val_acc", best_val_acc)),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "sanitizer": sanitizer_report,
    }

    if metrics_out:
        outp = Path(metrics_out).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(metrics, indent=2))
        print(f"[metrics] Wrote: {outp}")

    if plt is not None:
        plt.ioff()
        plt.show()

    return metrics

# ── main (spawn-safe for macOS) ────────────────────────────────
def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Train a firefly/background ResNet classifier.")
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR) if DATA_DIR else "", help="Path to 'final dataset'.")
    ap.add_argument(
        "--best-model-path",
        type=str,
        default=str(BEST_MODEL_PATH) if BEST_MODEL_PATH else "",
        help="Where to save the best checkpoint (.pt).",
    )
    ap.add_argument("--epochs", type=int, default=int(EPOCHS), help="Number of epochs.")
    ap.add_argument("--batch-size", type=int, default=int(BATCH_SIZE), help="Batch size.")
    ap.add_argument("--lr", type=float, default=float(LR), help="Learning rate.")
    ap.add_argument("--num-workers", type=int, default=int(NUM_WORKERS), help="DataLoader workers.")
    ap.add_argument(
        "--resnet",
        type=str,
        default=str(RESNET_MODEL),
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="ResNet backbone.",
    )
    ap.add_argument("--seed", type=int, default=1337, help="Random seed.")
    ap.add_argument("--no-gui", action="store_true", help="Disable live plotting / GUI (headless mode).")
    ap.add_argument("--metrics-out", type=str, default="", help="Optional JSON file to write summary metrics.")
    args = ap.parse_args(argv)

    data_dir = str(args.data_dir or "").strip()
    if not data_dir:
        if args.no_gui:
            raise SystemExit("--data-dir is required when --no-gui is set")
        import tkinter as tk  # local import
        from tkinter import filedialog  # local import

        tk.Tk().withdraw()
        data_dir = filedialog.askdirectory(title="Select dataset_split folder")
        if not data_dir:
            raise SystemExit("No dataset folder selected – exiting.")

    best_model_path = str(Path(args.best_model_path).expanduser().resolve()) if args.best_model_path else ""
    if not best_model_path:
        raise SystemExit("--best-model-path is required")

    return train_resnet_classifier(
        data_dir=data_dir,
        best_model_path=best_model_path,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        num_workers=int(args.num_workers),
        resnet_model=str(args.resnet),
        seed=int(args.seed),
        no_gui=bool(args.no_gui),
        metrics_out=str(args.metrics_out or "").strip() or None,
    )

# ── entry-point ────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
