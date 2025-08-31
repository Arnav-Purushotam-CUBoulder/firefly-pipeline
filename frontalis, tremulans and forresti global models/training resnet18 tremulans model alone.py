#!/usr/bin/env python3
"""
Firefly/background 10×10 CNN — now with a selectable ResNet backbone.
Everything else (data handling, loss, sampler, live training chart, checkpoints)
is unchanged from the original script, except we now feed RGB into the model.
"""

# ── USER SETTINGS ───────────────────────────────────────────────
DATA_DIR        = '/Users/arnavps/Desktop/tremulans and forresti individual datasets/final tremulans dataset'
BEST_MODEL_PATH = 'frontalis, tremulans and forresti global models/resnet18_Tremulans_best_model.pt'
EPOCHS, BATCH_SIZE, LR = 50, 128, 3e-4
NUM_WORKERS = 2

# choose your backbone: 'resnet18' | 'resnet34' | 'resnet50' | 'resnet101' | 'resnet152'
RESNET_MODEL   = 'resnet18'
# ────────────────────────────────────────────────────────────────

import sys, os, time, math, random
from pathlib import Path
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
import torch, torch.nn as nn, torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F
from collections import Counter
from tqdm import tqdm

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

# ── main (spawn-safe for macOS) ────────────────────────────────
def main():
    global DATA_DIR
    if DATA_DIR is None:
        tk.Tk().withdraw()
        DATA_DIR = tk.filedialog.askdirectory(title="Select dataset_split folder")
        if not DATA_DIR: sys.exit("No dataset folder selected – exiting.")
    print(f"Dataset root: {DATA_DIR}")

    DEVICE = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # ─ transforms (now keep color)
    train_tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRotate90(),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    plain_tfm = transforms.Compose([
        transforms.ToTensor()
    ])

    # ─ datasets
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_tfm)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   plain_tfm)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  plain_tfm)

    # ─ handle class imbalance
    counts = Counter(train_ds.targets)
    n_classes = len(train_ds.classes)
    class_weights = torch.tensor(
        [sum(counts.values()) / counts[c] for c in range(n_classes)],
        dtype=torch.float)
    sample_weights = [class_weights[t] for t in train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(train_ds),
                                    replacement=True)

    # ─ loaders
    train_dl = DataLoader(train_ds, BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    # ─ model / loss / optimiser
    model = build_resnet(RESNET_MODEL).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    Path(BEST_MODEL_PATH).expanduser().parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0

    # ─ live plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Value")
    ln_tr, = ax.plot([], [], label="train loss")
    ln_val, = ax.plot([], [], label="val loss")
    ln_acc, = ax.plot([], [], label="val acc")
    ax.legend()

    def update_plot(ep, tr, vl, acc, eta):
        for ln, y in zip((ln_tr, ln_val, ln_acc), (tr, vl, acc)):
            ln.set_data(list(ln.get_xdata())+[ep], list(ln.get_ydata())+[y])
        ax.relim(); ax.autoscale_view()
        fig.suptitle(f"Global ETA: {eta}", fontsize=10)
        plt.pause(0.001)

    # ─ epoch helper
    def run_epoch(loader, train=True, desc=""):
        model.train() if train else model.eval()
        tot_loss = tot_corr = tot = 0
        for x, y in tqdm(loader, desc=desc, ncols=100, leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train: optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                out = model(x); loss = criterion(out, y)
                if train:
                    loss.backward()
                    optimizer.step()
            tot_loss += loss.item()*x.size(0)
            tot_corr += (out.argmax(1)==y).sum().item(); tot += x.size(0)
        return tot_loss/tot, tot_corr/tot

    # ─ training loop
    start, ep_times = time.time(), []
    for ep in range(1, EPOCHS+1):
        t0 = time.time()
        tr_loss, _        = run_epoch(train_dl, True,  f"train {ep:02d}")
        val_loss, val_acc = run_epoch(val_dl,   False, f"val   {ep:02d}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'val_acc': val_acc},
                       BEST_MODEL_PATH)
            print(f"  ✅ saved new best (val_acc={val_acc:.2%}) → {BEST_MODEL_PATH}")

        ep_times.append(time.time()-t0)
        eta = time.strftime("%H:%M:%S", time.gmtime(sum(ep_times)/len(ep_times)*(EPOCHS-ep)))
        print(f"Epoch {ep:02d} | train_loss {tr_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.2%} | "
              f"epoch {time.strftime('%H:%M:%S', time.gmtime(ep_times[-1]))} | ETA {eta}")
        update_plot(ep, tr_loss, val_loss, val_acc, eta)

    # ─ reload best & evaluate on TEST ───────────────────────────
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    best_model = build_resnet(RESNET_MODEL).to(DEVICE); best_model.load_state_dict(ckpt['model'])
    best_model.eval()
    test_loss = corr = tot = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = best_model(x); test_loss += criterion(out, y).item()*x.size(0)
            corr += (out.argmax(1)==y).sum().item(); tot += x.size(0)
    print(f"\nBest checkpoint epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.2%})")
    print(f"TEST | loss {test_loss/tot:.4f} | acc {corr/tot:.2%}")
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}")
    plt.ioff(); plt.show()

# ── entry-point ────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
