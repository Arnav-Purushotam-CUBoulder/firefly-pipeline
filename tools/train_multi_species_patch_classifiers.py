#!/usr/bin/env python3
from __future__ import annotations

"""
Train patch-classification models across many species datasets.

What it trains
--------------
1) Global model: trained on ALL species.
2) LOSO models: for each species S, train on (ALL \ {S}) and evaluate on S.

Parallelism model
-----------------
This script runs multiple *independent* trainings in parallel: 1 GPU → 1 model.
If there are more models than GPUs, jobs are queued and each GPU trains multiple
models sequentially.

Expected dataset layout (per species)
------------------------------------
<datasets_dir>/<species>/<version>/final dataset/
  train/{firefly,background}/*.png
  val/{firefly,background}/*.png
  test/{firefly,background}/*.png

The script auto-discovers species under --datasets-dir (or <root>/datasets or
<root> as a fallback) and picks the newest "final dataset" per species.
"""

import argparse
import json
import os
import queue
import random
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASS_NAMES = ("firefly", "background")
DEFAULT_FINAL_DATASET_DIRNAME = "final dataset"
SPLITS = ("train", "val", "test")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _slugify(name: str) -> str:
    s = name.strip().lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"


def _list_images(root: Path) -> List[Path]:
    out: List[Path] = []
    if not root.exists():
        return out
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in IMG_EXTS:
                out.append(p)
    out.sort()
    return out


def _has_expected_split_layout(dataset_root: Path, class_names: Sequence[str]) -> bool:
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            return False
        for cls in class_names:
            if not (split_dir / cls).is_dir():
                return False
    return True


def _find_species_final_dataset(species_dir: Path, *, final_dataset_dirname: str, class_names: Sequence[str]) -> Path:
    """
    Choose the "best" final dataset directory for a given species folder.

    Tries (in order):
    - species_dir itself (if it contains train/val/test layout)
    - species_dir/<child>/<final dataset>
    - species_dir/<child>/<grandchild>/<final dataset>
    Picks newest (mtime) among candidates.
    """
    candidates: List[Path] = []

    if _has_expected_split_layout(species_dir, class_names):
        candidates.append(species_dir)

    for child in sorted(species_dir.iterdir()) if species_dir.is_dir() else []:
        if not child.is_dir():
            continue
        cand = child / final_dataset_dirname
        if _has_expected_split_layout(cand, class_names):
            candidates.append(cand)
            continue
        # one more level
        try:
            for grand in sorted(child.iterdir()):
                if not grand.is_dir():
                    continue
                cand2 = grand / final_dataset_dirname
                if _has_expected_split_layout(cand2, class_names):
                    candidates.append(cand2)
        except PermissionError:
            continue

    if not candidates:
        raise FileNotFoundError(
            f"Could not find '{final_dataset_dirname}' with expected split/class layout under: {species_dir}"
        )

    candidates.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return candidates[-1]


@dataclass(frozen=True)
class SpeciesIndex:
    species: str
    dataset_root: Path  # points to ".../final dataset"
    class_to_idx: Dict[str, int]
    # split -> list[(relative_path_from_dataset_root, label_idx)]
    samples: Dict[str, List[Tuple[str, int]]]


def _build_species_index(
    *,
    species: str,
    dataset_root: Path,
    class_names: Sequence[str],
) -> SpeciesIndex:
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    samples: Dict[str, List[Tuple[str, int]]] = {}
    for split in SPLITS:
        split_samples: List[Tuple[str, int]] = []
        for cls in class_names:
            cls_dir = dataset_root / split / cls
            for p in _list_images(cls_dir):
                rel = str(p.relative_to(dataset_root))
                split_samples.append((rel, class_to_idx[cls]))
        split_samples.sort(key=lambda x: x[0])
        samples[split] = split_samples
    return SpeciesIndex(species=species, dataset_root=dataset_root, class_to_idx=class_to_idx, samples=samples)


def _save_species_index(idx: SpeciesIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "species": idx.species,
        "dataset_root": str(idx.dataset_root),
        "class_to_idx": dict(idx.class_to_idx),
        "samples": {k: list(v) for k, v in idx.samples.items()},
        "created_at": _now(),
    }
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _load_species_index(path: Path) -> SpeciesIndex:
    payload = json.loads(path.read_text())
    return SpeciesIndex(
        species=str(payload["species"]),
        dataset_root=Path(payload["dataset_root"]),
        class_to_idx={str(k): int(v) for k, v in payload["class_to_idx"].items()},
        samples={str(k): [(str(p), int(y)) for (p, y) in v] for k, v in payload["samples"].items()},
    )


@dataclass(frozen=True)
class TrainJob:
    name: str
    train_species: Tuple[str, ...]
    val_species: Tuple[str, ...]
    test_in_species: Tuple[str, ...]
    test_out_species: Tuple[str, ...]


def _build_jobs(species: Sequence[str]) -> List[TrainJob]:
    sp_all = tuple(species)
    jobs: List[TrainJob] = [
        TrainJob(
            name="global_all_species",
            train_species=sp_all,
            val_species=sp_all,
            test_in_species=sp_all,
            test_out_species=(),
        )
    ]
    for sp in sp_all:
        train_species = tuple([s for s in sp_all if s != sp])
        jobs.append(
            TrainJob(
                name=f"leaveout_{_slugify(sp)}",
                train_species=train_species,
                val_species=train_species,
                test_in_species=train_species,
                test_out_species=(sp,),
            )
        )
    return jobs


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _ensure_species_index(
    *,
    species: str,
    dataset_root: Path,
    class_names: Sequence[str],
    index_cache_dir: Path,
    reindex: bool,
) -> Path:
    cache_path = index_cache_dir / f"{_slugify(species)}.json"
    if cache_path.exists() and (not reindex):
        try:
            idx = _load_species_index(cache_path)
            if idx.dataset_root == dataset_root:
                return cache_path
        except Exception:
            pass
    idx = _build_species_index(species=species, dataset_root=dataset_root, class_names=class_names)
    _save_species_index(idx, cache_path)
    return cache_path


def _resolve_roots(root: Path, datasets_dir: Optional[Path], output_dir: Optional[Path]) -> Tuple[Path, Path, Path]:
    root = root.expanduser().resolve()
    if datasets_dir is None:
        cand = root / "datasets"
        datasets_dir = cand if cand.is_dir() else root
    else:
        datasets_dir = datasets_dir.expanduser().resolve()

    if output_dir is None:
        output_dir = root / "outputs"
    else:
        output_dir = output_dir.expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    return root, datasets_dir, output_dir


def _discover_species_dirs(datasets_dir: Path) -> Dict[str, Path]:
    if not datasets_dir.is_dir():
        raise FileNotFoundError(datasets_dir)
    out: Dict[str, Path] = {}
    for p in sorted(datasets_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        out[p.name] = p
    if not out:
        raise RuntimeError(f"No species directories found under: {datasets_dir}")
    return out


def _print_gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }

    try:
        import torch
    except Exception as e:
        print(f"[gpu] torch import failed: {e}")
        return info

    info["cuda_available"] = bool(torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("[gpu] CUDA not available (torch.cuda.is_available() == False)")
        return info

    n = int(torch.cuda.device_count())
    info["device_count"] = n
    devices = []
    for i in range(n):
        try:
            name = str(torch.cuda.get_device_name(i))
        except Exception:
            name = "unknown"
        devices.append({"id": i, "name": name})
    info["devices"] = devices

    print(f"[gpu] CUDA_VISIBLE_DEVICES={info['cuda_visible_devices']!r}")
    print(f"[gpu] torch sees {n} GPU(s):")
    for d in devices:
        print(f"  - cuda:{d['id']}: {d['name']}")
    return info


def _print_cpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "os_cpu_count": os.cpu_count(),
        "slurm": {},
    }

    slurm_vars = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NODELIST",
        "SLURM_NTASKS",
        "SLURM_CPUS_ON_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
    ]
    slurm = {k: (os.environ.get(k) or "") for k in slurm_vars}
    info["slurm"] = slurm

    print(f"[cpu] os.cpu_count()={info['os_cpu_count']}")
    if any(v for v in slurm.values()):
        # only print if we're in a Slurm context
        print("[cpu] slurm allocation:")
        for k in slurm_vars:
            v = slurm.get(k, "")
            if v:
                print(f"  - {k}={v}")
    return info


def _assemble_samples(
    *,
    index_files: Dict[str, str],
    index_cache: Dict[str, SpeciesIndex],
    species: Sequence[str],
    split: str,
) -> List[Tuple[Path, int, str]]:
    """
    Return list[(absolute_path, label, species_name)] for a given split and subset of species.
    """
    out: List[Tuple[Path, int, str]] = []
    for sp in species:
        if sp not in index_cache:
            idx_path = Path(index_files[sp])
            index_cache[sp] = _load_species_index(idx_path)
        idx = index_cache[sp]
        for rel, y in idx.samples[split]:
            out.append((idx.dataset_root / rel, y, sp))
    return out


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _train_one_job(
    *,
    job: TrainJob,
    out_dir: Path,
    index_files: Dict[str, str],
    index_cache: Dict[str, SpeciesIndex],
    resnet: str,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    seed: int,
    gpu_id: Optional[int],
    amp: bool,
    checkpoint_every: int,
    resume: bool,
    force_retrain: bool,
) -> Dict[str, Any]:
    # Lazy imports so the parent process can do discovery without torch/torchvision if desired.
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import models, transforms
    import torchvision.transforms.functional as F

    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Pillow is required to train (PIL import failed): {e}") from e

    class RandomRotate90:
        def __call__(self, img):  # noqa: ANN001
            return F.rotate(img, random.choice([0, 90, 180, 270]))

    def build_resnet(name: str, num_classes: int = 2) -> nn.Module:
        resnet_fns = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }
        if name not in resnet_fns:
            raise ValueError(f"Unknown ResNet model '{name}'. Choose from {list(resnet_fns)}")
        net = resnet_fns[name](weights=None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        nn.init.xavier_uniform_(net.fc.weight)
        nn.init.zeros_(net.fc.bias)
        return net

    class ImagePathDataset(Dataset):
        def __init__(self, samples: Sequence[Tuple[Path, int, str]], tfm):  # noqa: ANN001
            self.samples = list(samples)
            self.tfm = tfm

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, i: int):  # noqa: ANN001
            p, y, _sp = self.samples[i]
            with Image.open(p) as im:  # type: ignore[attr-defined]
                im = im.convert("RGB")
            if self.tfm is not None:
                im = self.tfm(im)
            return im, int(y)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"
    done_marker = out_dir / "DONE"
    metrics_path = out_dir / "metrics.json"
    config_path = out_dir / "config.json"

    if done_marker.exists() and best_ckpt.exists() and metrics_path.exists() and (not force_retrain):
        print(f"[{job.name}] ✅ already done → {out_dir}")
        return json.loads(metrics_path.read_text())

    # device selection
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))
        device = torch.device(f"cuda:{int(gpu_id)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    _seed_everything(int(seed))
    torch.backends.cudnn.benchmark = True

    # transforms
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

    # samples
    train_samples = _assemble_samples(index_files=index_files, index_cache=index_cache, species=job.train_species, split="train")
    val_samples = _assemble_samples(index_files=index_files, index_cache=index_cache, species=job.val_species, split="val")
    test_in_samples = _assemble_samples(index_files=index_files, index_cache=index_cache, species=job.test_in_species, split="test")
    test_out_samples = _assemble_samples(index_files=index_files, index_cache=index_cache, species=job.test_out_species, split="test")

    if not train_samples:
        raise RuntimeError(f"[{job.name}] No training samples found.")
    if not val_samples:
        raise RuntimeError(f"[{job.name}] No validation samples found.")

    # datasets
    train_ds = ImagePathDataset(train_samples, train_tfm)
    val_ds = ImagePathDataset(val_samples, plain_tfm)
    test_in_ds = ImagePathDataset(test_in_samples, plain_tfm) if test_in_samples else None
    test_out_ds = ImagePathDataset(test_out_samples, plain_tfm) if test_out_samples else None

    # class weights from train set
    n_classes = 2
    counts = [0] * n_classes
    train_targets: List[int] = []
    for _p, y, _sp in train_samples:
        yi = int(y)
        if 0 <= yi < n_classes:
            counts[yi] += 1
        train_targets.append(yi)
    total = sum(counts)
    class_weights_list: List[float] = []
    missing: List[int] = []
    for c in range(n_classes):
        n_c = int(counts[c])
        if n_c <= 0:
            class_weights_list.append(0.0)
            missing.append(c)
        else:
            class_weights_list.append(float(total) / float(n_c))
    if missing:
        print(f"[{job.name}] WARNING: missing classes in train split: {missing} (counts={counts})")
    class_weights = torch.tensor(class_weights_list, dtype=torch.float32)

    sample_weights = [float(class_weights[t]) for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    pin_memory = device.type == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    def _make_eval_loader(ds: Optional[Dataset]) -> Optional[DataLoader]:
        if ds is None:
            return None
        return DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=pin_memory,
            persistent_workers=bool(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        )

    test_in_dl = _make_eval_loader(test_in_ds)
    test_out_dl = _make_eval_loader(test_out_ds)

    # model / loss / optimiser
    model = build_resnet(str(resnet), num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=float(lr))

    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp and device.type == "cuda"))

    # resume
    start_epoch = 1
    best_val_acc = -1.0
    best_epoch = -1

    if resume and last_ckpt.exists():
        ckpt = torch.load(str(last_ckpt), map_location=device)
        try:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_acc = float(ckpt.get("best_val_acc", best_val_acc))
            best_epoch = int(ckpt.get("best_epoch", best_epoch))
            print(f"[{job.name}] 🔁 resume from {last_ckpt} (epoch {start_epoch})")
        except Exception as e:
            print(f"[{job.name}] WARNING: failed to resume from {last_ckpt}: {e}")

    # save config early (so a killed job still leaves breadcrumbs)
    config = {
        "job": {
            "name": job.name,
            "train_species": list(job.train_species),
            "val_species": list(job.val_species),
            "test_in_species": list(job.test_in_species),
            "test_out_species": list(job.test_out_species),
        },
        "hyperparams": {
            "resnet": str(resnet),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "num_workers": int(num_workers),
            "seed": int(seed),
            "amp": bool(amp),
            "checkpoint_every": int(checkpoint_every),
        },
        "device": {
            "type": device.type,
            "gpu_id": int(gpu_id) if gpu_id is not None else None,
        },
        "counts": {
            "train_total": int(len(train_samples)),
            "val_total": int(len(val_samples)),
            "test_in_total": int(len(test_in_samples)),
            "test_out_total": int(len(test_out_samples)),
            "train_class_counts": list(map(int, counts)),
        },
        "created_at": _now(),
    }
    config_path.write_text(json.dumps(config, indent=2))

    stop_requested = {"flag": False}

    def _handle_signal(signum, _frame):  # noqa: ANN001
        stop_requested["flag"] = True
        print(f"[{job.name}] received signal {signum}; will stop after current step and checkpoint")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    def run_epoch(loader: DataLoader, *, train: bool, desc: str) -> Tuple[float, float]:
        model.train(mode=train)
        tot_loss = 0.0
        tot_corr = 0
        tot = 0

        for xb, yb in loader:
            if stop_requested["flag"]:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=bool(scaler.is_enabled())):
                    out = model(xb)
                    loss = criterion(out, yb)

                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            tot_loss += float(loss.detach().item()) * int(xb.size(0))
            tot_corr += int((out.detach().argmax(1) == yb).sum().item())
            tot += int(xb.size(0))

        loss_avg = (tot_loss / tot) if tot else float("nan")
        acc = (tot_corr / tot) if tot else float("nan")
        return loss_avg, acc

    def save_checkpoint(*, epoch: int) -> None:
        payload = {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "best_val_acc": float(best_val_acc),
            "best_epoch": int(best_epoch),
            "job_name": job.name,
            "saved_at": _now(),
        }
        torch.save(payload, str(last_ckpt))

    # training
    print(f"[{job.name}] start | device={device} | train={len(train_ds)} val={len(val_ds)}")

    ep_times: List[float] = []
    started = time.time()
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    last_val_acc = float("nan")

    if start_epoch > int(epochs):
        print(f"[{job.name}] already at epoch {start_epoch-1}/{epochs}; skipping training loop")

    for ep in range(int(start_epoch), int(epochs) + 1):
        t0 = time.time()
        tr_loss, _tr_acc = run_epoch(train_dl, train=True, desc=f"train {ep:03d}")
        val_loss, val_acc = run_epoch(val_dl, train=False, desc=f"val {ep:03d}")

        last_train_loss = float(tr_loss)
        last_val_loss = float(val_loss)
        last_val_acc = float(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_epoch = int(ep)
            torch.save(
                {
                    "epoch": int(ep),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "val_acc": float(val_acc),
                    "saved_at": _now(),
                },
                str(best_ckpt),
            )
            print(f"[{job.name}] ✅ new best val_acc={val_acc:.4f} → {best_ckpt}")

        if (int(ep) % int(checkpoint_every) == 0) or stop_requested["flag"] or (ep == int(epochs)):
            save_checkpoint(epoch=int(ep))

        ep_times.append(time.time() - t0)
        eta_s = (sum(ep_times) / len(ep_times)) * max(0, int(epochs) - int(ep))
        eta = time.strftime("%H:%M:%S", time.gmtime(eta_s))
        print(
            f"[{job.name}] ep {ep:03d}/{epochs} | tr_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | "
            f"val_acc {val_acc:.4f} | ep_time {ep_times[-1]:.1f}s | ETA {eta}"
        )

        if stop_requested["flag"]:
            print(f"[{job.name}] stopping early due to signal; checkpointed at epoch {ep}")
            break

    # evaluation helper
    def eval_loader(loader: Optional[DataLoader], tag: str) -> Dict[str, Any]:
        if loader is None:
            return {"tag": tag, "enabled": False}
        best = torch.load(str(best_ckpt), map_location=device) if best_ckpt.exists() else None
        if best is None:
            return {"tag": tag, "enabled": False, "error": "best checkpoint missing"}
        eval_model = build_resnet(str(resnet), num_classes=n_classes).to(device)
        eval_model.load_state_dict(best["model"])
        eval_model.eval()

        tot_loss = 0.0
        tot_corr = 0
        tot = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                out = eval_model(xb)
                loss = criterion(out, yb)
                tot_loss += float(loss.item()) * int(xb.size(0))
                tot_corr += int((out.argmax(1) == yb).sum().item())
                tot += int(xb.size(0))
        return {
            "tag": tag,
            "enabled": True,
            "loss": (tot_loss / tot) if tot else float("nan"),
            "acc": (tot_corr / tot) if tot else float("nan"),
            "n": int(tot),
        }

    test_in_metrics = eval_loader(test_in_dl, "test_in")
    test_out_metrics = eval_loader(test_out_dl, "test_out")

    metrics = {
        "job": job.name,
        "timestamp": _now(),
        "out_dir": str(out_dir),
        "device": {"type": device.type, "gpu_id": int(gpu_id) if gpu_id is not None else None},
        "epochs_target": int(epochs),
        "epoch_start": int(start_epoch),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "last_train_loss": float(last_train_loss),
        "last_val_loss": float(last_val_loss),
        "last_val_acc": float(last_val_acc),
        "train_n": int(len(train_samples)),
        "val_n": int(len(val_samples)),
        "test_in": test_in_metrics,
        "test_out": test_out_metrics,
        "paths": {"best_ckpt": str(best_ckpt), "last_ckpt": str(last_ckpt)},
        "wall_time_s": float(time.time() - started),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    if not stop_requested["flag"]:
        done_marker.write_text(_now() + "\n")
        print(f"[{job.name}] ✅ done → {out_dir}")
    else:
        print(f"[{job.name}] ⏸️  stopped early (resume later) → {out_dir}")

    return metrics


def _worker_main(
    *,
    worker_id: int,
    gpu_id: Optional[int],
    jobs_q,  # mp.Queue
    results_q,  # mp.Queue
    shared: Dict[str, Any],
) -> None:
    index_cache: Dict[str, SpeciesIndex] = {}
    while True:
        try:
            job = jobs_q.get(timeout=1)
        except queue.Empty:
            continue
        if job is None:
            return

        try:
            metrics = _train_one_job(
                job=job,
                out_dir=Path(shared["output_dir"]) / job.name,
                index_files=shared["index_files"],
                index_cache=index_cache,
                resnet=shared["resnet"],
                epochs=shared["epochs"],
                batch_size=shared["batch_size"],
                lr=shared["lr"],
                num_workers=shared["num_workers"],
                seed=shared["seed"],
                gpu_id=gpu_id,
                amp=shared["amp"],
                checkpoint_every=shared["checkpoint_every"],
                resume=shared["resume"],
                force_retrain=shared["force_retrain"],
            )
            results_q.put({"ok": True, "job": job.name, "metrics": metrics, "gpu_id": gpu_id, "worker_id": worker_id})
        except Exception as e:
            results_q.put(
                {"ok": False, "job": job.name, "error": repr(e), "gpu_id": gpu_id, "worker_id": worker_id}
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train patch classifiers: global + leave-one-species-out, parallelized across GPUs."
    )
    ap.add_argument("--root", type=str, required=True, help="Root folder (contains datasets and/or outputs).")
    ap.add_argument(
        "--datasets-dir",
        type=str,
        default="",
        help="Optional datasets dir (defaults to <root>/datasets if exists, else <root>).",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output dir (defaults to <root>/outputs).",
    )
    ap.add_argument("--gpus", type=int, default=1, help="How many GPUs to use concurrently (default: 1).")
    ap.add_argument("--dry-run", action="store_true", help="Print discovered species/jobs/GPU info and exit.")
    ap.add_argument("--list-jobs", action="store_true", help="List planned jobs (and their indices) and exit.")
    ap.add_argument("--job", type=str, default="", help="Run only a single job by name (use --list-jobs).")
    ap.add_argument("--job-index", type=int, default=-1, help="Run only a single job by index (use --list-jobs).")
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=-1,
        help="(Single-job mode) Which visible GPU id to use. -1 = auto (default).",
    )

    ap.add_argument("--resnet", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per training process.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP).")
    ap.add_argument("--checkpoint-every", type=int, default=1, help="Save last checkpoint every N epochs.")
    ap.add_argument("--no-resume", action="store_true", help="Disable resume from checkpoints.")
    ap.add_argument("--force", action="store_true", help="Force retrain even if DONE exists.")

    ap.add_argument(
        "--final-dataset-dirname",
        type=str,
        default=DEFAULT_FINAL_DATASET_DIRNAME,
        help="Folder name to look for under each species (default: 'final dataset').",
    )
    ap.add_argument(
        "--classes",
        type=str,
        default=",".join(DEFAULT_CLASS_NAMES),
        help="Comma-separated class folder names (default: firefly,background).",
    )
    ap.add_argument("--reindex", action="store_true", help="Rebuild species file indexes even if cached.")

    args = ap.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root)
    datasets_dir = Path(args.datasets_dir) if str(args.datasets_dir or "").strip() else None
    output_dir = Path(args.output_dir) if str(args.output_dir or "").strip() else None
    _root, datasets_dir, output_dir = _resolve_roots(root, datasets_dir, output_dir)

    class_names = tuple([c.strip() for c in str(args.classes).split(",") if c.strip()])
    if len(class_names) != 2:
        raise SystemExit(f"--classes must contain exactly 2 class names (got {class_names})")

    print(f"[main] {_now()}")
    print(f"[main] root={_root}")
    print(f"[main] datasets_dir={datasets_dir}")
    print(f"[main] output_dir={output_dir}")

    _print_cpu_info()
    gpu_info = _print_gpu_info()
    visible_gpu_count = int(gpu_info.get("device_count", 0) or 0)

    # decide worker count
    requested_gpus = max(0, int(args.gpus))
    if visible_gpu_count <= 0:
        if requested_gpus > 0:
            print("[main] WARNING: requested GPUs but CUDA not available; running on CPU sequentially.")
        worker_gpus: List[Optional[int]] = [None]
    else:
        use = min(requested_gpus, visible_gpu_count) if requested_gpus > 0 else 1
        if requested_gpus > visible_gpu_count:
            print(f"[main] WARNING: requested {requested_gpus} GPU(s) but only {visible_gpu_count} visible; using {use}.")
        worker_gpus = list(range(use))

    # discover species
    species_dirs = _discover_species_dirs(datasets_dir)
    print(f"[main] discovered {len(species_dirs)} species under {datasets_dir}: {list(species_dirs.keys())}")

    # pick per-species dataset root (newest "final dataset")
    species_to_dataset_root: Dict[str, Path] = {}
    for sp, sp_dir in species_dirs.items():
        ds_root = _find_species_final_dataset(
            sp_dir, final_dataset_dirname=str(args.final_dataset_dirname), class_names=class_names
        )
        species_to_dataset_root[sp] = ds_root
        print(f"[main] {sp}: dataset_root={ds_root}")

    # jobs: global + leave-one-out (stable ordering: global first, then leaveout in sorted species order)
    species = tuple(sorted(species_to_dataset_root.keys()))
    jobs = _build_jobs(species)

    print(f"[main] queued {len(jobs)} job(s) ({1} global + {len(species)} LOSO)")
    manifest_path = output_dir / "manifest.json"
    _write_json_atomic(
        manifest_path,
        {
            "created_at": _now(),
            "root": str(_root),
            "datasets_dir": str(datasets_dir),
            "output_dir": str(output_dir),
            "final_dataset_dirname": str(args.final_dataset_dirname),
            "classes": list(class_names),
            "species_to_dataset_root": {k: str(v) for k, v in sorted(species_to_dataset_root.items())},
            "jobs": [
                {
                    "index": i,
                    "name": j.name,
                    "train_species": list(j.train_species),
                    "val_species": list(j.val_species),
                    "test_in_species": list(j.test_in_species),
                    "test_out_species": list(j.test_out_species),
                }
                for i, j in enumerate(jobs)
            ],
        },
    )
    print(f"[main] wrote manifest: {manifest_path}")

    if args.list_jobs:
        for i, j in enumerate(jobs):
            print(f"[job {i:03d}] {j.name} | train={len(j.train_species)} | test_out={list(j.test_out_species)}")
        return 0

    if args.dry_run:
        print("[main] dry-run: exiting before indexing/training")
        return 0

    single_job_name = str(args.job or "").strip()
    single_job_index = int(args.job_index)
    if single_job_name and (single_job_index >= 0):
        raise SystemExit("Use only one of --job or --job-index (not both).")

    selected_job: Optional[TrainJob] = None
    if single_job_name:
        for j in jobs:
            if j.name == single_job_name:
                selected_job = j
                break
        if selected_job is None:
            raise SystemExit(f"--job not found: {single_job_name!r} (use --list-jobs)")
    elif single_job_index >= 0:
        if single_job_index < 0 or single_job_index >= len(jobs):
            raise SystemExit(f"--job-index out of range: {single_job_index} (valid: 0..{len(jobs)-1})")
        selected_job = jobs[single_job_index]

    # build/load per-species indexes
    index_cache_dir = output_dir / "_species_indexes"
    index_files: Dict[str, str] = {}

    if selected_job is not None:
        required_species = sorted(
            set(selected_job.train_species + selected_job.val_species + selected_job.test_in_species + selected_job.test_out_species)
        )
    else:
        required_species = list(species)

    for sp in required_species:
        ds_root = species_to_dataset_root[sp]
        cache_path = _ensure_species_index(
            species=sp,
            dataset_root=ds_root,
            class_names=class_names,
            index_cache_dir=index_cache_dir,
            reindex=bool(args.reindex),
        )
        index_files[sp] = str(cache_path)

    # Single-job execution path (Slurm job arrays friendly).
    if selected_job is not None:
        index_cache: Dict[str, SpeciesIndex] = {}
        if visible_gpu_count > 0:
            gpu_id = 0 if int(args.gpu_id) < 0 else int(args.gpu_id)
        else:
            gpu_id = None
        try:
            _train_one_job(
                job=selected_job,
                out_dir=output_dir / selected_job.name,
                index_files=index_files,
                index_cache=index_cache,
                resnet=str(args.resnet),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                num_workers=int(args.num_workers),
                seed=int(args.seed),
                gpu_id=gpu_id,
                amp=(not bool(args.no_amp)),
                checkpoint_every=max(1, int(args.checkpoint_every)),
                resume=(not bool(args.no_resume)),
                force_retrain=bool(args.force),
            )
            return 0
        except Exception as e:
            print(f"[main] job failed: {selected_job.name}: {e}")
            return 1

    # sanity: indexes exist for all species (multi-job mode)
    missing = [sp for sp in species if sp not in index_files]
    if missing:
        raise RuntimeError(f"Missing indexes for species: {missing}")

    # shared training config
    shared: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "index_files": index_files,
        "resnet": str(args.resnet),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "amp": (not bool(args.no_amp)),
        "checkpoint_every": max(1, int(args.checkpoint_every)),
        "resume": (not bool(args.no_resume)),
        "force_retrain": bool(args.force),
    }

    # CPU-only sequential path
    if worker_gpus == [None]:
        failed = 0
        index_cache: Dict[str, SpeciesIndex] = {}
        for job in jobs:
            try:
                _train_one_job(
                    job=job,
                    out_dir=output_dir / job.name,
                    index_files=index_files,
                    index_cache=index_cache,
                    resnet=shared["resnet"],
                    epochs=shared["epochs"],
                    batch_size=shared["batch_size"],
                    lr=shared["lr"],
                    num_workers=shared["num_workers"],
                    seed=shared["seed"],
                    gpu_id=None,
                    amp=False,
                    checkpoint_every=shared["checkpoint_every"],
                    resume=shared["resume"],
                    force_retrain=shared["force_retrain"],
                )
            except Exception as e:
                failed += 1
                print(f"[main] job failed: {job.name}: {e}")
        return 1 if failed else 0

    # multi-GPU worker path
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    jobs_q: mp.Queue = mp.Queue()
    results_q: mp.Queue = mp.Queue()

    for job in jobs:
        jobs_q.put(job)
    for _ in worker_gpus:
        jobs_q.put(None)

    workers: List[mp.Process] = []
    for wid, gpu_id in enumerate(worker_gpus):
        p = mp.Process(
            target=_worker_main,
            kwargs={
                "worker_id": int(wid),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
                "jobs_q": jobs_q,
                "results_q": results_q,
                "shared": shared,
            },
            daemon=False,
        )
        p.start()
        workers.append(p)
        print(f"[main] started worker {wid} on gpu_id={gpu_id}")

    ok = 0
    fail = 0
    remaining = len(jobs)
    while remaining > 0:
        msg = results_q.get()
        remaining -= 1
        if msg.get("ok"):
            ok += 1
            print(f"[main] ✅ finished {msg.get('job')} (worker={msg.get('worker_id')} gpu={msg.get('gpu_id')})")
        else:
            fail += 1
            print(f"[main] ❌ failed {msg.get('job')} (worker={msg.get('worker_id')} gpu={msg.get('gpu_id')}): {msg.get('error')}")

    for p in workers:
        p.join()

    print(f"[main] done: ok={ok} fail={fail} outputs at {output_dir}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
