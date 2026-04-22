"""
experiments/four_class/src/train.py

Training loop for the four-class dual BERTweet experiment.
Reads prepared .pt files from experiments/four_class/data/prepared/.
Saves checkpoints to experiments/four_class/checkpoints/.

Usage:
    uv run python experiments/four_class/src/train.py
    uv run python experiments/four_class/src/train.py --temperature 0.1
    uv run python experiments/four_class/src/train.py --epochs 15
    uv run python experiments/four_class/src/train.py --fresh
"""

import sys
import json
import time
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]  # project root
EXP_DIR  = Path(__file__).resolve().parents[1]  # experiments/four_class

sys.path.insert(0, str(Path(__file__).parent))
from model import DualEncoderModel


def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else EXP_DIR / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_dataset(path: Path) -> TensorDataset:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return TensorDataset(
        data["sup_ids"],
        data["sup_mask"],
        data["unsup_ids"],
        data["unsup_mask"],
        data["labels"],
    )


def run_epoch(model, loader, optimizer, scaler, scheduler, device, cfg, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    use_fp16   = cfg.get("fp16", True) and device.type == "cuda"
    ctx = torch.no_grad if not train else torch.enable_grad

    with ctx():
        for step, batch in enumerate(loader, 1):
            sup_ids, sup_mask, unsup_ids, unsup_mask, labels = [b.to(device) for b in batch]
            if train:
                optimizer.zero_grad()
            with autocast(enabled=use_fp16):
                loss, _, _ = model(sup_ids, sup_mask, unsup_ids, unsup_mask, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            total_loss += loss.item()
            if train and step % 50 == 0:
                print(f"    Step {step}/{len(loader)} | loss: {loss.item():.4f}")

    return total_loss / max(len(loader), 1)


def compute_val_accuracy(model, train_ds, val_ds, device, cfg, samples_per_class=1000):
    model.eval()
    batch_sz    = cfg["batch_size"]
    num_classes = cfg["num_classes"]

    all_labels = train_ds.tensors[4]
    indices = []
    for cls in range(num_classes):
        cls_idx = (all_labels == cls).nonzero(as_tuple=True)[0].tolist()
        indices.extend(cls_idx[:samples_per_class])

    proto_embs = [[] for _ in range(num_classes)]
    proto_loader = DataLoader(Subset(train_ds, indices), batch_size=batch_sz, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in proto_loader:
            _, _, unsup_ids, unsup_mask, lbls = [b.to(device) for b in batch]
            embs = model.unsupervised(unsup_ids, unsup_mask).cpu().numpy()
            for i, lbl in enumerate(lbls.cpu().tolist()):
                proto_embs[lbl].append(embs[i])

    prototypes = np.zeros((num_classes, model.unsupervised.bert.config.hidden_size), dtype=np.float32)
    for cls in range(num_classes):
        if proto_embs[cls]:
            mean = np.mean(proto_embs[cls], axis=0)
            prototypes[cls] = mean / (np.linalg.norm(mean) + 1e-8)

    val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False, num_workers=0)
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in val_loader:
            _, _, unsup_ids, unsup_mask, lbls = [b.to(device) for b in batch]
            embs = model.unsupervised(unsup_ids, unsup_mask).cpu().numpy()
            preds = (embs @ prototypes.T).argmax(axis=1)
            all_preds.extend(preds.tolist())
            all_true.extend(lbls.cpu().tolist())

    return sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.temperature is not None:
        cfg["temperature"] = args.temperature
        print(f"Temperature overridden to {args.temperature}")
    if args.epochs is not None:
        cfg["num_epochs"] = args.epochs
        print(f"Max epochs overridden to {args.epochs}")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    prepared_dir = BASE_DIR / cfg["prepared_dir"]
    ckpt_dir     = BASE_DIR / cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume_path = ckpt_dir / "resume_checkpoint.pt"
    best_path   = ckpt_dir / "best_model.pt"

    print("\nLoading prepared data...")
    train_ds = load_dataset(prepared_dir / "train.pt")
    val_ds   = load_dataset(prepared_dir / "val.pt")
    print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    print(f"\nInitializing model: {cfg['model_name']}  (temperature={cfg['temperature']})")
    model = DualEncoderModel(cfg["model_name"], cfg["temperature"]).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total parameters: {n_params:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg["warmup_steps"], num_training_steps=total_steps)
    scaler = GradScaler(enabled=cfg.get("fp16", True) and device.type == "cuda")

    start_epoch = 1
    best_val_acc = 0.0
    patience_counter = 0
    history = []

    if not args.fresh and resume_path.exists():
        print(f"\nResume checkpoint found — loading from {resume_path}")
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        saved_temp = resume.get("config", {}).get("temperature")
        if saved_temp is not None and abs(saved_temp - cfg["temperature"]) > 1e-6:
            print(f"  WARNING: temperature changed ({saved_temp} → {cfg['temperature']}). Use --fresh to start over.")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        start_epoch      = resume["epoch"] + 1
        best_val_acc     = resume["best_val_metric"]
        patience_counter = resume["patience_counter"]
        history          = resume["history"]
        print(f"  Resumed from epoch {resume['epoch']} (best_val_acc={best_val_acc:.4f})")
    else:
        if args.fresh and resume_path.exists():
            resume_path.unlink()
            print("\nFresh start — deleted existing resume checkpoint.")

    print(f"\nTraining for up to {cfg['num_epochs']} epochs "
          f"(early stop patience={cfg['early_stopping_patience']}, metric=val_acc)...\n")

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, optimizer, scaler, scheduler, device, cfg, train=True)

        val_loader_tmp = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
        val_loss = run_epoch(model, val_loader_tmp, optimizer, scaler, scheduler, device, cfg, train=False)

        val_acc = compute_val_accuracy(model, train_ds, val_ds, device, cfg)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{cfg['num_epochs']} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | {elapsed:.1f}s")

        history.append({"epoch": epoch, "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6), "val_acc": round(val_acc, 6)})

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss, "val_acc": best_val_acc, "config": cfg}, best_path)
            print(f"  -> Best model saved (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{cfg['early_stopping_patience']})")

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_metric": best_val_acc, "patience_counter": patience_counter,
            "history": history, "total_steps": total_steps, "config": cfg,
        }, resume_path)

        with open(ckpt_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if patience_counter >= cfg["early_stopping_patience"]:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if resume_path.exists():
        resume_path.unlink()
        print("Resume checkpoint deleted (training complete).")

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
