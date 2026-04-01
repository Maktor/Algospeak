"""
train.py

Training loop for the dual BERTweet model.

Both encoders are trained jointly using supervised InfoNCE loss.
The best checkpoint (lowest validation loss) is saved to poc/checkpoints/best_model.pt.

Usage:
    uv run python poc/src/train.py
"""

import sys
import json
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import DualEncoderModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_config() -> dict:
    with open(BASE_DIR / "poc" / "config.yaml") as f:
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
            sup_ids, sup_mask, unsup_ids, unsup_mask, labels = [
                b.to(device) for b in batch
            ]

            if train:
                optimizer.zero_grad()

            with autocast(enabled=use_fp16):
                loss, _, _ = model(sup_ids, sup_mask, unsup_ids, unsup_mask, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["gradient_clip"]
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item()

            if train and step % 100 == 0:
                print(
                    f"    Step {step}/{len(loader)} | loss: {loss.item():.4f}"
                )

    return total_loss / max(len(loader), 1)


def main():
    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    prepared_dir = BASE_DIR / cfg["prepared_dir"]
    ckpt_dir     = BASE_DIR / cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("\nLoading prepared data...")
    train_ds = load_dataset(prepared_dir / "train.pt")
    val_ds   = load_dataset(prepared_dir / "val.pt")
    print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=2, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=2, pin_memory=device.type == "cuda",
    )

    # Model
    print(f"\nInitializing model: {cfg['model_name']}")
    model = DualEncoderModel(cfg["model_name"], cfg["temperature"]).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total parameters: {n_params:.1f}M")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["warmup_steps"],
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=cfg.get("fp16", True) and device.type == "cuda")

    best_val_loss    = float("inf")
    patience_counter = 0
    history          = []

    print(f"\nTraining for up to {cfg['num_epochs']} epochs "
          f"(early stop patience={cfg['early_stopping_patience']})...\n")

    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()

        train_loss = run_epoch(
            model, train_loader, optimizer, scaler, scheduler, device, cfg, train=True
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, scaler, scheduler, device, cfg, train=False
        )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{cfg['num_epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
        })

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss":         best_val_loss,
                    "config":           cfg,
                },
                ckpt_dir / "best_model.pt",
            )
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(
                f"  No improvement ({patience_counter}/{cfg['early_stopping_patience']})"
            )
            if patience_counter >= cfg["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete.")
    print(f"Best val loss:  {best_val_loss:.4f}")
    print(f"Checkpoint:     {ckpt_dir / 'best_model.pt'}")
    print(f"History:        {ckpt_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()
