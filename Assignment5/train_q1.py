"""
Q1 – Baseline: Fine-tune only the classification head of ViT-S on CIFAR-100.
No LoRA – all backbone weights are frozen.
"""

import argparse
import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")   # use local cache only

import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

import wandb


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_data(batch_size: int):
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomCrop(224, padding=4),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4865, 0.4409],
                    std=[0.2673, 0.2564, 0.2761]),
    ])
    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4865, 0.4409],
                    std=[0.2673, 0.2564, 0.2761]),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True,  download=True, transform=train_tf)
    valset   = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=val_tf)

    train_loader = DataLoader(trainset, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(valset,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # WandB
    run = wandb.init(
        project="assignment5-vit-lora",
        name="q1_baseline_no_lora",
        config=vars(args),
        tags=["baseline", "no-lora"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ViT-S pre-trained on ImageNet (loaded from local download)
    VIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vit_small_local")
    model = ViTForImageClassification.from_pretrained(
        VIT_PATH,
        num_labels=100,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )

    # Freeze all backbone; keep only the classifier head trainable
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    wandb.config.update({"trainable_params": trainable, "total_params": total})

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    train_loader, val_loader = get_data(args.batch_size)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        # ---- Train ----
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(pixel_values=x).logits
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tr_loss    += loss.item() * x.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total   += x.size(0)

        scheduler.step()
        train_loss = tr_loss    / tr_total
        train_acc  = tr_correct / tr_total

        # ---- Validate ----
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(pixel_values=x).logits
                loss   = criterion(logits, y)
                v_loss    += loss.item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total   += x.size(0)

        val_loss = v_loss    / v_total
        val_acc  = v_correct / v_total

        epoch_time = time.time() - epoch_start
        print(f"[Epoch {epoch+1:>2}/{args.epochs}]  time={epoch_time:.1f}s  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        wandb.log({
            "epoch":      epoch + 1,
            "train/loss": train_loss,
            "train/acc":  train_acc,
            "val/loss":   val_loss,
            "val/acc":    val_acc,
            "lr":         scheduler.get_last_lr()[0],
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  → Saved best model (val_acc={best_val_acc:.4f})")

    print(f"\nBest val acc: {best_val_acc:.4f}")
    wandb.summary["best_val_acc"] = best_val_acc
    wandb.finish()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q1 Baseline – ViT-S head only fine-tune")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--save-path",  type=str,   default="models/q1_baseline_vit_s.pth")
    args = parser.parse_args()
    main(args)
