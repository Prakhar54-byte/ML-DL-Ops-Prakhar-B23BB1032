"""
Q2 Part i – Train ResNet-18 on CIFAR-10 from scratch (target ≥ 72% test acc).

Run inside Docker container:
    python q2_train_resnet.py --epochs 50 --save-path models/resnet18_cifar10.pth
"""

import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader

import wandb


def get_loaders(batch_size: int):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])
    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])
    tr = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    va = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=val_tf)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True),
            DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))


def build_resnet18(num_classes: int = 10):
    """ResNet-18 adapted for 32×32 CIFAR-10 input (no downsampling stem)."""
    net = models.resnet18(weights=None)
    # Replace 7×7 stride-2 conv with 3×3 stride-1 for small images
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def main(args):
    wandb.init(project="assignment5-q2",
               name="q2_train_resnet18_cifar10",
               config=vars(args),
               tags=["q2", "resnet18", "clean-training"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_resnet18().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    train_loader, val_loader = get_loaders(args.batch_size)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * x.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total   += x.size(0)
        scheduler.step()

        # ── Validate ──
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                v_loss    += criterion(logits, y).item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total   += x.size(0)

        train_loss = tr_loss / tr_total;  train_acc = tr_correct / tr_total
        val_loss   = v_loss  / v_total;   val_acc   = v_correct  / v_total

        print(f"[{epoch+1:>3}/{args.epochs}]  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        wandb.log({"epoch": epoch+1,
                   "train/loss": train_loss, "train/acc": train_acc,
                   "val/loss":   val_loss,   "val/acc":  val_acc,
                   "lr": scheduler.get_last_lr()[0]})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  → Best model saved (val_acc={best_val_acc:.4f})")

    print(f"\nBest val acc: {best_val_acc:.4f}")
    if best_val_acc < 0.72:
        print("WARNING: target ≥ 72% not reached – consider more epochs.")
    wandb.summary["best_val_acc"] = best_val_acc
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10 from scratch")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.1)
    parser.add_argument("--save-path",  type=str,
                        default="models/resnet18_cifar10.pth")
    args = parser.parse_args()
    main(args)
