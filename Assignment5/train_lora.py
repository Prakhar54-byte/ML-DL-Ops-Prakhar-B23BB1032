"""
Q1 – LoRA experiments on ViT-S / CIFAR-100.

Loops over all (rank, alpha) combinations specified on the CLI  and logs
train/val loss+accuracy, per-epoch gradient norms on LoRA weight matrices,
and a class-wise test-accuracy histogram to WandB.

Usage example (all 9 combos):
    python train_lora.py --ranks 2 4 8 --alphas 2 4 8 --dropout 0.1 --epochs 10
"""

import argparse
import os
import time
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

import wandb

# ── CIFAR-100 class names ──────────────────────────────────────────────────
CIFAR100_CLASSES = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle",
    "bicycle","bottle","bowl","boy","bridge","bus","butterfly","camel",
    "can","castle","caterpillar","cattle","chair","chimpanzee","clock",
    "cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox","girl","hamster",
    "house","kangaroo","keyboard","lamp","lawn_mower","leopard","lion",
    "lizard","lobster","man","maple_tree","motorcycle","mountain","mouse",
    "mushroom","oak_tree","orange","orchid","otter","palm_tree","pear",
    "pickup_truck","pine_tree","plain","plate","poppy","porcupine",
    "possum","rabbit","raccoon","ray","road","rocket","rose","sea",
    "seal","shark","shrew","skunk","skyscraper","snail","snake","spider",
    "squirrel","streetcar","sunflower","sweet_pepper","table","tank",
    "telephone","television","tiger","tractor","train","trout","tulip",
    "turtle","wardrobe","whale","willow_tree","wolf","woman","worm",
]


# ── Data ──────────────────────────────────────────────────────────────────

def get_loaders(batch_size: int):
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomCrop(224, padding=4),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761]),
    ])
    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761]),
    ])
    tr = torchvision.datasets.CIFAR100("./data", train=True,  download=True, transform=train_tf)
    va = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=val_tf)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True),
            DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))


# ── Gradient-norm hook ─────────────────────────────────────────────────────

def register_grad_hooks(model):
    """Attach hooks to every LoRA weight (lora_A / lora_B) to record grad norms."""
    grad_norms: dict[str, list[float]] = {}
    handles = []

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            grad_norms[name] = []

            def make_hook(n):
                def hook(grad):
                    grad_norms[n].append(grad.norm().item())
                return hook

            h = param.register_hook(make_hook(name))
            handles.append(h)

    return grad_norms, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ── Single run ────────────────────────────────────────────────────────────

def run_experiment(rank: int, alpha: int, dropout: float,
                   epochs: int, batch_size: int, lr: float,
                   save_dir: str, exp_no: int, total_exps: int):

    run_name = f"lora_r{rank}_a{alpha}_d{dropout}"
    print(f"\n{'='*60}")
    print(f"Experiment {exp_no}/{total_exps}  |  rank={rank}  alpha={alpha}  dropout={dropout}")
    print(f"{'='*60}")

    run = wandb.init(
        project="assignment5-vit-lora",
        name=run_name,
        config={
            "rank": rank, "alpha": alpha, "dropout": dropout,
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "experiment_no": exp_no,
        },
        tags=["lora", f"rank{rank}", f"alpha{alpha}"],
        reinit=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ──
    VIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vit_small_local")
    base = ViTForImageClassification.from_pretrained(
        VIT_PATH,
        num_labels=100,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "key", "value"],
    )
    model = get_peft_model(base, peft_cfg)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    wandb.config.update({"trainable_params": trainable, "total_params": total})

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader, val_loader = get_loaders(batch_size)
    os.makedirs(save_dir, exist_ok=True)

    best_val_acc = 0.0
    best_ckpt    = os.path.join(save_dir, f"{run_name}.pth")

    for epoch in range(epochs):
        epoch_start = time.time()
        # ── Train ──
        model.train()
        grad_norms, handles = register_grad_hooks(model)

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

        remove_hooks(handles)
        scheduler.step()

        train_loss = tr_loss    / tr_total
        train_acc  = tr_correct / tr_total

        # Average grad-norm per LoRA layer name (epoch summary)
        grad_log = {}
        for lname, norms in grad_norms.items():
            if norms:
                short = lname.replace("base_model.model.", "")
                grad_log[f"grad_norm/{short}"] = float(np.mean(norms))

        # ── Validate ──
        model.eval()
        v_loss = v_correct = v_total = 0
        class_correct = [0] * 100
        class_total   = [0] * 100

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(pixel_values=x).logits
                loss   = criterion(logits, y)
                v_loss    += loss.item() * x.size(0)
                preds  = logits.argmax(1)
                v_correct += (preds == y).sum().item()
                v_total   += x.size(0)
                for label, pred in zip(y.cpu(), preds.cpu()):
                    class_total[label]   += 1
                    class_correct[label] += int(pred == label)

        val_loss = v_loss    / v_total
        val_acc  = v_correct / v_total

        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1:>2}/{epochs}  time={epoch_time:.1f}s  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        log_dict = {
            "epoch":      epoch + 1,
            "train/loss": train_loss,
            "train/acc":  train_acc,
            "val/loss":   val_loss,
            "val/acc":    val_acc,
            "lr":         scheduler.get_last_lr()[0],
            **grad_log,
        }
        wandb.log(log_dict)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)

    # ── Class-wise histogram (final epoch) ──
    class_acc = [
        class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        for i in range(100)
    ]
    fig, ax = plt.subplots(figsize=(22, 5))
    ax.bar(range(100), class_acc, color="steelblue")
    ax.set_xticks(range(100))
    ax.set_xticklabels(CIFAR100_CLASSES, rotation=90, fontsize=6)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Class-wise Test Accuracy  |  rank={rank}  alpha={alpha}")
    plt.tight_layout()
    hist_path = os.path.join(save_dir, f"{run_name}_classwise.png")
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)
    wandb.log({"class_acc_histogram": wandb.Image(hist_path)})

    wandb.summary["best_val_acc"]      = best_val_acc
    wandb.summary["trainable_params"]  = trainable
    wandb.finish()

    print(f"  Best val acc = {best_val_acc:.4f}  →  {best_ckpt}")
    return best_val_acc, trainable


# ── Entry point ───────────────────────────────────────────────────────────

def main(args):
    ranks   = args.ranks
    alphas  = args.alphas
    combos  = list(product(ranks, alphas))
    total   = len(combos)

    results = []
    for i, (r, a) in enumerate(combos, start=1):
        val_acc, trainable = run_experiment(
            rank=r, alpha=a, dropout=args.dropout,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, save_dir=args.save_dir,
            exp_no=i, total_exps=total,
        )
        results.append({"rank": r, "alpha": a, "dropout": args.dropout,
                        "val_acc": val_acc, "trainable": trainable})

    # ── Summary table ──
    print("\n" + "="*70)
    print(f"{'LoRA':8} {'Rank':6} {'Alpha':6} {'Dropout':8} {'Val Acc':10} {'Trainable':12}")
    print("-"*70)
    for r in results:
        print(f"{'yes':8} {r['rank']:<6} {r['alpha']:<6} {r['dropout']:<8} "
              f"{r['val_acc']:<10.4f} {r['trainable']:<12,}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q1 LoRA sweep – ViT-S on CIFAR-100")
    parser.add_argument("--epochs",     type=int,   nargs="+", default=[10])
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--ranks",      type=int,   nargs="+", default=[2, 4, 8])
    parser.add_argument("--alphas",     type=int,   nargs="+", default=[2, 4, 8])
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--save-dir",   type=str,   default="models/lora")
    args = parser.parse_args()
    # If --epochs passed as a list, take first element as scalar
    if isinstance(args.epochs, list):
        args.epochs = args.epochs[0]
    main(args)
