"""
Q1 – Optuna hyperparameter search for LoRA on ViT-S / CIFAR-100.

Search space: rank in {2,4,8}, alpha in {2,4,8}, dropout fixed at 0.1
After finding the best trial, re-trains for full epochs and saves the model.

Usage:
    python optuna_search.py --trials 9 --optuna-epochs 5 --final-epochs 10 \
        --save-path models/lora_optuna_best.pth
"""

import argparse
import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import optuna
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

import wandb

# ── Data ──────────────────────────────────────────────────────────────────

def get_loaders(batch_size: int):
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
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

# ── Training helper ────────────────────────────────────────────────────────

def _build_model(rank, alpha, dropout):
    VIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vit_small_local")
    base = ViTForImageClassification.from_pretrained(
        VIT_PATH,
        num_labels=100,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "key", "value"],
    )
    return get_peft_model(base, cfg)


def train_and_eval(rank, alpha, dropout, epochs, batch_size, lr, log_wandb=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _build_model(rank, alpha, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader, val_loader = get_loaders(batch_size)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(pixel_values=x).logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate only on last epoch (fast Optuna)
        if log_wandb or epoch == epochs - 1:
            model.eval()
            v_correct = v_total = 0
            v_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(pixel_values=x).logits
                    v_loss    += criterion(logits, y).item() * x.size(0)
                    v_correct += (logits.argmax(1) == y).sum().item()
                    v_total   += x.size(0)
            val_acc  = v_correct / v_total
            val_loss = v_loss    / v_total
            if log_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": v_loss / v_total,     # proxy
                    "val_loss":   val_loss,
                    "val_acc":    val_acc,
                })

    return val_acc, model


# ── Optuna objective ───────────────────────────────────────────────────────

_ARGS = None   # set in main

def objective(trial: optuna.Trial):
    rank    = trial.suggest_categorical("rank",    [2, 4, 8])
    alpha   = trial.suggest_categorical("alpha",   [2, 4, 8])
    dropout = 0.1

    print(f"\nTrial {trial.number}  rank={rank}  alpha={alpha}")
    val_acc, _ = train_and_eval(rank, alpha, dropout,
                                epochs=_ARGS.optuna_epochs,
                                batch_size=_ARGS.batch_size,
                                lr=_ARGS.lr)
    return val_acc


# ── Main ──────────────────────────────────────────────────────────────────

def main(args):
    global _ARGS
    _ARGS = args

    wandb.init(project="assignment5-vit-lora",
               name="optuna_lora_search",
               config=vars(args),
               tags=["optuna"])

    # ── Optuna search ──
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.GridSampler(
                                    {"rank": [2, 4, 8], "alpha": [2, 4, 8]}
                                ))
    study.optimize(objective, n_trials=args.trials)

    best_params = study.best_trial.params
    best_val    = study.best_trial.value
    print(f"\nBest params: {best_params}  val_acc={best_val:.4f}")
    wandb.summary["best_rank"]    = best_params["rank"]
    wandb.summary["best_alpha"]   = best_params["alpha"]
    wandb.summary["best_val_acc"] = best_val

    # ── Log Optuna results table ──
    table = wandb.Table(columns=["trial", "rank", "alpha", "val_acc"])
    for t in study.trials:
        table.add_data(t.number, t.params.get("rank"), t.params.get("alpha"), t.value)
    wandb.log({"optuna_results": table})

    # ── Final training with best config ──
    print(f"\nRe-training with best config for {args.final_epochs} epochs …")
    final_acc, best_model = train_and_eval(
        rank=best_params["rank"],
        alpha=best_params["alpha"],
        dropout=0.1,
        epochs=args.final_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_wandb=True,
    )
    print(f"Final val acc: {final_acc:.4f}")
    wandb.summary["final_val_acc"] = final_acc

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(best_model.state_dict(), args.save_path)
    print(f"Best model saved to {args.save_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna LoRA hyperparameter search")
    parser.add_argument("--trials",        type=int,   default=9,
                        help="Number of Optuna trials (max 9 for 3x3 grid)")
    parser.add_argument("--optuna-epochs", type=int,   default=5,
                        help="Epochs per trial (short run for speed)")
    parser.add_argument("--final-epochs",  type=int,   default=10,
                        help="Epochs for final best-config retraining")
    parser.add_argument("--batch-size",    type=int,   default=128)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--save-path",     type=str,
                        default="models/lora_optuna_best.pth")
    args = parser.parse_args()
    main(args)
