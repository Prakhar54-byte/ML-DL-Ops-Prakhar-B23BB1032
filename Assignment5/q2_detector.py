"""
Q2 Part ii – Adversarial Detection using ResNet-34.

Two separate detectors are trained:
  (a) Detector for PGD-generated adversarial examples
  (b) Detector for BIM-generated adversarial examples

Each detector is a binary classifier (clean=0, adversarial=1).
Target detection accuracy ≥ 70%.

Also logs 10 sample images per attack type (FGSM scratch, FGSM ART, PGD, BIM) to WandB.

Usage:
    python q2_detector.py --model-path models/resnet18_cifar10.pth \
        --epochs 15 --batch-size 128
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

import wandb

# ART imports
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod, FastGradientMethod
from art.estimators.classification import PyTorchClassifier

# ── Constants ─────────────────────────────────────────────────────────────
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]


# ── Helper: build ResNet-18 (victim model) ────────────────────────────────

def build_resnet18_victim():
    net = models.resnet18(weights=None)
    net.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc     = nn.Linear(net.fc.in_features, 10)
    return net


# ── Helper: build ResNet-34 (detector) ────────────────────────────────────

def build_resnet34_detector():
    net = models.resnet34(weights=None)
    net.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc     = nn.Linear(net.fc.in_features, 2)  # binary: clean vs adversarial
    return net


# ── Data loading ──────────────────────────────────────────────────────────

def load_cifar10_normalized(train: bool, n: int = None):
    """Returns (x_norm float32 numpy (N,3,32,32), y int64 numpy (N,))."""
    tf = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.CIFAR10("./data", train=train, download=True, transform=tf)
    if n is not None:
        indices = np.random.choice(len(ds), n, replace=False)
        imgs   = np.stack([ds[i][0].numpy() for i in indices])
        labels = np.array([ds[i][1]         for i in indices])
    else:
        imgs   = np.stack([ds[i][0].numpy() for i in range(len(ds))])
        labels = np.array([ds[i][1]         for i in range(len(ds))])
    # Normalize
    x_norm = (imgs - MEAN[:, None, None]) / STD[:, None, None]
    return x_norm.astype(np.float32), labels.astype(np.int64)


# ── Generate adversarial examples using ART ───────────────────────────────

def make_art_classifier(victim_model, device):
    return PyTorchClassifier(
        model=victim_model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=10,
        optimizer=torch.optim.SGD(victim_model.parameters(), lr=0.01),
        clip_values=(
            float((-MEAN / STD).min()),
            float(((1 - MEAN) / STD).max()),
        ),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )


def generate_pgd(art_clf, x_clean, eps=0.03, eps_step=0.007, max_iter=40):
    attack = ProjectedGradientDescent(
        estimator=art_clf, eps=eps, eps_step=eps_step,
        max_iter=max_iter, norm=np.inf, targeted=False)
    return attack.generate(x=x_clean)


def generate_bim(art_clf, x_clean, eps=0.03, eps_step=0.007, max_iter=40):
    attack = BasicIterativeMethod(
        estimator=art_clf, eps=eps, eps_step=eps_step,
        max_iter=max_iter, norm=np.inf, targeted=False)
    return attack.generate(x=x_clean)


def generate_fgsm_art(art_clf, x_clean, eps=0.03):
    attack = FastGradientMethod(estimator=art_clf, eps=eps, norm=np.inf)
    return attack.generate(x=x_clean)


# ── Train detector ────────────────────────────────────────────────────────

def train_detector(
    x_clean_tr, x_adv_tr,   # training arrays (normalized)
    x_clean_va, x_adv_va,   # validation arrays
    epochs, batch_size, lr,
    device, name: str,
):
    # Labels: clean=0, adversarial=1
    y_clean_tr = np.zeros(len(x_clean_tr), dtype=np.int64)
    y_adv_tr   = np.ones(len(x_adv_tr),   dtype=np.int64)
    y_clean_va = np.zeros(len(x_clean_va), dtype=np.int64)
    y_adv_va   = np.ones(len(x_adv_va),   dtype=np.int64)

    x_tr = np.concatenate([x_clean_tr, x_adv_tr])
    y_tr = np.concatenate([y_clean_tr, y_adv_tr])
    x_va = np.concatenate([x_clean_va, x_adv_va])
    y_va = np.concatenate([y_clean_va, y_adv_va])

    # Shuffle training data
    idx = np.random.permutation(len(x_tr))
    x_tr, y_tr = x_tr[idx], y_tr[idx]

    tr_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    va_ds = TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va))
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    detector = build_resnet34_detector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(detector.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc   = 0.0
    best_state = None

    for epoch in range(epochs):
        # Train
        detector.train()
        tr_loss = tr_correct = tr_total = 0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = detector(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * x.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total   += x.size(0)
        scheduler.step()

        # Validate
        detector.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(device), y.to(device)
                logits = detector(x)
                v_loss    += criterion(logits, y).item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total   += x.size(0)

        train_acc = tr_correct / tr_total
        val_acc   = v_correct  / v_total
        train_loss = tr_loss / tr_total
        val_loss   = v_loss  / v_total

        print(f"  [{name}] Epoch {epoch+1:>2}/{epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        wandb.log({
            f"{name}/epoch":      epoch + 1,
            f"{name}/train_loss": train_loss,
            f"{name}/train_acc":  train_acc,
            f"{name}/val_loss":   val_loss,
            f"{name}/val_acc":    val_acc,
        })

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.clone() for k, v in detector.state_dict().items()}

    detector.load_state_dict(best_state)
    return detector, best_acc


# ── Denormalize ───────────────────────────────────────────────────────────

def denorm(img: np.ndarray) -> np.ndarray:
    img = img * STD[:, None, None] + MEAN[:, None, None]
    img = np.clip(img, 0, 1)
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


# ── Main ─────────────────────────────────────────────────────────────────

def main(args):
    wandb.init(project="assignment5-q2",
               name="q2_adversarial_detectors",
               config=vars(args),
               tags=["q2", "detector", "pgd", "bim"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ── Load victim model ──
    print("\nLoading victim ResNet-18 …")
    victim = build_resnet18_victim().to(device)
    victim.load_state_dict(torch.load(args.model_path, map_location=device))
    victim.eval()
    art_clf = make_art_classifier(victim, device)

    # ── Generate adversarial examples ──
    print("Loading CIFAR-10 …")
    np.random.seed(42)
    x_train, _ = load_cifar10_normalized(train=True,  n=args.n_train)
    x_test,  _ = load_cifar10_normalized(train=False, n=args.n_test)

    print("Generating PGD adversarial examples (train) …")
    x_pgd_train = generate_pgd(art_clf, x_train,
                                eps=args.eps, eps_step=args.eps / 10,
                                max_iter=args.max_iter)
    print("Generating PGD adversarial examples (test) …")
    x_pgd_test  = generate_pgd(art_clf, x_test,
                                eps=args.eps, eps_step=args.eps / 10,
                                max_iter=args.max_iter)

    print("Generating BIM adversarial examples (train) …")
    x_bim_train = generate_bim(art_clf, x_train,
                                eps=args.eps, eps_step=args.eps / 10,
                                max_iter=args.max_iter)
    print("Generating BIM adversarial examples (test) …")
    x_bim_test  = generate_bim(art_clf, x_test,
                                eps=args.eps, eps_step=args.eps / 10,
                                max_iter=args.max_iter)

    # ── Also generate FGSM (scratch + ART) for WandB visual logs ──
    print("Generating FGSM (ART) examples for WandB samples …")
    x_fgsm_art = generate_fgsm_art(art_clf, x_test[:10], eps=args.eps)

    # FGSM scratch (manual) on first 10 test samples
    def fgsm_scratch(x_np, eps):
        x_t = torch.from_numpy(x_np).float().to(device)
        y_dummy = torch.zeros(len(x_np), dtype=torch.long, device=device)
        x_t.requires_grad = True
        loss = nn.CrossEntropyLoss()(victim(x_t), y_dummy)
        loss.backward()
        return (x_t + eps * x_t.grad.sign()).detach().cpu().numpy()

    x_fgsm_scratch = fgsm_scratch(x_test[:10], args.eps)

    # ── Log 10 samples per attack type to WandB ──
    print("Logging sample images to WandB …")
    for i in range(10):
        wandb.log({
            "samples/clean":        wandb.Image(denorm(x_test[i])),
            "samples/fgsm_scratch": wandb.Image(denorm(x_fgsm_scratch[i])),
            "samples/fgsm_art":     wandb.Image(denorm(x_fgsm_art[i])),
            "samples/pgd":          wandb.Image(denorm(x_pgd_test[i])),
            "samples/bim":          wandb.Image(denorm(x_bim_test[i])),
        })

    # ── Train Detector A (PGD) ──
    print("\n" + "="*60)
    print("Training Detector A – PGD")
    print("="*60)
    detector_pgd, best_pgd_acc = train_detector(
        x_train, x_pgd_train,
        x_test,  x_pgd_test,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device, name="detector_pgd",
    )
    torch.save(detector_pgd.state_dict(), "models/resnet34_detector_pgd.pth")
    print(f"  → Best PGD detection accuracy: {best_pgd_acc:.4f}")
    if best_pgd_acc < 0.70:
        print("  WARNING: target ≥70% not reached – consider more epochs or stronger attacks.")

    # ── Train Detector B (BIM) ──
    print("\n" + "="*60)
    print("Training Detector B – BIM")
    print("="*60)
    detector_bim, best_bim_acc = train_detector(
        x_train, x_bim_train,
        x_test,  x_bim_test,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device, name="detector_bim",
    )
    torch.save(detector_bim.state_dict(), "models/resnet34_detector_bim.pth")
    print(f"  → Best BIM detection accuracy: {best_bim_acc:.4f}")

    # ── Summary ──
    print("\n" + "="*50)
    print(f"  PGD detection accuracy: {best_pgd_acc:.4f}")
    print(f"  BIM detection accuracy: {best_bim_acc:.4f}")
    print("="*50)

    summary_table = wandb.Table(columns=["Attack", "Detection Accuracy"])
    summary_table.add_data("PGD", best_pgd_acc)
    summary_table.add_data("BIM", best_bim_acc)
    wandb.log({"detection_summary": summary_table})

    wandb.summary["pgd_detection_acc"] = best_pgd_acc
    wandb.summary["bim_detection_acc"] = best_bim_acc
    wandb.finish()
    print("\nDetector weights saved to models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q2 Part ii – Adversarial detectors")
    parser.add_argument("--model-path", type=str,
                        default="models/resnet18_cifar10.pth",
                        help="Path to trained ResNet-18 victim model")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--eps",        type=float, default=0.03,
                        help="Attack perturbation budget (epsilon)")
    parser.add_argument("--max-iter",   type=int,   default=40,
                        help="Max iterations for PGD / BIM")
    parser.add_argument("--n-train",    type=int,   default=5000,
                        help="Training samples used for detector")
    parser.add_argument("--n-test",     type=int,   default=1000,
                        help="Test samples used for detector evaluation")
    args = parser.parse_args()
    main(args)
