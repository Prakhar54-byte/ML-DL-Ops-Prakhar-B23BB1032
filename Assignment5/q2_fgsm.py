"""
Q2 Part i – FGSM Attack: from-scratch vs IBM ART on CIFAR-10 / ResNet-18.

Steps:
  1. Load trained ResNet-18 (≥72% on clean data)
  2. Run FGSM from scratch (manual gradient-sign)
  3. Run FGSM via IBM ART
  4. Compare accuracy & log visual samples to WandB

Usage:
    python q2_fgsm.py --model-path models/resnet18_cifar10.pth --epsilons 0.01 0.03 0.05 0.1
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
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

import wandb

# ART imports
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

# ── Constants ─────────────────────────────────────────────────────────────
CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]
MEAN    = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD     = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)


# ── Model ─────────────────────────────────────────────────────────────────

def build_resnet18(num_classes: int = 10):
    net = models.resnet18(weights=None)
    net.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc     = nn.Linear(net.fc.in_features, num_classes)
    return net


# ── Data (unnormalized tensors + labels, for ART) ─────────────────────────

def load_raw_test(n_samples: int = 1000):
    """Return raw [0,1] images and labels (numpy arrays)."""
    tf = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    imgs   = torch.stack([ds[i][0] for i in range(n_samples)]).numpy()   # (N,3,32,32)
    labels = np.array([ds[i][1] for i in range(n_samples)])
    return imgs, labels


def normalize_np(imgs: np.ndarray) -> np.ndarray:
    """ImageNet-style normalization for ART (expects float32 [0,1])."""
    return (imgs - MEAN[:, None, None]) / STD[:, None, None]


def eval_accuracy(model, device, x_np: np.ndarray, y_np: np.ndarray,
                  batch_size: int = 256) -> float:
    """Evaluate on pre-normalized numpy arrays."""
    x_t = torch.from_numpy(x_np.astype(np.float32))
    y_t = torch.from_numpy(y_np.astype(np.int64))
    ds  = TensorDataset(x_t, y_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += x.size(0)
    return correct / total


# ── Manual FGSM (from scratch) ─────────────────────────────────────────────

def fgsm_scratch(model, device, x_clean: np.ndarray,
                 y: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Classic FGSM: x_adv = x + eps * sign(∇_x L).
    x_clean: normalized float32 numpy (N,3,32,32).
    Returns adversarial numpy array (same shape).
    """
    criterion = nn.CrossEntropyLoss()
    x_t = torch.from_numpy(x_clean.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y.astype(np.int64)).to(device)
    x_t.requires_grad = True

    model.eval()
    logits = model(x_t)
    loss   = criterion(logits, y_t)
    loss.backward()

    x_adv = x_t + epsilon * x_t.grad.sign()
    # Clip to valid normalized range
    min_v = torch.tensor((-MEAN / STD)[:, None, None],
                         dtype=torch.float32, device=device)
    max_v = torch.tensor(((1 - MEAN) / STD)[:, None, None],
                         dtype=torch.float32, device=device)
    x_adv = torch.max(torch.min(x_adv, max_v), min_v)
    return x_adv.detach().cpu().numpy()


# ── Visualization helpers ─────────────────────────────────────────────────

def denorm(img: np.ndarray) -> np.ndarray:
    """Denormalize (3,32,32) → uint8 (32,32,3)."""
    img = img * STD[:, None, None] + MEAN[:, None, None]
    img = np.clip(img, 0, 1)
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


def make_comparison_grid(x_clean, x_adv_scratch, x_adv_art,
                         y, pred_clean, pred_scratch, pred_art,
                         n: int = 10) -> plt.Figure:
    fig, axes = plt.subplots(3, n, figsize=(n * 2, 7))
    titles = ["Clean", "FGSM (scratch)", "FGSM (ART)"]
    for col in range(n):
        for row, (imgs, preds) in enumerate([
            (x_clean, pred_clean),
            (x_adv_scratch, pred_scratch),
            (x_adv_art, pred_art),
        ]):
            ax = axes[row, col]
            ax.imshow(denorm(imgs[col]))
            color = "green" if preds[col] == y[col] else "red"
            ax.set_title(f"{CLASSES[preds[col]]}", fontsize=7, color=color)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(titles[row], fontsize=8)
    plt.suptitle("Original vs Adversarial (FGSM scratch vs ART)", fontsize=10)
    plt.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────

def main(args):
    wandb.init(project="assignment5-q2",
               name="q2_fgsm_comparison",
               config=vars(args),
               tags=["q2", "fgsm", "art"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_resnet18().to(device)
    state  = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Load test data (raw [0,1] float32)
    print("Loading test data …")
    x_raw, y = load_raw_test(args.n_samples)   # [0,1], (N,3,32,32)
    x_norm   = normalize_np(x_raw)             # normalized

    # Clean accuracy
    acc_clean = eval_accuracy(model, device, x_norm, y)
    print(f"Clean accuracy: {acc_clean:.4f}")

    # ── ART classifier wrapper ──
    art_classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=10,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        clip_values=(
            float((-MEAN / STD).min()),
            float(((1 - MEAN) / STD).max()),
        ),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )

    results = []
    os.makedirs("results", exist_ok=True)

    for eps in args.epsilons:
        print(f"\n── ε = {eps} ──────────────────────────────")

        # FGSM from scratch
        x_adv_scratch = fgsm_scratch(model, device, x_norm, y, eps)
        acc_scratch   = eval_accuracy(model, device, x_adv_scratch, y)

        # FGSM via ART
        fgsm_art = FastGradientMethod(estimator=art_classifier, eps=eps, norm=np.inf)
        x_adv_art = fgsm_art.generate(x=x_norm)
        acc_art   = eval_accuracy(model, device, x_adv_art, y)

        print(f"  Clean:           {acc_clean:.4f}")
        print(f"  FGSM (scratch):  {acc_scratch:.4f}")
        print(f"  FGSM (ART):      {acc_art:.4f}")

        results.append({
            "epsilon": eps,
            "acc_clean":   acc_clean,
            "acc_scratch": acc_scratch,
            "acc_art":     acc_art,
        })

        # ── Predictions for visualization (first 10 samples) ──
        with torch.no_grad():
            pred_clean   = model(torch.from_numpy(x_norm[:10]).float().to(device)).argmax(1).cpu().numpy()
            pred_scratch = model(torch.from_numpy(x_adv_scratch[:10]).float().to(device)).argmax(1).cpu().numpy()
            pred_art     = model(torch.from_numpy(x_adv_art[:10]).float().to(device)).argmax(1).cpu().numpy()

        fig = make_comparison_grid(
            x_norm[:10], x_adv_scratch[:10], x_adv_art[:10],
            y[:10], pred_clean, pred_scratch, pred_art)
        fig_path = f"results/fgsm_eps{eps}_comparison.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)

        wandb.log({
            f"fgsm/eps{eps}/acc_clean":   acc_clean,
            f"fgsm/eps{eps}/acc_scratch": acc_scratch,
            f"fgsm/eps{eps}/acc_art":     acc_art,
            f"fgsm/eps{eps}/comparison":  wandb.Image(fig_path),
        })

        # ── Log 10 sample images to WandB (clean, scratch, art) ──
        for i in range(10):
            wandb.log({
                f"samples/eps{eps}/clean_{i}":   wandb.Image(denorm(x_norm[i])),
                f"samples/eps{eps}/scratch_{i}": wandb.Image(denorm(x_adv_scratch[i])),
                f"samples/eps{eps}/art_{i}":     wandb.Image(denorm(x_adv_art[i])),
            })

    # ── Summary table ──
    print("\n" + "="*55)
    print(f"{'ε':>8}  {'Clean':>8}  {'Scratch':>9}  {'ART':>8}")
    print("-"*55)
    for r in results:
        print(f"{r['epsilon']:>8.3f}  {r['acc_clean']:>8.4f}  "
              f"{r['acc_scratch']:>9.4f}  {r['acc_art']:>8.4f}")
    print("="*55)

    table = wandb.Table(columns=["epsilon", "acc_clean", "acc_fgsm_scratch", "acc_fgsm_art"])
    for r in results:
        table.add_data(r["epsilon"], r["acc_clean"], r["acc_scratch"], r["acc_art"])
    wandb.log({"fgsm_summary_table": table})

    # ── Perturbation vs accuracy drop ──
    eps_vals = [r["epsilon"]       for r in results]
    drop_s   = [acc_clean - r["acc_scratch"] for r in results]
    drop_a   = [acc_clean - r["acc_art"]     for r in results]

    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps_vals, drop_s, "o-", label="FGSM Scratch")
    ax.plot(eps_vals, drop_a, "s-", label="FGSM ART")
    ax.set_xlabel("Epsilon (perturbation strength)")
    ax.set_ylabel("Accuracy drop")
    ax.set_title("Perturbation Strength vs Performance Drop")
    ax.legend()
    ax.grid(True)
    fig2_path = "results/fgsm_perf_drop.png"
    fig2.savefig(fig2_path, dpi=120)
    plt.close(fig2)
    wandb.log({"fgsm/perf_drop_plot": wandb.Image(fig2_path)})

    wandb.finish()
    print("\nDone. Results in results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q2 Part i – FGSM comparison")
    parser.add_argument("--model-path", type=str,
                        default="models/resnet18_cifar10.pth")
    parser.add_argument("--n-samples",  type=int,   default=1000,
                        help="Number of test samples to attack")
    parser.add_argument("--epsilons",   type=float, nargs="+",
                        default=[0.01, 0.03, 0.05, 0.1],
                        help="Perturbation magnitudes to test")
    args = parser.parse_args()
    main(args)
