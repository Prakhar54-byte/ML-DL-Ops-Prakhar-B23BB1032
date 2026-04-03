"""
Q1 – Comprehensive evaluation script.

Works for both the baseline model and LoRA-wrapped models.
Outputs:
  - Overall test accuracy
  - Class-wise accuracy histogram (saved + logged to WandB)
  - Per-class table printed to stdout
  - Optional: push model to HuggingFace Hub

Usage:
  # Baseline
  python eval_q1.py --model-path models/q1_baseline_vit_s.pth --mode baseline

  # LoRA checkpoint
  python eval_q1.py --model-path models/lora/lora_r4_a4_d0.1.pth \
      --mode lora --rank 4 --alpha 4 --dropout 0.1

  # Push to HuggingFace
  python eval_q1.py --model-path models/lora/lora_r4_a4_d0.1.pth \
      --mode lora --rank 4 --alpha 4 --push-to-hub YOUR_HF_REPO
"""

import argparse
import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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


def get_test_loader(batch_size: int):
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761]),
    ])
    dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=tf)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False, num_workers=4, pin_memory=True)


def load_model(args, device):
    VIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vit_small_local")
    base = ViTForImageClassification.from_pretrained(
        VIT_PATH,
        num_labels=100,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    if args.mode == "lora":
        peft_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=True,
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=["query", "key", "value"],
        )
        model = get_peft_model(base, peft_cfg)
    else:
        model = base

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    return model.to(device)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args, device)
    model.eval()

    run_name = (f"eval_{args.mode}_r{args.rank}_a{args.alpha}"
                if args.mode == "lora" else "eval_baseline")
    wandb.init(project="assignment5-vit-lora",
               name=run_name, config=vars(args),
               tags=["eval", args.mode])

    loader = get_test_loader(args.batch_size)
    criterion = nn.CrossEntropyLoss()

    total = correct = 0
    test_loss = 0.0
    class_correct = [0] * 100
    class_total   = [0] * 100

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(pixel_values=x).logits
            test_loss += criterion(logits, y).item() * x.size(0)
            preds      = logits.argmax(1)
            correct   += (preds == y).sum().item()
            total     += x.size(0)
            for label, pred in zip(y.cpu(), preds.cpu()):
                class_total[label]   += 1
                class_correct[label] += int(pred == label)

    test_loss /= total
    test_acc   = correct / total

    class_acc = [
        class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        for i in range(100)
    ]

    # ── Print table ──
    print(f"\n{'Class':<22} {'Correct':>8} {'Total':>6} {'Acc':>7}")
    print("-" * 48)
    for i, name in enumerate(CIFAR100_CLASSES):
        print(f"{name:<22} {class_correct[i]:>8} {class_total[i]:>6} {class_acc[i]:>7.4f}")
    print("-" * 48)
    print(f"{'OVERALL':<22} {correct:>8} {total:>6} {test_acc:>7.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # ── Histogram ──
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(22, 5))
    colors  = ["#e74c3c" if a < 0.5 else "#2ecc71" for a in class_acc]
    ax.bar(range(100), class_acc, color=colors)
    ax.axhline(test_acc, color="navy", linestyle="--", linewidth=1.5,
               label=f"Mean={test_acc:.3f}")
    ax.set_xticks(range(100))
    ax.set_xticklabels(CIFAR100_CLASSES, rotation=90, fontsize=5.5)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    mode_str = (f"LoRA r={args.rank} α={args.alpha}" if args.mode == "lora"
                else "Baseline (head only)")
    ax.set_title(f"CIFAR-100 Class-wise Test Accuracy  [{mode_str}]")
    ax.legend()
    plt.tight_layout()
    hist_path = f"results/{run_name}_classwise_hist.png"
    fig.savefig(hist_path, dpi=130)
    plt.close(fig)
    print(f"\nHistogram saved to {hist_path}")

    wandb.log({
        "test/loss": test_loss,
        "test/acc":  test_acc,
        "class_acc_histogram": wandb.Image(hist_path),
    })
    # Log per-class accuracy as a WandB Table
    table = wandb.Table(columns=["class", "accuracy", "correct", "total"])
    for i, name in enumerate(CIFAR100_CLASSES):
        table.add_data(name, class_acc[i], class_correct[i], class_total[i])
    wandb.log({"class_accuracy_table": table})

    wandb.summary["test_acc"]  = test_acc
    wandb.summary["test_loss"] = test_loss

    # ── Optional HuggingFace push ──
    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        print(f"Model pushed to HuggingFace Hub: {args.push_to_hub}")
        wandb.summary["hf_repo"] = args.push_to_hub

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q1 Evaluation – ViT-S on CIFAR-100")
    parser.add_argument("--model-path",   type=str,   required=True)
    parser.add_argument("--mode",         type=str,   default="baseline",
                        choices=["baseline", "lora"])
    parser.add_argument("--rank",         type=int,   default=4)
    parser.add_argument("--alpha",        type=int,   default=4)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--push-to-hub",  type=str,   default=None,
                        help="HuggingFace repo id, e.g. username/vit-s-cifar100-lora")
    args = parser.parse_args()
    main(args)
