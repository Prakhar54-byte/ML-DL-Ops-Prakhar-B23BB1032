#!/usr/bin/env bash
# ============================================================
# run_all.sh  –  Assignment 5 full pipeline
# Run INSIDE the Docker container: bash /work/run_all.sh
# ============================================================
set -euo pipefail

echo "======================================================"
echo " Assignment 5  |  Roll: B23BB1032"
echo "======================================================"

echo ""
echo "[0] Device check:"
python -c "
import torch
print('  torch:', torch.__version__)
print('  CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU:', torch.cuda.get_device_name(0))
else:
    print('  Running on CPU (training will be slower)')
"

# ── Q1 Baseline ───────────────────────────────────────────────
echo ""
echo "[Q1] Baseline – ViT-S head-only fine-tune on CIFAR-100 …"
python /work/train_q1.py \
    --epochs 10 \
    --batch-size 128 \
    --lr 3e-4 \
    --save-path /work/models/q1_baseline_vit_s.pth

# ── Q1 Evaluate baseline ──────────────────────────────────────
echo ""
echo "[Q1] Evaluating baseline …"
python /work/eval_q1.py \
    --model-path /work/models/q1_baseline_vit_s.pth \
    --mode baseline \
    --batch-size 256

# ── Q1 LoRA sweep (9 combos) ──────────────────────────────────
echo ""
echo "[Q1] LoRA sweep – ranks {2,4,8} × alphas {2,4,8} …"
python /work/train_lora.py \
    --epochs 10 \
    --batch-size 128 \
    --lr 3e-4 \
    --ranks 2 4 8 \
    --alphas 2 4 8 \
    --dropout 0.1 \
    --save-dir /work/models/lora

# ── Q1 Optuna ─────────────────────────────────────────────────
echo ""
echo "[Q1] Optuna hyperparameter search …"
python /work/optuna_search.py \
    --trials 9 \
    --optuna-epochs 5 \
    --final-epochs 10 \
    --batch-size 128 \
    --save-path /work/models/lora_optuna_best.pth

# ── Q2 Train ResNet-18 ────────────────────────────────────────
echo ""
echo "[Q2] Training ResNet-18 on CIFAR-10 from scratch …"
python /work/q2_train_resnet.py \
    --epochs 50 \
    --batch-size 128 \
    --lr 0.1 \
    --save-path /work/models/resnet18_cifar10.pth

# ── Q2i FGSM ─────────────────────────────────────────────────
echo ""
echo "[Q2i] FGSM attack (scratch vs ART) …"
python /work/q2_fgsm.py \
    --model-path /work/models/resnet18_cifar10.pth \
    --n-samples 1000 \
    --epsilons 0.01 0.03 0.05 0.1

# ── Q2ii Detectors ───────────────────────────────────────────
echo ""
echo "[Q2ii] Adversarial detectors (PGD & BIM) …"
python /work/q2_detector.py \
    --model-path /work/models/resnet18_cifar10.pth \
    --epochs 15 \
    --batch-size 128 \
    --eps 0.03 \
    --max-iter 40 \
    --n-train 5000 \
    --n-test 1000

echo ""
echo "======================================================"
echo "DONE! All outputs in /work/models/ and /work/results/"
echo "======================================================"
