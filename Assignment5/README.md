# Assignment 5 – ViT + LoRA on CIFAR-100 & Adversarial Attacks on CIFAR-10

**Roll Number:** B23BB1032  
**Deadline:** 03/04/2026

> **GitHub Branch:** [Link to Assignment 5 Branch](https://github.com/Prakhar54-byte/ML-DL-Ops-Prakhar-B23BB1032/tree/Assignment_05)
> **WandB Project:** [Link to Weights & Biases Dashboard](https://wandb.ai/b23bb1032-iit-jodhpur/Assignmet_05/weave)
> **HuggingFace Model:** [Link to Best Q1 Model](https://huggingface.co/YOUR_HF_USERNAME/MODEL_NAME)

---

## Project Structure

```
assignment_5/
├── Dockerfile
├── requirements.txt
├── run_all.sh               # Full pipeline runner
│
├── train_q1.py              # Q1 – Baseline ViT-S (head-only fine-tune)
├── train_lora.py            # Q1 – LoRA sweep (all rank × alpha combos)
├── eval_q1.py               # Q1 – Evaluation + class-wise histogram
├── optuna_search.py         # Q1 – Optuna hyperparameter search
│
├── q2_train_resnet.py       # Q2 – Train ResNet-18 on CIFAR-10 from scratch
├── q2_fgsm.py               # Q2i – FGSM (scratch vs IBM ART)
├── q2_detector.py           # Q2ii – Adversarial detectors (PGD & BIM)
│
├── models/                  # Saved model weights
└── results/                 # Plots and histograms
```

---

## Setup

### 1. Build Docker Image

```bash
docker build -t assignment5 .
```

### 2. Run Container (with GPU)

```bash
docker run --dns 8.8.8.8 --gpus all -it --rm \
    -v $(pwd):/work \
    -w /work \
    assignment5 /bin/bash
```

### 3. Install Dependencies (inside container)

```bash
pip install -r requirements.txt
```

> **Note:** The Docker image already runs `pip install -r requirements.txt` at build time.  
> Only run this manually if dependencies change.

### 4. Set WandB API Key

```bash
export WANDB_API_KEY=your_api_key_here
wandb login
```

---

## Q1 – ViT-S Fine-tuning on CIFAR-100

### Step 1: Baseline (head only, no LoRA)

```bash
python train_q1.py --epochs 10 --batch-size 128 --lr 3e-4 \
    --save-path models/q1_baseline_vit_s.pth
```

### Step 2: LoRA Sweep (all 9 combinations of rank ∈ {2,4,8} × alpha ∈ {2,4,8})

```bash
python train_lora.py \
    --epochs 10 --batch-size 128 --lr 3e-4 \
    --ranks 2 4 8 --alphas 2 4 8 --dropout 0.1 \
    --save-dir models/lora
```

### Step 3: Evaluate Baseline

```bash
python eval_q1.py --model-path models/q1_baseline_vit_s.pth \
    --mode baseline --batch-size 256
```

### Step 4: Evaluate LoRA Model

```bash
python eval_q1.py --model-path models/lora/lora_r4_a4_d0.1.pth \
    --mode lora --rank 4 --alpha 4 --batch-size 256
```

### Step 5: Optuna Hyperparameter Search

```bash
python optuna_search.py --trials 9 --optuna-epochs 5 --final-epochs 10 \
    --save-path models/lora_optuna_best.pth
```

### Step 6: Push Best Model to HuggingFace

```bash
python eval_q1.py --model-path models/lora_optuna_best.pth \
    --mode lora --rank 4 --alpha 4 \
    --push-to-hub B23BB1032/vit-small-cifar100-lora
```

---

## Q2 – Adversarial Attacks using IBM ART

### Step 1: Train ResNet-18 on CIFAR-10 from Scratch (target ≥ 72%)

```bash
python q2_train_resnet.py --epochs 50 --batch-size 128 --lr 0.1 \
    --save-path models/resnet18_cifar10.pth
```

### Step 2: FGSM Attack – Scratch vs IBM ART (Q2 Part i)

```bash
python q2_fgsm.py --model-path models/resnet18_cifar10.pth \
    --n-samples 1000 --epsilons 0.01 0.03 0.05 0.1
```

### Step 3: Adversarial Detectors – PGD & BIM (Q2 Part ii)

```bash
python q2_detector.py --model-path models/resnet18_cifar10.pth \
    --epochs 15 --batch-size 128 --eps 0.03 --max-iter 40 \
    --n-train 5000 --n-test 1000
```

### Run Everything at Once

```bash
bash run_all.sh
```

---

## Q1 Results

### Experiment 1 Training Logs: Rank 2, Alpha 2, Dropout 0.1

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|---------------|-----------------|-------------------|---------------------|
| 1     | 1.2159        | 0.4672          | 72.51%            | 86.57%              |
| 2     | 0.3830        | 0.3861          | 88.43%            | 88.16%              |
| 3     | 0.3202        | 0.3617          | 89.99%            | 88.84%              |
| 4     | 0.2802        | 0.3483          | 91.18%            | 89.39%              |
| 5     | 0.2543        | 0.3354          | 92.07%            | 89.58%              |
| 6     | 0.2350        | 0.3292          | 92.50%            | 89.68%              |
| 7     | 0.2182        | 0.3245          | 93.16%            | 89.94%              |
| 8     | 0.2091        | 0.3231          | 93.40%            | 90.03%              |
| 9     | 0.2017        | 0.3210          | 93.74%            | 90.11%              |
| 10    | 0.1991        | 0.3210          | 93.77%            | 90.08%              |

### Testing Table (all configurations)

| LoRA | Rank | Alpha | Dropout | Test Accuracy | Trainable Params |
|------|------|-------|---------|---------------|------------------|
| No   | –    | –     | –       | 80.87%        | 38,500           |
| Yes  | 2    | 2     | 0.1     | 90.11%        | 93,796           |
| Yes  | 2    | 4     | 0.1     | 89.80%        | 93,796           |
| Yes  | 2    | 8     | 0.1     | 89.92%        | 93,796           |
| Yes  | 4    | 2     | 0.1     | 90.09%        | 149,092          |
| Yes  | 4    | 4     | 0.1     | 89.97%        | 149,092          |
| Yes  | 4    | 8     | 0.1     | 90.20%        | 149,092          |
| Yes  | 8    | 2     | 0.1     | 89.96%        | 259,684          |
| Yes  | 8    | 4     | 0.1     | %             | 259,684          |
| Yes  | 8    | 8     | 0.1     | %             | 259,684          |
| **Best (Optuna)** | ? | ? | ? | % (Best)      | ?                |

---

## Q2 Results

### FGSM Attack – Accuracy vs Perturbation Strength

| ε     | Clean Acc | FGSM Scratch | FGSM ART |
|-------|-----------|--------------|----------|
| 0.01  | 89.50%    | 70.00%       | 76.00%   |
| 0.03  | 89.50%    | 34.10%       | 40.70%   |
| 0.05  | 89.50%    | 18.70%       | 25.40%   |
| 0.10  | 89.50%    | 8.10%        | 14.30%   |

### Adversarial Detection Accuracy

| Attack | Detection Accuracy |
|--------|--------------------|
| PGD    | %                  |
| BIM    | %                  |

### Analysis: PGD vs BIM Detection Difficulty
*Insert your brief analysis here comparing the detection difficulties of PGD versus BIM attacks based on your accuracy observations...*

---

## Links

- **GitHub Branch:** [Link to Assignment 5 Branch](https://github.com/Prakhar54-byte/ML-DL-Ops-Prakhar-B23BB1032/tree/Assignment_05)
- **WandB:** [Link to Weights & Biases Dashboard](https://wandb.ai/b23bb1032-iit-jodhpur/Assignmet_05/weave)
- **HuggingFace:** [Link to Best Q1 Model](https://huggingface.co/YOUR_HF_USERNAME/MODEL_NAME)
