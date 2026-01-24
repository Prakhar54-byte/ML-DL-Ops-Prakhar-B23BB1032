# ML-DL-Ops-Prakhar-B23BB1032

**Name:** Prakhar Chauhan  
**Roll Number:** B23BB1032  

---

## 📝 Project Overview
This repository contains the experiments for Assignment 1 of the DLOPs course. The objective is to analyze the performance of Deep Learning models (ResNet) and Machine Learning models (SVM) on MNIST and FashionMNIST datasets under various constraints (CPU vs GPU, Hyperparameter tuning).

**🔗 Colab Notebook Link:** https://colab.research.google.com/drive/1phC0UhzqrKJ_swj5Jz_fhyyYDIrgTUuW?usp=sharing 
> *Note: Experiments were performed using PyTorch with Automatic Mixed Precision (AMP) enabled.*

---

## 📊 Q1(a): Deep Learning Classification Results
**Dataset Split:** 70% Train | 10% Val | 20% Test  
**Models:** ResNet-18 & ResNet-50 (Pretrained=False)

### 1. MNIST Dataset
| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
| :---: | :---: | :---: | :---: | :---: |
| 16 | SGD | 0.001 | 98.0 | 98.6 |
| 16 | SGD | 0.0001 | 98.4 | 98.9 |
| 16 | Adam | 0.001 | 99.0 | 99.3 |
| 16 | Adam | 0.0001 | 99.2 | 99.4 |
| 32 | SGD | 0.001 | 97.9 | 98.4 |
| 32 | SGD | 0.0001 *(Opt)* | 98.3 | 98.8 |
| 32 | Adam | 0.001 *(Opt)* | 98.9 | 99.2 |
| 32 | Adam | 0.0001 | 99.1 | 99.3 |

### 2. FashionMNIST Dataset
| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
| :---: | :---: | :---: | :---: | :---: |
| 16 | SGD | 0.001 | 88.9 | 90.4 |
| 16 | SGD | 0.0001 | 89.4 | 91.0 |
| 16 | Adam | 0.001 | 91.38 | 92.4 |
| 16 | Adam | 0.0001 | 91.53 | 92.6 |
| 32 | SGD | 0.001 | 88.4 | 89.6 |
| 32 | SGD | 0.0001 *(Opt)* | 89.1 | 90.6 |
| 32 | Adam | 0.001 *(Opt)* | 90.9 | 92.0 |
| 32 | Adam | 0.0001 | 91.1 | 92.3 |

### 🔍 Analysis (Q1a)
Adam optimizer consistently outperformed SGD across both datasets due to adaptive learning rates. Lower learning rates (0.0001) improved stability and generalization. ResNet-50 achieved slightly higher accuracy than ResNet-18, but with increased computational cost. MNIST achieved significantly higher accuracy than FashionMNIST.

---

## 📈 Q1(b): SVM Classifier Results
**Task:** Training SVM with varying kernels ('poly', 'rbf') and hyperparameters.

| Dataset | Kernel | Best Hyperparameters | Test Accuracy (%) | Train Time (ms) |
| :--- | :---: | :--- | :---: | :---: |
| **MNIST** | Poly | C=10, degree=2 | 95.20 | 5204.85 |
| **MNIST** | RBF | C=10, gamma=scale | 95.55 | 8001.53 |
| **FashionMNIST** | Poly | C=10, degree=2 | 86.80 | 6729.06 |
| **FashionMNIST** | RBF | C=5, gamma=scale | 87.45 | 6820.15 |

### 🔍 Analysis (Q1b)
SVM achieved higher accuracy on MNIST than FashionMNIST due to simpler digit patterns. RBF kernel performed best overall, especially on FashionMNIST. Polynomial kernels provided competitive accuracy with slightly lower training time. Increasing C generally improved performance.

---

## ⚡ Q2: CPU vs GPU Performance Analysis
**Dataset:** FashionMNIST  
**Comparison:** Training time and FLOPs count for ResNet models on different hardware.

| Compute | Batch | Optim | LR | R-18 Acc (%) | R-18 Time (ms) | R-18 FLOPs | R-50 Acc (%) | R-50 Time (ms) | R-50 FLOPs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU** | 16 | SGD | 0.001 | 89.20 | 4,250,000 | 0.03G | 88.50 | 13,900,000 | 0.08G |
| **CPU** | 16 | Adam | 0.001 | 90.10 | 4,050,000 | 0.03G | 89.80 | 13,600,000 | 0.08G |
| **GPU** | 16 | SGD | 0.001 | 91.73 | 545,392 | 0.03G | 90.40 | 1,663,622 | 0.08G |
| **GPU** | 16 | Adam | 0.001 | 92.04 | 533,329 | 0.03G | 91.90 | 1,663,071 | 0.08G |

### 💻 System Analysis (Q2)
GPU training provides a significant speedup of approximately **7–8×** compared to CPU across all models. While FLOPs remain constant for a given architecture, GPU acceleration drastically reduces wall-clock training time. Deeper models such as ResNet-50 require substantially more computation, but provide only modest accuracy gains.

---
