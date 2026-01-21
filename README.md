# ML-DL-Ops-Prakhar-B23BB1032


**Name:** [Your Name]
**Roll Number:** [Your Roll Number]

---

## 📝 Project Overview
This repository contains the experiments for Assignment 1 of the DLOPs course. The objective is to analyze the performance of Deep Learning models (ResNet) and Machine Learning models (SVM) on MNIST and FashionMNIST datasets under various constraints (CPU vs GPU, Hyperparameter tuning).

**🔗 Colab Notebook Link:** [PASTE YOUR COLAB LINK HERE]
> *Note: Experiments were performed using PyTorch with Automatic Mixed Precision (AMP) enabled.*

---

## 📊 Q1(a): Deep Learning Classification Results
**Dataset Split:** 70% Train | 10% Val | 20% Test
**Models:** ResNet-18 & ResNet-50 (Pretrained=False)

### 1. MNIST Dataset
| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
| :---: | :---: | :---: | :---: | :---: |
| 16 | SGD | 0.001 | | |
| 16 | SGD | 0.0001 | | |
| 16 | Adam | 0.001 | | |
| 16 | Adam | 0.0001 | | |
| 32 | SGD | 0.001 | | |
| 32 | SGD | 0.0001 *(Opt)* | | |
| 32 | Adam | 0.001 *(Opt)* | | |
| 32 | Adam | 0.0001 | | |

### 2. FashionMNIST Dataset
| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) |
| :---: | :---: | :---: | :---: | :---: |
| 16 | SGD | 0.001 | | |
| 16 | SGD | 0.0001 | | |
| 16 | Adam | 0.001 | | |
| 16 | Adam | 0.0001 | | |
| 32 | SGD | 0.001 | | |
| 32 | SGD | 0.0001 *(Opt)* | | |
| 32 | Adam | 0.001 *(Opt)* | | |
| 32 | Adam | 0.0001 | | |

### 🔍 Analysis (Q1a)
*[Replace this text with your detailed analysis. Discuss which optimizer worked best, the effect of batch size, and how ResNet-50 compared to ResNet-18.]*

---

## 📈 Q1(b): SVM Classifier Results
**Task:** Training SVM with varying kernels ('poly', 'rbf') and hyperparameters.

| Dataset | Kernel | Hyperparameters | Test Accuracy (%) | Train Time (ms) |
| :--- | :---: | :--- | :---: | :---: |
| **MNIST** | Poly | | | |
| **MNIST** | RBF | | | |
| **FashionMNIST** | Poly | | | |
| **FashionMNIST** | RBF | | | |

---

## ⚡ Q2: CPU vs GPU Performance Analysis
**Dataset:** FashionMNIST
**Comparison:** Training time and FLOPs count for ResNet models on different hardware.

| Compute | Batch | Optim | LR | R-18 Acc (%) | R-18 Time (ms) | R-18 FLOPs | R-50 Acc (%) | R-50 Time (ms) | R-50 FLOPs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU** | 16 | SGD | 0.001 | | | | | | |
| **CPU** | 16 | Adam | 0.001 | | | | | | |
| **GPU** | 16 | SGD | 0.001 | | | | | | |
| **GPU** | 16 | Adam | 0.001 | | | | | | |

### 💻 System Analysis (Q2)
*[Replace this text with your report on the hardware differences. Mention the speedup factor of GPU over CPU and how FLOPs correlate with training time.]*

---