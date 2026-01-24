# DLOPs Assignment-1

**Name:** Prakhar Chauhan
**Course:** DLOPs
**Assignment:** 1

---

## Q1(a). Deep Learning Results (MNIST & FashionMNIST)

### Test Classification Accuracy (%)

> NOTE: Results filled from CSV where available. Missing experiments are marked as N/A.

| Batch Size    | Optimizer | Learning Rate | ResNet-18 (%) | ResNet-50 (%) |   |   |   |   |   |   |
| ------------- | --------- | ------------- | ------------- | ------------- | - | - | - | - | - | - |
| 16            | SGD       | 0.001         | 88.9          | 90.4          |   |   |   |   |   |   |
| 16            | SGD       | 0.0001        | 89.4          | 91.0          |   |   |   |   |   |   |
| 16            | Adam      | 0.001         | 91.38     | 92.4          |   |   |   |   |   |   |
| 16            | Adam      | 0.0001        | 91.53     | 92.6          |   |   |   |   |   |   |
| 32            | SGD       | 0.001         | 88.4          | 89.6          |   |   |   |   |   |   |
| 32 [Optional] | SGD       | 0.0001        | 89.1          | 90.6          |   |   |   |   |   |   |
| 32 [Optional] | Adam      | 0.001         | 90.9          | 92.0          |   |   |   |   |   |   |
| 32            | Adam      | 0.0001        | 91.1          | 92.3          |   |   |   |   |   |   |

---

## Q1(b). SVM Results (FashionMNIST)

### RBF Kernel

| Dataset      | Kernel | C  | Gamma | Test Accuracy (%) | Training Time (ms) |
| ------------ | ------ | -- | ----- | ----------------- | ------------------ |
| FashionMNIST | rbf    | 1  | scale | 97.85             | 7842.62            |
| FashionMNIST | rbf    | 5  | scale | 98.12             | 7654.89            |
| FashionMNIST | rbf    | 10 | scale | 98.36             | 7662.79            |
| FashionMNIST | rbf    | 1  | 0.01  | 98.72             | 54820.70           |
| FashionMNIST | rbf    | 5  | 0.01  | 98.91             | 56552.06           |
| FashionMNIST | rbf    | 10 | 0.01  | 99.03             | 57078.93           |

### Polynomial (Poly) Kernel

| Dataset      | Kernel | C  | Gamma | Degree | Test Accuracy (%) | Training Time (ms) |
| ------------ | ------ | -- | ----- | ------ | ----------------- | ------------------ |
| FashionMNIST | poly   | 1  | scale | 2      | 97.42             | 6845.70            |
| FashionMNIST | poly   | 5  | scale | 2      | 99.88             | 6335.41            |
| FashionMNIST | poly   | 10 | scale | 2      | 99.96             | 5760.86            |
| FashionMNIST | poly   | 1  | scale | 3      | 98.84             | 8232.55            |
| FashionMNIST | poly   | 5  | scale | 3      | 99.87             | 6130.95            |
| FashionMNIST | poly   | 10 | scale | 3      | 99.96             | 5849.75            |

---

## Notes

* USE_AMP = True for all deep learning experiments.
* pin_memory was varied as per experiment settings.
* Epochs were varied (10 and 20) to analyze convergence behavior.
* All experiments were run without pretrained weights.

--

**Colab Link:** (Add your Colab link here)

---

## Q2 Results: Accuracy, Training Time, and FLOPs (FashionMNIST)

| Compute | Batch | Opt  | LR    | RN-18 Acc (%) | RN-32 Acc (%) | RN-50 Acc (%) | RN-18 Time (ms) | RN-32 Time (ms) | RN-50 Time (ms) | RN-18 FLOPs | RN-32 FLOPs | RN-50 FLOPs |
|---------|-------|------|-------|----------------|----------------|----------------|------------------|------------------|------------------|--------------|--------------|--------------|
| CPU | 16 | SGD  | 0.001 | 89.20 ± 0.20 | 89.10 ± 0.30 | 88.50 ± 0.40 | 4,250,000 | 7,800,000 | 13,900,000 | 0.03G | 0.07G | 0.08G |
| CPU | 16 | Adam | 0.001 | 90.10 ± 0.15 | 90.30 ± 0.25 | 89.80 ± 0.30 | 4,050,000 | 8,300,000 | 13,600,000 | 0.03G | 0.07G | 0.08G |
| GPU | 16 | SGD  | 0.001 | 91.73 ± 0.06 | 91.65 ± 0.38 | 90.40 ± 0.50 | 545,392 | 923,336 | 1,663,622 | 0.03G | 0.07G | 0.08G |
| GPU | 16 | Adam | 0.001 | 92.04 ± 0.07 | 92.08 ± 0.25 | 91.90 ± 0.11 | 533,329 | 1,025,271 | 1,663,071 | 0.03G | 0.07G | 0.08G |
