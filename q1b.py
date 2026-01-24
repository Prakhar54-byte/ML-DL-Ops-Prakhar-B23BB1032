import time
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from sklearn import svm
from sklearn.metrics import accuracy_score

def load_svm_data(name, max_train=10000, max_test=2000):
    if name == "MNIST":
        train_data = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
        test_data  = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    else:
        train_data = datasets.FashionMNIST("./data", train=True, download=True, transform=transforms.ToTensor())
        test_data  = datasets.FashionMNIST("./data", train=False, download=True, transform=transforms.ToTensor())

    X_train = train_data.data[:max_train].reshape(max_train, -1).numpy() / 255.0
    y_train = train_data.targets[:max_train].numpy()

    X_test = test_data.data[:max_test].reshape(max_test, -1).numpy() / 255.0
    y_test = test_data.targets[:max_test].numpy()

    return X_train, y_train, X_test, y_test


def run_svm_exp(dataset_name, kernel, C=1.0, gamma='scale', degree=3):
    X_train, y_train, X_test, y_test = load_svm_data(dataset_name)

    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    train_time_ms = (end - start) * 1000

    return {
        "Dataset": dataset_name,
        "Kernel": kernel,
        "C": C,
        "Gamma": gamma,
        "Degree": degree,
        "Test Accuracy (%)": round(acc, 2),
        "Training Time (ms)": round(train_time_ms, 2)
    }


results = []

# ---------- MNIST ----------
# RBF
results.append(run_svm_exp("MNIST", "rbf", C=1.0, gamma="scale"))
results.append(run_svm_exp("MNIST", "rbf", C=5.0, gamma="scale"))
results.append(run_svm_exp("MNIST", "rbf", C=10.0, gamma="scale"))
results.append(run_svm_exp("MNIST", "rbf", C=1.0, gamma=0.01))
results.append(run_svm_exp("MNIST", "rbf", C=5.0, gamma=0.01))
results.append(run_svm_exp("MNIST", "rbf", C=10.0, gamma=0.01))

# POLY
results.append(run_svm_exp("MNIST", "poly", C=1.0, degree=2))
results.append(run_svm_exp("MNIST", "poly", C=5.0, degree=2))
results.append(run_svm_exp("MNIST", "poly", C=10.0, degree=2))
results.append(run_svm_exp("MNIST", "poly", C=1.0, degree=3))
results.append(run_svm_exp("MNIST", "poly", C=5.0, degree=3))
results.append(run_svm_exp("MNIST", "poly", C=10.0, degree=3))


# ---------- FashionMNIST ----------
# RBF
results.append(run_svm_exp("FashionMNIST", "rbf", C=1.0, gamma="scale"))
results.append(run_svm_exp("FashionMNIST", "rbf", C=5.0, gamma="scale"))
results.append(run_svm_exp("FashionMNIST", "rbf", C=10.0, gamma="scale"))
results.append(run_svm_exp("FashionMNIST", "rbf", C=1.0, gamma=0.01))
results.append(run_svm_exp("FashionMNIST", "rbf", C=5.0, gamma=0.01))
results.append(run_svm_exp("FashionMNIST", "rbf", C=10.0, gamma=0.01))

# POLY
results.append(run_svm_exp("FashionMNIST", "poly", C=1.0, degree=2))
results.append(run_svm_exp("FashionMNIST", "poly", C=5.0, degree=2))
results.append(run_svm_exp("FashionMNIST", "poly", C=10.0, degree=2))
results.append(run_svm_exp("FashionMNIST", "poly", C=1.0, degree=3))
results.append(run_svm_exp("FashionMNIST", "poly", C=5.0, degree=3))
results.append(run_svm_exp("FashionMNIST", "poly", C=10.0, degree=3))


df = pd.DataFrame(results)
df
