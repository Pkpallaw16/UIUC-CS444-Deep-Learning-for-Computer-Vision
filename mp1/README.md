
# Assignment 1: Linear Classifiers

In this assignment, we implement and evaluate various linear classifiers — Perceptron, Support Vector Machine (SVM), Softmax, and Logistic Regression — on the Rice and Fashion-MNIST datasets.

We experimented with different hyperparameters and learning rate decay strategies to find optimal configurations for each model on both datasets.

---

## 📈 Models

We implemented the following classifiers (in their respective files):

- **Perceptron** — `perceptron.py`
- **Support Vector Machine (SVM)** — `svm.py`
- **Softmax** — `softmax.py`
- **Logistic Regression** — `logistic.py`

For the logistic regression classifier, multi-class prediction was not attempted as it requires one-vs-rest or one-vs-one strategies, so it was applied only to the Rice dataset.

---

## 📂 Data Setup

To download and set up the Fashion-MNIST dataset, run:

```bash
cd assignment1/fashion-mnist/
bash get_data.sh
```

---

## 📊 Results

### Rice Dataset

| Benchmark                | Test Accuracy |
|--------------------------|---------------|
| Logistic Regression      | 98.73% |
| Perceptron                | 99.78% |
| SVM                       | 80.97% |
| Softmax                   | 79.59% |

### Fashion-MNIST Dataset

| Benchmark                | Test Accuracy |
|--------------------------|---------------|
| Logistic Regression      | Not Applicable |
| Perceptron                | 81.6% |
| SVM                       | 82.52% |
| Softmax                   | 83.32% |

---

## 📋 Hyperparameter Highlights

### Perceptron

| Dataset         | Learning Rate | Epochs | Batch Size | LR Decay | Test Accuracy |
|-----------------|----------------|--------|------------|---------|---------------|
| Rice            | 0.5            | 10     | 1000       | 10% every 5 epochs | 99.78% |
| Fashion-MNIST   | 0.05           | 20     | 1000       | 10% every 5 epochs | 81.6% |

---

### SVM

| Dataset         | Learning Rate | Epochs | Batch Size | Reg. Const. | LR Decay | Test Accuracy |
|-----------------|----------------|--------|------------|-------------|---------|---------------|
| Rice            | 0.1            | 30     | 256        | 0.05        | 10% every 5 epochs | 80.97% |
| Fashion-MNIST   | 0.8            | 6      | 256        | 0.4         | 10% every 5 epochs | 82.52% |

---

### Softmax

| Dataset         | Learning Rate | Epochs | Batch Size | Reg. Const. | LR Decay | Test Accuracy |
|-----------------|----------------|--------|------------|-------------|---------|---------------|
| Rice            | 0.5            | 20     | 256        | 0.05        | 1% every 5 epochs | 79.59% |
| Fashion-MNIST   | 0.4            | 7      | 256        | 0.5         | 1% every 5 epochs | 83.32% |

---

### Logistic Regression

| Dataset         | Learning Rate | Epochs | Threshold | Test Accuracy |
|-----------------|----------------|--------|-----------|---------------|
| Rice            | 0.2            | 10     | 0.5       | 98.73% |

---

## 👥 Team

- **Name(s):** Pallaw Kumar, Neha Jain
- **NetID(s):** pallawk2, nehaj4
- **Team Name (Kaggle):** Ninja4

---
