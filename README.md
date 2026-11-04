# 🧠 MNIST Handwritten Digit Classifier (CNN)

A Convolutional Neural Network (CNN) built with **TensorFlow / Keras** that classifies handwritten digits from the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The model achieves **~98.4% test accuracy** within 5 epochs.

---

## 📌 Project Overview

This repository demonstrates how to build, train, evaluate, and visualize predictions for an image classification task using Python and TensorFlow.

### Key Features
* **Automated Data Processing:** Normalizes pixel values ($0-255 \rightarrow 0-1$) and reshapes input tensors to include channel dimensions.
* **Efficient CNN Architecture:** Combines Convolutional, Max Pooling, and Dense layers for fast training and high accuracy.
* **Visualization:** Plots sample test predictions using `matplotlib`.

---

## 🏗️ Model Architecture

| Layer Type | Configuration | Output Shape | Param # |
| :--- | :--- | :--- | :--- |
| **Input** | Image Tensor | (28, 28, 1) | 0 |
| **Conv2D** | 32 filters, (3, 3) kernel, ReLU | (26, 26, 32) | 320 |
| **MaxPooling2D** | (2, 2) pool size | (13, 13, 32) | 0 |
| **Flatten** | Vectorization | (5408,) | 0 |
| **Dense** | 64 units, ReLU | (64,) | 346,176 |
| **Dense (Output)** | 10 units, Softmax | (10,) | 650 |

---

## 📊 Performance Metrics

* **Training Epochs:** 5
* **Batch Size:** 64
* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Test Accuracy:** `~98.39%`

---

## 🛠️ Prerequisites & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/your-repository-name.git](https://github.com/YOUR_GITHUB_USERNAME/your-repository-name.git)
cd your-repository-name
