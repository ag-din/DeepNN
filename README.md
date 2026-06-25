# CNN vs Fully Connected Networks — Effectiveness & Efficiency on CIFAR-10



> Comparative study of the effectiveness and efficiency of convolutional (CNN)
> vs. fully connected / dense (FC) neural networks on multi-class image
> classification. Original code from my Physics thesis (5-year integrated
> BSc+MSc program), 2018.

> ⚠️ **Archived / historical repository.** This is the original experimental
> code from my thesis, preserved for reference. It is not actively maintained
> and may require adaptation to run on modern library versions.

## Context

Experiments from my Physics thesis at National University of Cuyo (2018). Deep neural
networks achieve strong performance on AI tasks, but training them carries a
high computational cost, and they are designed in many shapes and sizes
depending on the application. This work studies that trade-off for two
architectures widely used in Computer Vision: dense networks (FC) and
convolutional networks (CNN).

## Objective

Measure how **effective** and **efficient** different FC configurations are
compared to CNNs on a multi-class image classification task. The work involved:

- Training FC and CNN models to classify images.
- Evaluating each network by **classification accuracy** (effectiveness) and
  **training time** (efficiency).
- Comparing both quantities across the two model types.

## Dataset

**[CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html)** — 60,000 low-resolution color images (32×32 RGB) across 10 classes
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), split
into 50,000 training and 10,000 test images.

## Architectures

**Dense (FC) — 11 networks (RD01–RD11):** input vector of 3,072 values, 2–5
hidden layers with varying widths (up to 5k units) and activations (sigmoid,
ReLU, tanh, linear, PReLU, Leaky ReLU, Thresholded ReLU), softmax output over
10 classes. Several based on Ba & Caruana (2013) and Lin et al. (2015).

**Convolutional (CNN) — 11 networks (RC01–RC11):** input tensor 32×32×3,
convolutional + pooling stacks (max / average / global average pooling), some
with locally-connected or fully-connected layers, softmax output over 10
classes. Several based on Zeiler & Fergus (2013) and Springenberg et al. (2014).

## Methodology

- **Preprocessing:** Global Contrast Normalization (zero mean, unit std; pixel
  values in [-1, 1]).
- **Weight init:** Glorot/Xavier Normal (He Normal for rectified units); biases
  initialized to zero.
- **Loss:** cross-entropy (negative log-likelihood).
- **Optimizer:** Adam (Keras defaults: lr=0.001, β1=0.9, β2=0.999, ε=1e-8),
  chosen partly because its adaptive learning rate removes the need to tune it.
- **Hyperparameter search:** randomized search + 10-fold cross-validation
  (Scikit-learn `RandomizedSearchCV`) over batch size and number of epochs,
  explored on a logarithmic scale.
- **Regularization:** Dropout (p=0.5) and Batch Normalization.

## Requirements

Written using the libraries of the time and trained on an **Nvidia GeForce
GTX 970** GPU:

- Python
- [Keras](https://keras.io/)
- [Scikit-learn](https://scikit-learn.org/)

> Current Keras/TensorFlow stacks are **not** compatible without changes. To
> reproduce the original results, use library versions from May 2018.

## How to run

```bash
[command to launch the experiments]
```

## Results

Results were partially consistent with the proposed hypotheses:

- **Effectiveness:** 91% of the trained CNNs were at least **9.11% more
  effective** than the FC networks — a stronger ability to learn complex
  patterns, attributed to CNNs having more hidden units, local connectivity, and
  parameter sharing within convolutional layers.
- **Efficiency:** 82% of the trained FC networks were at least **6.4 min more
  efficient** (faster to train) than the CNNs — attributed mainly to the number
  and complexity of operations performed, and to a lesser extent to the number
  of parameters learned.

| Architecture | Effectiveness (accuracy) | Efficiency (training time) |
|--------------|--------------------------|----------------------------|
| CNN          | Higher (≥ +9.11% in 91% of cases) | Slower |
| FC (dense)   | Lower                    | Faster (≥ 6.4 min in 82% of cases) |

These results help characterize how structural variations in deep networks
affect their performance — useful for balancing effectiveness and efficiency
when designing a network for a given task.


## Status

Archived for historical and reference purposes. Not maintained.
