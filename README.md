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

22 networks in total — 11 dense (RD01–RD11) and 11 convolutional (RC01–RC11).
See [ARCHITECTURES.md](./ARCHITECTURES.md) for the full per-layer specification.

## Methodology

- **Preprocessing:** Global Contrast Normalization (zero mean, unit std; pixel
  values in [-1, 1]).
- **Weight init:** Glorot/Xavier Normal (He Normal for rectified units); biases
  initialized to zero.
- **Loss:** cross-entropy (negative log-likelihood).
- **Optimizer:** Adam (Keras defaults: lr=0.001, β1=0.9, β2=0.999, ε=1e-8),
  chosen partly because its adaptive learning rate removes the need to tune it.
- **Hyperparameter search:** randomized search + 10-fold cross-validation
  (Scikit-learn `RandomizedSearchCV`). Only **two** hyperparameters were tuned —
  batch size and number of epochs — explored on a logarithmic scale over the
  ranges {30, 100, 500, 1,000, 3,000, 5,000, 8,000} and {10, 30, 100, 300, 800,
  1,000, 1,500} respectively. The learning rate did not need tuning because Adam
  adapts it automatically. For each hyperparameter configuration and a given network
   architecture, **k-fold cross-validation (CV)** was applied. The training set (50,000 images)
  was partitioned into *k* equal-sized subsets: one subset served as validation
  data and the remaining *k*−1 as training data. The network was then trained and
  evaluated, repeating this process for *k* iterations so that each subset served
  as the validation set once. Thus, a network with a given hyperparameter
  configuration was trained and evaluated *k* times, yielding *k* performance
  measures that were averaged arithmetically. The number of folds was set to
  **k = 10**, an optimal value according to Refaeilzadeh et al. (2008). Each
  hyperparameter configuration was sampled at random from the hyperparameter space
  via random search.
- **Regularization:** Dropout (p=0.5) and Batch Normalization. According to Srivastava et
  al. (2014), the value of 0.5 for dropout appears to be close to optimal for a wide range of
  networks and tasks.

## Requirements

Written using the libraries of the time and trained on an **Nvidia GeForce
GTX 970** GPU:

- Python
- [Keras](https://keras.io/)
- [Scikit-learn](https://scikit-learn.org/)

> Current Keras/TensorFlow stacks are **not** compatible without changes. To
> reproduce the original results, use library versions from May 2018.

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

## References

Ba, J. y Caruana, R. (2013). Do Deep Nets Really Need to be Deep?. Computing Research
Repository, abs/1312.6184. Recuperado de http://arxiv.org/abs/1312.6184

Lin, Z., Memisevic, R. y Konda, K.R. (2015). How far can we go without convolution:
Improving fully-connected networks. Computing Research Repository, abs/1511.02580.
Recuperado de http://arxiv.org/abs/1511.02580

Srivastava, N., Hinton, G., Frizhevsky, A., Sutskever, I. y Salakhutdinov, R. (2014). 
Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine
Learning Research, 15 (1), 1929-1958.


