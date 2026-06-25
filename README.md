# CNN vs FC Networks — Effectiveness & Efficiency on CIFAR-10



> Comparative study of the effectiveness and efficiency of convolutional (CNN)
> vs. fully connected / dense (FC) neural networks on multi-class image
> classification. Original code from my Physics thesis (5-year combined
> BSc+MSc program), 2018.

> ⚠️ **Archived / historical repository.** This is the original experimental
> code from my thesis, preserved for reference. It is not actively maintained
> and may require adaptation to run on modern library versions.

## Context

Experiments from my Physics thesis ("Learning and Analysis of Deep Artificial Neural Networks")
at National University of Cuyo (2018). Deep neural
networks achieve strong performance on AI tasks, but training them carries a
high computational cost, and they are designed in many shapes and sizes
depending on the application. This work studies that trade-off for two
architectures widely used in Computer Vision: dense networks (FC) and
convolutional networks (CNN). The full thesis (in Spanish), available [here](https://bdigital.uncu.edu.ar/13989) 
at the National University of Cuyo Digital Library.

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
- **Metrics:** For each network (trained with its best hyperparameter configuration), the
  following were measured: classification accuracy (effectiveness), training time
  (efficiency), number of hidden units, number of hidden layers, and number of
  parameters.

## Requirements

Written using the libraries of the time and trained on an **Nvidia GeForce
GTX 970** GPU:

- Python
- [Keras](https://keras.io/)
- [Scikit-learn](https://scikit-learn.org/)

> Current Keras/TensorFlow stacks are **not** compatible without changes. To
> reproduce the original results, use library versions from May 2018.

## Results

Across all 22 networks, CNNs were consistently more **effective** but less
**efficient** than dense networks. Convolutional models reached 68–81% test
accuracy (vs. 29–59% for dense networks), while dense networks trained
substantially faster — most in under 25 minutes, against well over an hour for
several CNNs. This reflects a clear effectiveness–efficiency trade-off. CNNs buy
higher accuracy at the cost of longer training, thanks to having many more
hidden units but far fewer parameters than dense networks.

See **[RESULTS.md](./RESULTS.md)** for the results — per-network accuracy,
training time, parameter counts, and all figures from the thesis.

## Status

Archived for historical and reference purposes. Not maintained.

## References

Ba, J. y Caruana, R. (2013). Do Deep Nets Really Need to be Deep?. Computing Research
Repository, abs/1312.6184. http://arxiv.org/abs/1312.6184

Lin, Z., Memisevic, R. y Konda, K.R. (2015). How far can we go without convolution:
Improving fully-connected networks. Computing Research Repository, abs/1511.02580.
http://arxiv.org/abs/1511.02580

Nielsen, M. A. (2015). Neural Networks and Deep Learning [online]. Retrieved from
http://neuralnetworksanddeeplearning.com

Srivastava, N., Hinton, G., Frizhevsky, A., Sutskever, I. y Salakhutdinov, R. (2014). 
Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine
Learning Research, 15 (1), 1929-1958.

Springenberg, J. T., Dosovitskiy, A., Brox, T. y Riedmiller, M. (2014). Striving for
Simplicity: The All Convolutional Net. Computing Research Repository, abs/1412.6806.
http://arxiv.org/abs/1412.6806

Zeiler, M. D. y Fergus, R. (2013). Stochastic Pooling for Regularization of Deep Convo-
lutional Neural Networks. Computing Research Repository, abs/1301.3557. https://arxiv.org/pdf/1301.3557.pdf


