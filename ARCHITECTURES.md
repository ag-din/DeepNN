
# Network Architectures — Dense (FC) & Convolutional (CNN)

## Dense (FC) Architectures

Existing architectures from the literature plus original variants with similar
characteristics. Each dense network takes an input vector of **3,072 values**
(the pixel intensities of a typical CIFAR-10 image) and ends in a **10-unit
softmax** output layer, producing a probability distribution over the 10 object
classes — the standard choice for multi-class classification (Nielsen, 2015). 
See Table 1.

- **RD01.** Based on a model proposed by Ba & Caruana (2013). Two hidden layers
  of 2,000 sigmoid units each.
- **RD02.** Architecture suggested by Lin et al. (2015). Two hidden layers of
  4,000 ReLU units each.
- **RD03.** Another structure from Lin et al. (2015). Three hidden layers: 4,000
  ReLU, 1,000 linear, and 4,000 ReLU.
- **RD04.** Five hidden layers: 2,000 tanh, 500 linear, 1,000 tanh, 250 linear,
  and 500 tanh.
- **RD05.** Three hidden layers: 3,000 sigmoid, 1,000 linear, and 2,000 tanh.
- **RD06.** Five hidden layers: three layers of 1,000 ReLU units each,
  interleaved with 200-unit linear layers.
- **RD07.** Three hidden layers: 1,000 tanh, 2,000 ReLU, and 3,000 tanh.
- **RD08.** Two hidden layers of 5,000 and 2,000 PReLU units, respectively.
- **RD09.** Three hidden layers: 5,000 PReLU, 1,000 linear, and 2,000 PReLU.
- **RD10.** Four hidden layers of 2,000 units each: layers 1 and 3 use Leaky
  ReLU, layers 2 and 4 use PReLU.
- **RD11.** Two hidden layers: 3,000 PReLU and 2,000 Thresholded ReLU.

**Table 1: Dense architectures.** Abbreviation: k = 1,000.

| Network | Architecture |
|---------|--------------|
| RD01 (Ba & Caruana, 2013) | 2k sigmoid – 2k sigmoid – 10 softmax |
| RD02 (Lin et al., 2015) | 4k ReLU – 4k ReLU – 10 softmax |
| RD03 (Lin et al., 2015) | 4k ReLU – 1k linear – 4k ReLU – 10 softmax |
| RD04 | 2k tanh – 0.5k linear – 1k tanh – 0.25k linear – 0.5k tanh – 10 softmax |
| RD05 | 3k sigmoid – 1k linear – 2k tanh – 10 softmax |
| RD06 | 1k ReLU – 0.2k linear – 1k ReLU – 0.2k linear – 1k ReLU – 10 softmax |
| RD07 | 1k tanh – 2k ReLU – 3k tanh – 10 softmax |
| RD08 | 5k PReLU – 2k PReLU – 10 softmax |
| RD09 | 5k PReLU – 1k linear – 2k PReLU – 10 softmax |
| RD10 | 2k Leaky ReLU – 2k PReLU – 2k Leaky ReLU – 2k PReLU – 10 softmax |
| RD11 | 3k PReLU – 2k Thresholded ReLU – 10 softmax |

## Convolutional (CNN) Architectures

Existing architectures from the literature plus original variants with similar
characteristics. Each convolutional network takes an input tensor of
**32×32×3** and ends in a **10-unit softmax** output layer. In all CNNs, filters
are applied with a stride of 1 pixel along the width and height of the activation
volumes, and zero-padding is set to preserve the width and height after each
convolution. See Table 2.

- **RC01.** Based on a model by Hinton et al. (2012). Three convolutional layers,
  each followed by a pooling layer, plus a final locally-connected (non-conv)
  layer. Conv layers: 64 filters of 5×5 with ReLU. Pooling summarizes 3×3 regions
  with stride 2 — first layer max-pooling, second and third average-pooling
  (Hinton et al. use stochastic-pooling in the third). Locally-connected layer:
  16 filters of 3×3, stride 1, no zero-padding.
- **RC02.** Architecture by Zeiler & Fergus (2013). Three convolutional layers,
  each followed by pooling. First two conv layers: 64 filters of 5×5 with ReLU;
  third: 128 filters of the same kind. Pooling: average-pooling over 3×3 regions,
  stride 2.
- **RC03.** Model C from Springenberg et al. (2014). Seven convolutional layers
  and three pooling layers. Conv 1–2: 96 filters of 3×3, ReLU → max-pooling (3×3,
  stride 2). Conv 3–4: 192 filters of 3×3, ReLU → max-pooling (3×3, stride 2).
  Conv 5–7: 192, 192, and 10 filters of 3×3, 1×1, and 1×1 respectively, ReLU →
  global average-pooling over a 6×6 region.
- **RC04.** Two conv layers and two pooling layers. Conv 1: 96 filters of 5×5
  with ReLU; conv 2: 64 filters of the same kind. Both pooling layers:
  average-pooling over 3×3 regions, stride 2.
- **RC05.** Two conv layers, two pooling layers, and a locally-connected layer.
  Conv 1: 256 filters of 3×3 with ReLU; conv 2: 128 filters of the same kind.
  Pooling (3×3, stride 2): first max-pooling, second average-pooling.
  Locally-connected layer: 64 filters of 3×3, stride 1, no zero-padding.
- **RC06.** Two conv layers, two pooling layers, and two fully-connected layers.
  Conv 1: 96 filters of 5×5 with ReLU; conv 2: 192 filters of the same kind.
  Pooling (3×3, stride 2): first average-pooling, second max-pooling. Then two
  fully-connected layers of 2,000 PReLU units each.
- **RC07.** Two conv layers, two pooling layers, and one fully-connected layer.
  Conv 1: 64 filters of 5×5 with ReLU; conv 2: 128 filters of the same kind.
  Pooling: max-pooling over 3×3 regions, stride 2. Then a fully-connected layer
  of 1,000 ReLU units.
- **RC08.** Five conv layers and five pooling layers. Each conv layer: 96 filters
  of 5×5 with ReLU. Each pooling: average-pooling over 2×2 regions, stride 2.
- **RC09.** Six conv layers and three pooling layers. Each conv layer: 64 filters
  of 3×3 with ReLU. A max-pooling layer (2×2, stride 2) follows every two conv
  layers.
- **RC10.** Two conv layers, two pooling layers, and one fully-connected layer.
  Conv 1: 32 filters of 5×5 with ReLU; conv 2: 32 filters of 3×3 of the same
  kind. Pooling: average-pooling, stride 2, over 3×3 and 2×2 regions
  respectively. Then a fully-connected layer of 1,000 ReLU units.
- **RC11.** Two conv layers and two pooling layers. Conv 1: 64 filters of 5×5
  with ReLU; conv 2: 64 filters of 3×3 of the same kind. Pooling:
  average-pooling, stride 2, over 3×3 and 2×2 regions respectively.

**Table 2: Convolutional architectures.** Abbreviations: c = convolutional;
k = 1,000; p = pooling; lc = locally-connected.

| Model | Architecture |
|-------|--------------|
| RC01 | 64c – p – 64c – p – 64c – p – 16lc – 10 softmax |
| RC02 (Zeiler & Fergus, 2013) | 64c – p – 64c – p – 128c – p – 10 softmax |
| RC03 (Springenberg et al., 2014) | 96c – 96c – p – 192c – 192c – p – 192c – 192c – 10c – p – 10 softmax |
| RC04 | 96c – p – 64c – p – 10 softmax |
| RC05 | 256c – p – 128c – p – 64lc – 10 softmax |
| RC06 | 96c – p – 192c – p – 2k PReLU – 2k PReLU – 10 softmax |
| RC07 | 64c – p – 128c – p – 1k ReLU – 10 softmax |
| RC08 | 96c – p – 96c – p – 96c – p – 96c – p – 96c – p – 10 softmax |
| RC09 | 64c – 64c – p – 64c – 64c – p – 64c – 64c – p – 10 softmax |
| RC10 | 32c – p – 32c – p – 1k ReLU – 10 softmax |
| RC11 | 64c – p – 64c – p – 10 softmax |
