# Results

Full results for the 22 networks (11 dense, 11 convolutional) trained on
CIFAR-10 and evaluated on the 10,000-image test set.

> **Note:** The figures referenced below are reproduced from the original
> thesis; their axis labels and captions are in Spanish.

As a reference point, average **human performance** on CIFAR-10 is **94%**
accuracy. None of the 22 networks reached this level — the best was at least
13.3% less effective than a human.

---

## Effectiveness (classification accuracy)

CNNs were clearly more effective than dense networks. The convolutional group
sits in the ~68–81% range, while dense networks fall in the ~29–59% range.
**10 of the 11 CNNs** were at least **9.11% more accurate** than the dense
networks, indicating a stronger ability to capture abstract patterns and
generalize.

![Classification accuracy per network, with human performance as reference](figures/fig_5.1_accuracy_ranking.png)
*Figure 5.1 — Classification accuracy (%) of dense (RD) and convolutional (RC)
networks. Average human performance (94%) is shown for reference.*

![Box plot of accuracy by network type](figures/fig_5.2_accuracy_boxplot.png)
*Figure 5.2 — Distribution of classification accuracy for dense (RD) vs.
convolutional (RC) networks.*

The single exception is **RC03**, which failed to train under this setup
(10.00%, i.e. chance level), despite reaching ~90% in the original paper
(Springenberg et al., 2014).

| Rank | Network | Accuracy (%) | Type |
|------|---------|--------------|------|
| — | Humans | 94.00 | reference |
| 1 | RC02 | 80.70 | CNN |
| 2 | RC09 | 79.93 | CNN |
| 3 | RC01 | 79.81 | CNN |
| 4 | RC08 | 79.54 | CNN |
| 5 | RC07 | 78.94 | CNN |
| 6 | RC06 | 77.24 | CNN |
| 7 | RC04 | 75.13 | CNN |
| 8 | RC10 | 73.53 | CNN |
| 9 | RC11 | 70.78 | CNN |
| 10 | RC05 | 67.95 | CNN |
| 11 | RD08 | 58.84 | FC |
| 12 | RD11 | 58.60 | FC |
| 13 | RD10 | 56.92 | FC |
| 14 | RD01 | 56.44 | FC |
| 15 | RD02 | 55.63 | FC |
| 16 | RD06 | 52.45 | FC |
| 17 | RD05 | 48.52 | FC |
| 18 | RD07 | 48.15 | FC |
| 19 | RD03 | 47.16 | FC |
| 20 | RD04 | 45.04 | FC |
| 21 | RD09 | 29.25 | FC |
| 22 | RC03 | 10.00 | CNN (failed to train) |

- Most effective overall: **RC02** (80.70%).
- Most effective dense network: **RD08** (58.84%).

---

## Efficiency (training time)

Dense networks were considerably faster to train. Most finished in under 25
minutes, while several CNNs took well over an hour. **9 of the 11 dense
networks** were at least **6.4 min more efficient** than the CNNs.

![Training time per network](figures/fig_5.5_training_time.png)
*Figure 5.5 — Training time (minutes) for dense (RD) and convolutional (RC)
networks.*

![Box plot of training time by network type](figures/fig_5.6_training_time_boxplot.png)
*Figure 5.6 — Distribution of training time for dense (RD) vs. convolutional
(RC) networks.*

| Network | Training time (min) | Type |
|---------|---------------------|------|
| RD04 | 1.6 | FC |
| RD09 | 6.5 | FC |
| RD02 | 7.1 | FC |
| RD07 | 7.3 | FC |
| RD05 | 7.9 | FC |
| RD06 | 9.7 | FC |
| RD01 | 16.9 | FC |
| RD10 | 19.9 | FC |
| RD11 | 24.6 | FC |
| RC05 | 31.0 | CNN |
| RC03 | 46.0 | CNN |
| RC01 | 48.0 | CNN |
| RC06 | 50.0 | CNN |
| RC02 | 52.0 | CNN |
| RC10 | 55.0 | CNN |
| RC04 | 57.0 | CNN |
| RD08 | 68.5 | FC |
| RD03 | 83.2 | FC |
| RC07 | 89.0 | CNN |
| RC11 | 185.0 | CNN |
| RC09 | 217.0 | CNN |
| RC08 | 240.0 | CNN |

- Most efficient overall: **RD04** (1.6 min).
- Most efficient CNN: **RC05** (31 min).

---

## Capacity: parameters and hidden units

A key structural difference explains the trade-off: CNNs have **far more hidden
units** but **far fewer parameters** than dense networks, thanks to local
connectivity and parameter sharing.

- **All** CNNs had more hidden units than the dense networks.
- **82%** of the CNNs had fewer parameters to learn than the dense networks.

### Number of parameters (millions)

![Number of parameters per network](figures/fig_5.7_parameters_bar.png)
*Figure 5.7 — Number of parameters (millions) for dense (RD) and convolutional
(RC) networks.*

![Box plot of parameters by network type](figures/fig_5.8_parameters_boxplot.png)
*Figure 5.8 — Distribution of the number of parameters (millions) for dense (RD)
vs. convolutional (RC) networks.*

| Network | Parameters (M) | Type |
|---------|----------------|------|
| RC11 | 0.1 | CNN |
| RC09 | 0.2 | CNN |
| RC04 | 0.2 | CNN |
| RC01 | 0.2 | CNN |
| RC02 | 0.3 | CNN |
| RC08 | 0.9 | CNN |
| RC03 | 1.0 | CNN |
| RC10 | 2.1 | CNN |
| RC05 | 3.0 | CNN |
| RD06 | 3.9 | FC |
| RD04 | 8.0 | FC |
| RC07 | 8.4 | CNN |
| RD01 | 10.2 | FC |
| RD07 | 11.1 | FC |
| RD05 | 14.2 | FC |
| RD11 | 15.2 | FC |
| RD10 | 18.2 | FC |
| RD03 | 20.3 | FC |
| RD09 | 22.4 | FC |
| RD08 | 25.4 | FC |
| RD02 | 28.3 | FC |
| RC06 | 29.1 | CNN |

### Number of hidden units (thousands)

![Number of hidden units per network](figures/fig_5.9_hidden_units_bar.png)
*Figure 5.9 — Number of hidden units (thousands) for dense (RD) and
convolutional (RC) networks.*

![Box plot of hidden units by network type](figures/fig_5.10_hidden_units_boxplot.png)
*Figure 5.10 — Distribution of the number of hidden units (thousands) for dense
(RD) vs. convolutional (RC) networks.*

![Box plot of hidden-units-to-parameters ratio](figures/fig_5.11_units_params_ratio.png)
*Figure 5.11 — Ratio of hidden units to parameters for dense (RD) vs.
convolutional (RC) networks. CNNs pack far more units per parameter.*

| Network | Hidden units (k) | Type |
|---------|------------------|------|
| RD06 | 3.4 | FC |
| RD01 | 4.0 | FC |
| RD04 | 4.25 | FC |
| RD11 | 5.0 | FC |
| RD05 | 6.0 | FC |
| RD07 | 6.0 | FC |
| RD08 | 7.0 | FC |
| RD02 | 8.0 | FC |
| RD09 | 8.0 | FC |
| RD10 | 8.0 | FC |
| RD03 | 9.0 | FC |
| RC10 | 41.96 | CNN |
| RC11 | 81.92 | CNN |
| RC01 | 86.08 | CNN |
| RC02 | 90.11 | CNN |
| RC07 | 99.3 | CNN |
| RC04 | 114.69 | CNN |
| RC08 | 130.94 | CNN |
| RC06 | 151.46 | CNN |
| RC09 | 172.03 | CNN |
| RC05 | 297.22 | CNN |
| RC03 | 320.13 | CNN |

---

## Reproduction of published architectures

Five of the 22 networks were based on previously published architectures. Their
accuracies compared to the reported values as follows:

| Network | Source | This study (%) | Difference |
|---------|--------|----------------|------------|
| RC02 | Zeiler & Fergus (2013) | 80.70 | −0.06% |
| RC03 | Springenberg et al. (2014) | 10.00 | −80.26% |
| RD01 | Ba & Caruana (2013) | 56.44 | +14.24% |
| RD02 | Lin et al. (2015) | 55.63 | +1.71% |
| RD03 | Lin et al. (2015) | 47.16 | −9.68% |

RD01 and RD02 improved on their published results, and RC02 closely matched.
RC03 and RD03 underperformed — the training setup was not optimal for those two
networks.

![Accuracy in this study vs. the literature](figures/fig_5.3_literature_comparison.png)
*Figure 5.3 — Comparison between the accuracies obtained in this study and those
reported in the literature, for the five architectures taken from other works.*

---

## Correlations between performance and structure

Within each model type the correlations were unclear: networks with similar
parameter or unit counts often reached different accuracies, and vice versa.
Across the full set of networks, however, some general trends emerged:

- More parameters → lower effectiveness (slight trend).
- More hidden units → higher effectiveness (slight trend).
- More parameters → lower efficiency (longer training).
- More hidden units → lower efficiency (longer training).

![Accuracy vs. training time](figures/fig_5.12_accuracy_vs_time.png)
*Figure 5.12 — Correlation between classification accuracy (%) and training time
(min) for dense (RD) and convolutional (RC) networks.*

![Accuracy vs. number of parameters](figures/fig_5.13_accuracy_vs_params.png)
*Figure 5.13 — Correlation between accuracy (%) and number of parameters
(millions). The x-axis is logarithmic.*

![Accuracy vs. number of hidden units](figures/fig_5.14_accuracy_vs_units.png)
*Figure 5.14 — Correlation between accuracy (%) and number of hidden units
(thousands). The x-axis is logarithmic.*

![Training time vs. number of parameters](figures/fig_5.15_time_vs_params.png)
*Figure 5.15 — Correlation between training time (min) and number of parameters
(millions). The x-axis is logarithmic.*

![Training time vs. number of hidden units](figures/fig_5.16_time_vs_units.png)
*Figure 5.16 — Correlation between training time (min) and number of hidden
units (millions). The x-axis is logarithmic.*

---

## Learned convolutional filters

The patterns learned by the convolutional layers became progressively less
interpretable from one layer to the next. Subjectively, the first-layer filters
appeared to capture edges, contours, and contrasts at different orientations.

---

## Summary

- CNNs were consistently **more effective** but **less efficient** than dense
  networks — a clear effectiveness–efficiency trade-off.
- 10 of 11 CNNs were at least 9.11% more accurate; 9 of 11 dense networks
  trained at least 6.4 min faster.
- The trade-off is driven by structure: CNNs have far more hidden units yet far
  fewer parameters, thanks to local connectivity and parameter sharing.
- Networks reproduced from the literature matched their reported accuracies
  closely, except RC03, which failed to train in this setup.
