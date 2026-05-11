# Activation Transport

A channel-level similarity metric for neural network representations. For two trained networks, we z-score per-channel activations, compute a 1-Wasserstein distance between every pair of channel histograms, and solve the resulting linear-sum assignment. The mean matched cost is the AT score (lower = more similar).

## Findings

- Cross-architecture AT is $7\times$ same-architecture AT on CIFAR-10 (10 seeds) and $26\times$ on CIFAR-100 (10 seeds).
- On ImageNet-V2 with `timm` checkpoints: cross/same AT ratio is $60.5\times$ against the ViT-vs-DeiT-B baseline ($t = 6597$, $p = 1.05 \times 10^{-13}$). Across three same-architecture baselines and four depth fractions, ResNet-vs-ViT exceeds same-architecture in 12/12 comparisons.
- CKA on the same comparisons reports only $1.4$--$1.5\times$ differences.
- Layer-wise AT increases monotonically with depth on all three datasets.
- ViT and MLP-Mixer cluster together and apart from CNNs; ViT-vs-Mixer AT is $5.9\times$ smaller than ResNet-vs-ViT AT on ImageNet.

## Datasets and seeds

| Dataset | Models | Seeds |
|---|---|---|
| CIFAR-10 | ResNet-18, ViT-Small, MLP-Mixer (trained from scratch) | 10 (42--51) |
| CIFAR-100 | ResNet-18, ViT-Small, MLP-Mixer (trained from scratch) | 10 (42--51) |
| ImageNet-V2 | ResNet-50, ViT-B/16, MLP-Mixer-B/16 (timm checkpoints) | 3 (42--44), 5K samples per seed |

## Layout

```
paper/       LaTeX source + refs.bib
results/     JSON results for CIFAR-10, CIFAR-100, ImageNet-V2
experiment/  Python experiment code for all three datasets
```
