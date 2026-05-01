# Activation Transport Reveals Fundamental Representation Divergence Between Convolutional and Attention-Based Architectures That CKA Fails to Detect

**Author:** Anonymous Author, Anonymous Institution

## Key Results

- **Cross-architecture AT is 7-26x same-architecture AT** on CIFAR-10 (10 seeds) and CIFAR-100 (10 seeds)
- **Validated at scale on ImageNet-V2:** cross/same AT ratio is **60.5x** (ResNet-50 vs ViT-B/16 = 2.96 vs ViT-vs-DeiT-B = 0.049; t = 6597, p = 1.05e-13) using publicly available `timm` checkpoints
- CKA reports only 1.4-1.5x difference for the same comparisons across all three datasets — it misses the divergence
- Divergence **increases monotonically with depth** at both CIFAR and ImageNet scale
- The effect **amplifies on harder/larger tasks**: cross/same ratio goes from 7x (CIFAR-10) to 26x (CIFAR-100) to 60x (ImageNet-V2)
- ViT and MLP-Mixer form a shared "attention family" distinct from CNNs (ViT-vs-Mixer AT is 5.9x smaller than ResNet-vs-ViT AT on ImageNet, replicating CIFAR finding)

## Datasets and seeds

| Dataset | Models | Seeds |
|---|---|---|
| CIFAR-10 | ResNet-18, ViT-Small, MLP-Mixer (trained from scratch) | 10 (42-51) |
| CIFAR-100 | ResNet-18, ViT-Small, MLP-Mixer (trained from scratch) | 10 (42-51) |
| ImageNet-V2 | ResNet-50, ViT-B/16, MLP-Mixer-B/16 (timm checkpoints) | 3 (42-44), 5K samples per seed |

## Structure

```
paper/       LaTeX source + refs.bib (NeurIPS-style, anonymized for blind submission)
results/     JSON results from CIFAR-10 (10 seeds), CIFAR-100 (10 seeds), and ImageNet-V2 (3 seeds)
experiment/  Python experiment code for all three datasets
```

## Anonymized code

Code and data: https://anonymous.4open.science/r/activation-transport
