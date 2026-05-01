# Experiment Code

Canonical experiment scripts.

## Scripts

- **experiment_cifar10.py** — CIFAR-10 cross-architecture and same-architecture experiments. 10 seeds. Writes `../results/cifar10_results_10seed.json`.
- **experiment_cifar100.py** — CIFAR-100 cross-architecture experiments. 10 seeds. Writes `../results/cifar100_results_10seed.json`.
- **at_imagenet_full.py** — ImageNet-V2 experiment with six models, three same-architecture baselines, per-model preprocessing, vectorized AT. Writes `../results/imagenet_v2_full_results.json`.

## Archive

See `archive/` for earlier iterations preserved for reproducibility.
