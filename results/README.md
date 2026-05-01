# Results

Canonical experimental data cited by the paper.

## Files

- **cifar10_results_10seed.json** — CIFAR-10 cross-architecture and same-architecture AT/CKA. 10 seeds (42-51). Source for Table 1, layer-wise Table 3, abstract numbers.
- **cifar100_results_10seed.json** — CIFAR-100 cross-architecture and same-architecture AT/CKA. 10 seeds. Source for the cross-dataset comparison and the 26× cross/same ratio.
- **imagenet_v2_full_results.json** — ImageNet-V2 experiment: 3 seeds, six models (3 architectures × 2 variants each) with per-model preprocessing. Cross-architecture AT, three same-architecture baselines (ResNet-50 vs ResNet-50-tv, ViT-B/16 vs DeiT-B, Mixer-B/16 vs Mixer-MIIL), 12-cell robustness grid. Source for the ImageNet section.
- **imagenet_analysis_summary.md** — Paper-ready writeup of ImageNet findings with full statistical analysis. Reference document.

## Archive

See `archive/` for earlier iterations and pipeline-replication runs.
