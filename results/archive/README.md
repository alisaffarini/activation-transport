# Archive

Earlier experimental data preserved for reproducibility. Superseded by canonical files in `../`.

## Files

- **cifar10_results.json** — 5-seed CIFAR-10 (seeds 42-46). True strict subset of `../cifar10_results_10seed.json` (seeds 42-51). Numbers match within seed noise. Preserved as a reproducibility checkpoint.
- **cifar100_results.json** — Earlier CIFAR-100 summary (no per-seed data). Less detailed than `../cifar100_results_10seed.json`; some t-statistics differ slightly due to recomputation.
- **imagenet_v2_results.json** — Output of `../../experiment/archive/at_imagenet_lean.py`. 3-seed ImageNet-V2 with cross-architecture AT/CKA only (no same-arch baselines). Confirms the `2.96 / 2.61 / 0.50` cross-pair values via an independent leaner pipeline; values agree with canonical full run within seed noise (~0.005-0.012 absolute).
