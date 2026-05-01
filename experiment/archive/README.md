# Archive

Files here are valid earlier-iteration scripts preserved for reproducibility. Superseded by canonical scripts in `../`.

## Files

- **at_imagenet_lean.py** — 3-seed lean ImageNet-V2 replication script. Uses standard ImageNet preprocessing for all three models, vectorized AT, and CPU histograms. Output: `../../results/archive/imagenet_v2_results.json`. Confirms cross-pair AT values from the canonical run; superseded by `../at_imagenet_full.py` which adds same-architecture baselines and per-model preprocessing.
- **imagenet_pretrained.py** — Oldest ImageNet script (April 2026). Uses an O(N²) Python double-loop for AT cost-matrix construction (functionally correct but slow). No surviving output JSON. Superseded by `../at_imagenet_full.py`.
