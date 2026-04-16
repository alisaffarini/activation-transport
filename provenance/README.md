# Provenance — Activation Transport

Raw run data from the experiment pipeline.

## Run History

| Run | Description | Key Output |
|-----|-------------|------------|
| `run_047_interp` | **Original MNIST experiment** — first AT metric implementation | Conversation log, iteration code, training outputs |
| `run_082_at_cifar10` | CIFAR-10 replication (5 seeds) | experiment.py |
| `run_083_at_cifar100` | CIFAR-100 extension (10 seeds) | experiment.py |
| `run_084_at_dynamics` | Training dynamics analysis | experiment.py |
| `run_086_at_normalization` | Normalization effects on AT | experiment.py |
| `original_paper/` | Files from burn-tokens paper directory | Original MNIST experiment, paper draft, next steps |

## Notes

- Run 047 was the original MNIST proof-of-concept
- Runs 082-083 generated the CIFAR-10/100 results reported in the paper
- ViT architecture differs between CIFAR-10 (dim=192, 6L, 6H) and CIFAR-100 (dim=256, 8L, 8H) — this is documented in the paper
- Runs 082-086 only have experiment.py files (output was captured in the results JSONs in the main repo)
