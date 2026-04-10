# Activation Transport Reveals Fundamental Representation Divergence Between Convolutional and Attention-Based Architectures That CKA Fails to Detect

**Author:** Ali Saffarini, Harvard University

## Key Results

- **Cross-architecture AT is 7-26x same-architecture AT** across CIFAR-10 (5 seeds) and CIFAR-100 (10 seeds)
- CKA reports only 1.4x difference for the same comparisons — it misses the divergence
- Divergence **increases monotonically with depth** (AT grows 6.6x from early to late layers)
- The effect **amplifies on harder tasks**: cross/same ratio goes from 7x (CIFAR-10) to 26x (CIFAR-100)
- ViT and MLP-Mixer form a shared "attention family" distinct from CNNs

## Structure

```
paper/       LaTeX source + refs.bib
results/     JSON results from CIFAR-10 (5 seeds) and CIFAR-100 (10 seeds)
experiment/  Python experiment code for both datasets
```

## Citation

Target venue: ICLR 2027
