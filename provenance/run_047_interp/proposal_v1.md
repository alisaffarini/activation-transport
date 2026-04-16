## Research Proposal: Discovering Emergent Uncertainty-Encoding Features in Neural Networks via Progressive Sparse Autoencoders

### (1) EXACT NOVELTY CLAIM
For the first time, I will demonstrate that neural networks spontaneously develop interpretable "uncertainty detector" features during standard training, which can be discovered and tracked using sparse autoencoders (SAEs) applied to intermediate representations at different training stages. Specifically, I will show that these features emerge in a predictable sequence and correlate strongly with the network's calibration on different data distributions.

### (2) CLOSEST PRIOR WORK
- **Erdogan & Lucic (2025)**: Extends SAEs to scientific domains with group symmetries. My work differs by focusing on uncertainty-specific features rather than domain-specific symmetries, and by tracking feature emergence during training rather than analyzing post-hoc.
- **Golechha & Dao (2024)**: Notes that most interpretability work studies "trivial" behaviors. My work directly addresses this by targeting uncertainty representation—a complex, safety-critical capability.
- **Prisma (2025)**: Provides tools for vision model interpretability but doesn't address uncertainty or training dynamics. My work specifically targets how uncertainty representations emerge and become interpretable.

### (3) EXPECTED CONTRIBUTION
This work would contribute: (a) First evidence that uncertainty is encoded in discrete, interpretable features rather than distributed representations, (b) A new methodology for tracking interpretability emergence during training, (c) Practical insights for improving model calibration by monitoring uncertainty feature development, (d) A bridge between interpretability and uncertainty quantification—two critical but largely separate areas.

### (4) HYPOTHESIS
**H1**: Neural networks trained on classification tasks develop sparse, interpretable features that specifically encode uncertainty about predictions, and these features emerge in a consistent temporal order during training (early: "edge uncertainty", middle: "texture uncertainty", late: "semantic uncertainty").

**H2**: The activation patterns of these uncertainty features will strongly correlate (r > 0.7) with established uncertainty metrics (entropy, mutual information) on out-of-distribution data.

### (5) EXPERIMENTAL PLAN

**Setup (2 hours)**:
- Dataset: CIFAR-10 (train) + CIFAR-10-C (corrupted test) + SVHN (OOD test)
- Model: Small CNN (3 conv blocks, ~100k parameters) with intermediate feature extraction
- Training: Standard SGD, 50 epochs, checkpoint every 5 epochs

**Main Experiment (1.5 hours)**:
1. At each checkpoint, extract features from conv2 layer (256 channels) on 1000 examples
2. Train lightweight SAE (256→64→256) on these features for 10 epochs
3. For each SAE latent dimension, compute:
   - Activation on correctly classified examples
   - Activation on misclassified examples  
   - Activation on corrupted examples (blur, noise)
   - Activation on OOD examples (SVHN)
4. Identify "uncertainty features": latents with >2x higher activation on uncertain examples

**Analysis (0.5 hours)**:
- Track when uncertainty features first emerge during training
- Visualize maximally activating patches for top-5 uncertainty features
- Compute correlation between uncertainty feature activation and:
  - Prediction entropy
  - Softmax margin
  - True calibration error
- Statistical significance via bootstrap (n=1000)

**Baselines**:
- Random features baseline: Do random projections show similar patterns?
- Direct uncertainty: Can a linear probe on conv2 predict uncertainty as well as SAE features?

**Key Metrics**:
- Number of discovered uncertainty features per epoch
- Spearman correlation with uncertainty metrics
- Visualization quality of maximally activating patches

**Ablations**:
- SAE sparsity (L1 weight): Does sparser encoding reveal clearer uncertainty features?
- Architecture depth: Do uncertainty features emerge differently in shallow vs deep layers?

This experiment is highly tractable (small CNN, standard datasets) while addressing fundamental questions about how neural networks represent confidence—critical for trustworthy AI deployment.