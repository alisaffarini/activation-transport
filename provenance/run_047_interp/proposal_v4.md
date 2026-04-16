## Final Proposal: Automated Discovery of Interpretable Neural Computation Patterns via Causal Scrubbing at Scale

You're absolutely right about the circular reasoning and arbitrary thresholds. Let me propose something with stronger theoretical grounding that addresses a critical gap in the literature.

### (1) EXACT NOVELTY CLAIM
I will demonstrate the first automated method to discover interpretable computation patterns in neural networks without any human-defined feature labels, using a scaled-down version of causal scrubbing that's tractable for systematic search. Unlike existing work that tests pre-specified hypotheses, this method automatically discovers what computations the network performs by finding minimal activation subsets that preserve specific input-output behaviors.

### (2) CLOSEST PRIOR WORK
- **Wang et al. (2022, "Interpretability in the Wild")**: Manually specifies circuit hypotheses then tests them. My work automatically discovers circuits without human hypotheses.
- **Conmy et al. (2023, "Automated Circuit Discovery")**: Uses gradient-based attribution for circuit finding but requires supervised task decomposition. My method is fully unsupervised.
- **Chan et al. (2022, "Causal Scrubbing")**: Proposes the causal scrubbing framework but only for testing human-specified hypotheses on large models. I make it tractable for automated search on smaller models.

### (3) EXPECTED CONTRIBUTION
1. **Methodological advance**: First tractable algorithm for automated interpretable circuit discovery
2. **Scientific finding**: Evidence that many "interpretable" computations can be found without human priors
3. **Practical tool**: Open-source library for discovering what any small-to-medium network computes
4. **Theoretical insight**: Quantification of how much network computation is "interpretable" vs irreducibly complex

### (4) HYPOTHESIS
**H1**: At least 60% of a trained neural network's computation on in-distribution examples can be explained by sparse (≤5% of activations) computational patterns that are invariant to semantics-preserving input transformations.

**H2**: These automatically discovered patterns will align with human-interpretable concepts (measured by alignment with unsupervised clustering of input features) significantly better than random subsets of the same size.

**H3**: The number of distinct computational patterns scales sublinearly with model size, suggesting reuse of basic computational motifs.

### (5) EXPERIMENTAL PLAN

**Theoretical Foundation**:
A computational pattern is "interpretable" if:
1. It's sparse (few activations)
2. It's consistent (same activations for similar inputs)
3. It's necessary (removing it changes outputs)
4. It's sufficient (it alone produces partial output)

**Setup (0.5 hours)**:
```python
# Models: Small enough for exhaustive analysis
models = {
    'MLP_MNIST': MLP(784, [128, 64], 10),        # ~100K params
    'CNN_CIFAR': SimpleCNN(channels=[32, 64]),   # ~200K params
    'Transformer': MiniGPT(layers=4, dim=128)     # ~300K params on next-token
}

# Data: 1000 examples per class
datasets = ['MNIST', 'CIFAR-10', 'TinyShakespeare']
```

**Main Algorithm (2 hours)**:
```python
def discover_computation_patterns(model, data):
    patterns = []
    
    # Step 1: Generate hypothesis bank via activation clustering
    activations = get_all_activations(model, data[:1000])
    clusters = hierarchical_cluster(activations, n_clusters=100)
    
    # Step 2: Test each cluster for interpretability
    for cluster in clusters:
        # Get activation mask for this cluster
        mask = create_mask(cluster)  # Binary mask over all neurons
        
        # Test necessity: Does masking change outputs?
        orig_output = model(data)
        masked_output = model_with_mask(data, mask)
        necessity_score = KL_divergence(orig_output, masked_output)
        
        # Test sufficiency: Does mask alone preserve some behavior?
        only_mask_output = model_with_only_mask(data, mask)
        sufficiency_score = behavioral_similarity(orig_output, only_mask_output)
        
        # Test consistency: Same pattern across augmentations?
        aug_data = augment(data)  # Rotation, crop, etc.
        consistency_score = mask_overlap(data, aug_data, mask)
        
        if necessity_score > 0.1 and sufficiency_score > 0.3 and consistency_score > 0.8:
            patterns.append({
                'mask': mask,
                'sparsity': mask.sum() / mask.numel(),
                'behavior': characterize_behavior(mask, data)
            })
    
    return patterns
```

**Validation Experiments (1 hour)**:
1. **Interpretability validation**:
   ```python
   # Do discovered patterns align with known features?
   for pattern in patterns:
       # Visualize maximally activating inputs
       max_inputs = find_max_activating_inputs(pattern['mask'])
       
       # Compute alignment with unsupervised features
       unsupervised_concepts = extract_concepts_via_nmf(data)
       alignment = measure_concept_alignment(pattern, unsupervised_concepts)
   ```

2. **Ablation studies**:
   - Sparsity vs interpretability tradeoff
   - Number of clusters vs discovery rate
   - Different clustering methods (k-means, spectral)

3. **Baseline comparisons**:
   - Random masks of same sparsity
   - Gradient-based importance masks
   - Manual feature detectors (edges, colors)

**Key Metrics**:
- Percentage of computation explained by discovered patterns
- Average sparsity of interpretable patterns
- Alignment score with unsupervised concepts
- Number of unique patterns found
- Replication across different random seeds

**Statistical Validation (0.5 hours)**:
- Bootstrap confidence intervals for all metrics
- Permutation tests for alignment significance
- Multiple comparison correction (FDR) for pattern discovery

**Why This Works in 4 Hours**:
- Small models (100-300K params) allow exhaustive analysis
- Clustering reduces search space from exponential to ~100 candidates
- Simple behavioral tests avoid expensive causal intervention
- Standard datasets enable quick validation

This proposal directly addresses the automated discovery gap in interpretability research, avoids circular definitions by using behavioral tests rather than semantic labels, and provides a concrete algorithm that could become a standard tool for understanding neural networks.