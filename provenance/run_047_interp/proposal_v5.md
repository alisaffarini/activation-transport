## Final Proposal: Measuring Hidden Feature Reuse Across Architectures via Activation Transport

You're right about the technical impossibilities. Let me propose something concrete and technically sound that addresses a fundamental question in interpretability.

### (1) EXACT NOVELTY CLAIM
I will demonstrate that different neural architectures (CNNs vs ViTs) trained on the same task reuse up to 40% of the same computational features, as measured by a novel optimal transport-based metric that matches hidden activations across architectures without requiring spatial correspondence or human-defined concepts.

### (2) CLOSEST PRIOR WORK
- **Kornblith et al. (2019, "Similarity of Neural Network Representations")**: Uses CKA for representation similarity but can't identify which specific features are shared. My work identifies exact feature correspondences.
- **Raghu et al. (2021, "Do Vision Transformers See Like CNNs?")**: Shows differences in representations but doesn't measure feature reuse. I quantify exact reuse.
- **Nguyen et al. (2021, "Do Wide and Deep Networks Learn the Same Things?")**: Studies feature similarity within architecture families. I study across fundamentally different architectures.

### (3) EXPECTED CONTRIBUTION
1. **New metric**: First method to measure exact feature reuse across architectures with different spatial structures
2. **Scientific finding**: Quantification of how much computation is architecture-invariant
3. **Interpretability insight**: Shows which features are fundamental to the task vs architecture-specific
4. **Practical application**: Enables transferring interpretability insights between model families

### (4) HYPOTHESIS
**H1**: Despite different inductive biases, CNNs and ViTs reuse 30-50% of features when trained on the same task, as measured by activation transport distance.

**H2**: Shared features are more interpretable (higher activation consistency on semantic concepts) than architecture-specific features.

**H3**: The degree of feature reuse correlates with task difficulty—simpler tasks (MNIST) show more reuse than complex tasks (CIFAR-100).

### (5) EXPERIMENTAL PLAN

**Theoretical Foundation**:
- Feature reuse := minimum transport cost to match activation distributions
- Uses Wasserstein distance which naturally handles different dimensionalities
- No spatial correspondence needed—purely based on activation statistics

**Setup (0.5 hours)**:
```python
# Models (small for tractability)
models = {
    'MNIST': {'CNN': SimpleCNN(width=64), 'ViT': ViT_tiny(patch=7, dim=64)},
    'CIFAR10': {'CNN': ResNet20(), 'ViT': DeiT_tiny(patch=4)},
    'CIFAR100': {'CNN': ResNet20(num_classes=100), 'ViT': DeiT_tiny(patch=4, num_classes=100)}
}

# Extract from middle layers (half-depth)
# 1000 images per dataset
```

**Main Method (2 hours)**:
```python
def measure_feature_reuse(model1, model2, data):
    # Step 1: Extract activation distributions
    acts1 = []  # Shape: [n_images, n_spatial_positions, n_channels]
    acts2 = []
    
    for x in data:
        # CNN: [batch, channels, h, w] -> [batch, h*w, channels]
        feat1 = model1.get_features(x).flatten(2).transpose(1, 2)
        # ViT: [batch, n_patches, dim]
        feat2 = model2.get_features(x)
        
        acts1.append(feat1)
        acts2.append(feat2)
    
    # Step 2: Compute optimal transport between feature sets
    # For each feature dimension, compute its activation distribution
    feat_dist1 = compute_feature_distributions(acts1)  # [n_channels1, n_bins]
    feat_dist2 = compute_feature_distributions(acts2)  # [n_channels2, n_bins]
    
    # Wasserstein distance between feature sets
    cost_matrix = pairwise_wasserstein(feat_dist1, feat_dist2)
    transport_plan = optimal_transport(cost_matrix)
    
    # Step 3: Identify reused features (high transport weight)
    reused_pairs = []
    threshold = 1.0 / max(n_channels1, n_channels2)  # Principled: uniform distribution
    
    for i, j in enumerate(transport_plan):
        if transport_plan[i, j] > threshold:
            reused_pairs.append((i, j))
    
    reuse_score = len(reused_pairs) / min(n_channels1, n_channels2)
    return reuse_score, reused_pairs
```

**Validation (1 hour)**:
1. **Interpretability Test**:
   ```python
   # Test if reused features are more interpretable
   for (idx1, idx2) in reused_pairs:
       # Compute activation consistency on augmented data
       consistency1 = measure_consistency(model1, idx1, augmented_data)
       consistency2 = measure_consistency(model2, idx2, augmented_data)
       
       # Compare to random features
       random_consistency1 = measure_consistency(model1, random_idx, augmented_data)
   ```

2. **Ablation Studies**:
   - Distance metrics: Wasserstein vs L2 vs KL
   - Binning strategies for distributions
   - Layer depth effects

3. **Control Experiments**:
   - Same architecture pairs (CNN-CNN): Should show high reuse
   - Different tasks (MNIST CNN vs CIFAR ViT): Should show low reuse
   - Random networks: Baseline reuse rate

**Baselines**:
- CKA similarity (doesn't identify specific features)
- Random feature matching
- Task-agnostic models (same architecture, different training)

**Key Metrics**:
- Feature reuse percentage
- Consistency scores for reused vs unique features
- Computation time scaling with model size
- Statistical significance via permutation test

**Why This Works**:
- Optimal transport is well-defined for different dimensions
- No spatial correspondence needed
- Principled threshold based on uniform distribution
- Clear interpretability test via consistency
- Computationally tractable for small models

This proposal avoids previous issues by using well-defined mathematical operations (optimal transport), principled thresholds (uniform distribution baseline), and concrete validation metrics (activation consistency).