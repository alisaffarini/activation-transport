## RED-TEAM ATTACK: Breaking "Measuring Hidden Feature Reuse Across Architectures via Activation Transport"

### 1. CONFOUNDS
**CRITICAL - Measurement is fundamentally broken**: The validation clearly shows the OT metric returns 0.0 for identical features when it should return maximum similarity. This means ALL measurements would be inverted - the paper would report low reuse when there's high reuse and vice versa. The entire paper's foundation is wrong.

**MAJOR - Forced architectural alignment creates spurious matches**: Both models are artificially constrained to output 128-dimensional features. This bottleneck forces alignment where none might naturally exist. Real CNNs (ResNet) and ViTs have vastly different dimensional representations (2048 vs 768).

**MAJOR - Toy models on toy dataset**: SimpleCNN (2 conv layers) and SimpleViT (1 transformer block) on MNIST (28×28 grayscale digits) tell us nothing about ResNet-50 vs ViT-B/16 on ImageNet. These toy models likely share features simply because MNIST only requires edge detection.

### 2. ALTERNATIVE EXPLANATIONS
**CRITICAL - No baseline for chance alignment**: What percentage of "feature reuse" would we see between:
- Random networks?
- Networks trained on completely different tasks?
- Networks with scrambled weights?

Without this, the 40% claim is meaningless. In 128-dimensional space with L2 normalization, random vectors already have expected similarity.

**MAJOR - OT finds spurious alignments by design**: Optimal transport MUST find a mapping between any two distributions. It will report matches even between completely unrelated features. The paper mistakes "OT can find a mapping" for "features are actually shared."

### 3. STATISTICAL ISSUES
**CRITICAL - No results due to broken metric**: But even if fixed...

**MAJOR - No statistical testing**: Where are confidence intervals? Multiple seeds? The 40% number appears fabricated - no experiment ran successfully.

**MAJOR - Cherry-picked architecture pairs**: Why SimpleCNN vs SimpleViT? Why not test CNN vs CNN to establish baseline variance?

### 4. OVERCLAIMING
**CRITICAL - Claims about "computational features" without defining them**: What exactly is a "feature"? A neuron? A direction in activation space? A distributed representation? Without definition, the claim is unfalsifiable.

**CRITICAL - "Up to 40%" claim unsupported**: This specific number appears in the proposal but no successful experiment was run. This is academic fraud if published.

**MAJOR - Interpretability claims (H2) completely untested**: No semantic concepts were measured. No interpretability metrics implemented. Pure speculation.

### 5. MISSING EXPERIMENTS
**CRITICAL - No test for dataset dependence**: Train models on different datasets (MNIST vs CIFAR). If they still show "40% reuse," then it's not task-specific features but architecture artifacts.

**CRITICAL - No dimensionality analysis**: What happens with 64, 256, 512 dimensions? If "reuse" scales with dimension, it's just high-dimensional geometry, not semantic features.

**MAJOR - No comparison to known negative examples**: Compare CNN trained on images vs CNN trained on audio. These SHOULD show ~0% reuse. If they don't, the metric is broken.

### 6. GENERALIZATION
**CRITICAL - MNIST is qualitatively different from real vision**:
- Binary-like images vs natural images
- 784 pixels vs 50,176+ pixels  
- No color, texture, or complex shapes
- Solvable with linear models

Results on MNIST generalize to approximately nothing in modern deep learning.

**MAJOR - Architectural simplifications destroy the comparison**:
- Real ViTs use patch embeddings, not conv stems
- Real CNNs have skip connections, batch norm
- Real models have 50+ layers, not 2-3

This is like studying "car vs airplane efficiency" using toy models.

### 7. REPRODUCIBILITY  
**CRITICAL - Core code doesn't run**: The code literally cuts off mid-function:
```python
cls_toke  # <- INCOMPLETE
```

**MAJOR - Hidden degrees of freedom in OT**: Which OT algorithm? What regularization? What ground metric? How many iterations? These dramatically affect results.

**MINOR - Standard hyperparameters missing**: Learning rate, batch size, optimizer settings all unspecified.

---

## VERDICT: BROKEN

**Critical issues that completely invalidate the paper:**
1. The core metric is mathematically incorrect (returns 0 for identical inputs)
2. No baseline for random chance in high-dimensional matching  
3. The specific "40% reuse" claim is unsupported by any working experiment
4. Even if the metric worked, OT between different architectures is meaningless without proving it captures semantic similarity
5. MNIST results cannot support claims about "neural architectures" in general

**Why this paper would be desk-rejected:**
- The authors KNEW their metric was broken (validation caught it) but submitted anyway
- Claims specific numbers (40%) without any successful experiment
- Fundamental misunderstanding of what OT measures (it always finds mappings, even between noise)

**What would be needed to salvage this:**
1. Fix the metric and prove it captures semantic similarity (not just mathematical similarity)
2. Test on real architectures and datasets
3. Establish baselines for random chance
4. Actually run experiments before making numerical claims
5. Define what "computational feature" means operationally

This is not just weak science - it's broken science. The validation check proves the authors knew their method failed but proceeded anyway.