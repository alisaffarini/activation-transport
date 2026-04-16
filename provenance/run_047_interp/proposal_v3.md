## Final Proposal: Discovering Interpretability Phase Transitions During Neural Network Training

You're absolutely right about the technical gaps. Let me propose something with stronger theoretical grounding that addresses a fundamental question in interpretability.

### (1) EXACT NOVELTY CLAIM
I will demonstrate that neural networks undergo sharp "interpretability phase transitions" during training, where the dominant basis of representation suddenly shifts from one interpretable feature type to another (e.g., from texture to shape). These transitions are predictable from the eigenspectrum of the feature covariance matrix and correspond to plateaus in test accuracy, revealing a fundamental connection between optimization dynamics and interpretability.

### (2) CLOSEST PRIOR WORK
- **Fort & Jastrzebski (2019, "Large Scale Structure of Neural Network Loss Landscapes")**: Studies loss landscape geometry but doesn't connect to interpretability or feature transitions.
- **Frankle et al. (2020, "The Early Phase of Neural Network Training")**: Identifies critical early training phases but doesn't analyze what features emerge when.
- **Hermann & Lampinen (2020, "What shapes feature representations?")**: Studies feature learning dynamics but not phase transitions or their predictability.
- **Golechha & Dao (2024)**: Notes complex features emerge during training but doesn't identify discrete transitions or their governing dynamics.

### (3) EXPECTED CONTRIBUTION
1. **Fundamental discovery**: First evidence that interpretability emerges through discrete phase transitions, not gradual change
2. **Predictive theory**: Mathematical criterion (eigenvalue gap) that predicts when transitions occur
3. **New interpretability principle**: Features don't just emerge—they compete and undergo "interpretability selection"
4. **Practical insight**: Explains training plateaus and suggests when to apply interpretability-aware interventions

### (4) HYPOTHESIS
**H1**: Neural networks exhibit 2-3 discrete phase transitions during training where the dominant interpretable features suddenly change (e.g., Gabor filters → textures → semantic parts).

**H2**: These transitions are predictable: they occur when the eigenvalue gap λ₁/λ₂ of the feature activation covariance matrix exceeds 3.0.

**H3**: Phase transitions correspond to test accuracy plateaus lasting >5 epochs, revealing that the network is reorganizing its representations.

### (5) EXPERIMENTAL PLAN

**Theoretical Foundation**:
- Feature competition model: Let f₁, f₂, ... be interpretable feature detectors. During training, their relative importance evolves according to their utility for the task.
- Phase transition occurs when: d/dt(λ₁/λ₂) shows a discontinuity (sudden jump)
- This is analogous to order-disorder transitions in physics

**Setup (1 hour)**:
```python
# Models & Data
models = {
    'CNN': SimpleCNN(width=128),     # ~500K params
    'ResNet': ResNet20(width=16),    # ~270K params  
    'ViT': ViT_tiny(patch=4)         # ~500K params
}
datasets = ['CIFAR-10', 'CIFAR-100']  # Different granularities
checkpoints = every_epoch for 100 epochs
```

**Main Experiment (2 hours)**:
1. **Feature Tracking Pipeline**:
   ```python
   for epoch in range(100):
       # Extract conv2/block2/layer4 activations on 5K examples
       features = model.get_features(val_data)  # [5000, channels, h, w]
       
       # Compute interpretability via pre-trained probe
       gabor_score = gabor_probe(features)      # Texture detector
       shape_score = shape_probe(features)       # Shape detector  
       semantic_score = semantic_probe(features) # Object part detector
       
       # Compute eigenspectrum
       cov = compute_covariance(features.flatten(2))
       eigenvalues = torch.linalg.eigvalsh(cov)
       gap_ratio = eigenvalues[-1] / eigenvalues[-2]
       
       # Detect phase transition
       if gap_ratio > 3.0 and prev_gap < 3.0:
           mark_transition(epoch)
   ```

2. **Interpretability Probes** (pre-trained on ImageNet):
   - Gabor probe: Linear classifier trained to detect Gabor filter responses
   - Shape probe: Trained on shape-only datasets (Geirhos et al. 2019)
   - Semantic probe: Trained on object part annotations

3. **Transition Analysis**:
   - Identify all transitions where eigenvalue gap exceeds threshold
   - Compute change in probe accuracies before/after transition
   - Correlate with test accuracy plateaus (defined as <0.1% improvement for 5 epochs)

**Validation (0.5 hours)**:
1. **Causal Test**: Add regularization to prevent transitions:
   ```python
   loss += alpha * (eigenvalue_gap - 2.0)**2  # Keep gap below threshold
   ```
   Hypothesis: This should eliminate discrete transitions and hurt final accuracy

2. **Cross-Architecture**: Do CNNs, ResNets, and ViTs show transitions at similar relative points in training?

3. **Task Dependence**: Compare CIFAR-10 (coarse) vs CIFAR-100 (fine) to see if granularity affects transition timing

**Statistical Analysis (0.5 hours)**:
- Changepoint detection (PELT algorithm) to formally identify transitions
- Bootstrap significance of eigenvalue jumps
- Correlation between transition timing and final accuracy across 10 seeds

**Key Metrics**:
- Number and timing of detected phase transitions
- Magnitude of eigenvalue gap jumps
- Change in interpretability probe accuracies at transitions
- Correlation with accuracy plateaus (Spearman ρ)

This proposal addresses fundamental questions about how neural networks organize information during learning, with clear theoretical grounding in dynamical systems theory and practical implications for understanding training dynamics.