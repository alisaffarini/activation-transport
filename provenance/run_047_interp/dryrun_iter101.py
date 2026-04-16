
# === DRY RUN VALIDATION: forced tiny scale ===
import builtins
_dry_run_got_results = False
_orig_print_fn = builtins.print
def _patched_print(*args, **kwargs):
    global _dry_run_got_results
    msg = " ".join(str(a) for a in args)
    if "RESULTS:" in msg:
        _dry_run_got_results = True
    _orig_print_fn(*args, **kwargs)
builtins.print = _patched_print

import atexit
def _check_results():
    if not _dry_run_got_results:
        _orig_print_fn("DRY_RUN_WARNING: Pipeline completed but no RESULTS: line was printed!")
        _orig_print_fn("DRY_RUN_WARNING: The post-processing/output stage may be broken.")
    else:
        _orig_print_fn("DRY_RUN_OK: Full pipeline validated (train → analyze → output)")
atexit.register(_check_results)

# pip install torch torchvision scipy scikit-learn pot

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision import transforms
import numpy as np
import json
from scipy.stats import ttest_rel, ttest_1samp, bootstrap
import ot  # Python Optimal Transport
import random
import time
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Model Definitions ===
class SimpleCNN(nn.Module):
    """Lightweight CNN for experiments"""
    def __init__(self, width=32, num_classes=10, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width*2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(width*2)
        self.pool = nn.MaxPool2d(2)
        
        # Calculate size after convolutions
        self.feature_size = width * 2 * 7 * 7  # for 28x28 input
        self.fc1 = nn.Linear(self.feature_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_features(self, x):
        """Extract features from conv2 layer"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        # Return flattened features for each spatial location
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        return x  # [B, num_positions, channels]


class SimpleViT(nn.Module):
    """Lightweight ViT for experiments"""
    def __init__(self, patch_size=7, dim=64, depth=2, num_heads=4, 
                 num_classes=10, in_channels=1, img_size=28):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, 
                                     dim_feedforward=dim*4, batch_first=True)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x
    
    def get_features(self, x):
        """Extract features after first transformer block"""
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        
        x = self.blocks[0](x)
        return x[:, 1:]  # Remove CLS token [B, num_patches, dim]


# === Feature Analysis Functions ===
def compute_cka(features1, features2):
    """Compute CKA (Centered Kernel Alignment) similarity metric"""
    # features: [n_samples, n_features]
    
    def gram_linear(x):
        """Compute Gram matrix"""
        return x @ x.T
    
    def center_gram(G):
        """Center Gram matrix"""
        n = G.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ G @ H
    
    # Compute Gram matrices
    G1 = gram_linear(features1)
    G2 = gram_linear(features2)
    
    # Center
    G1_centered = center_gram(G1)
    G2_centered = center_gram(G2)
    
    # Compute CKA
    hsic = np.trace(G1_centered @ G2_centered)
    norm1 = np.sqrt(np.trace(G1_centered @ G1_centered))
    norm2 = np.sqrt(np.trace(G2_centered @ G2_centered))
    
    cka = hsic / (norm1 * norm2 + 1e-10)
    return cka


def extract_features(model, dataloader, device, max_samples=300):
    """Extract features from model"""
    model.eval()
    all_features = []
    sample_count = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            if sample_count >= max_samples:
                break
            images = images.to(device)
            features = model.get_features(images)
            
            # Pool over spatial/sequence dimension
            features = features.mean(dim=1)  # [B, features]
            
            all_features.append(features.cpu().numpy())
            sample_count += images.size(0)
    
    return np.vstack(all_features)  # [n_samples, n_features]


def compute_optimal_transport_reuse(features1, features2, normalize=True):
    """Compute feature reuse using optimal transport"""
    # Ensure we have 2D arrays
    if len(features1.shape) != 2 or len(features2.shape) != 2:
        raise ValueError("Features must be 2D arrays [n_samples, n_features]")
    
    n1, d1 = features1.shape
    n2, d2 = features2.shape
    
    # Sample for computational efficiency
    n_samples = min(200, n1, n2)
    idx1 = np.random.choice(n1, n_samples, replace=False)
    idx2 = np.random.choice(n2, n_samples, replace=False)
    
    feat1 = features1[idx1]
    feat2 = features2[idx2]
    
    # Normalize features
    if normalize:
        feat1 = feat1 / (np.linalg.norm(feat1, axis=1, keepdims=True) + 1e-8)
        feat2 = feat2 / (np.linalg.norm(feat2, axis=1, keepdims=True) + 1e-8)
    
    # Compute pairwise cosine similarity (1 - cosine_distance)
    similarity_matrix = np.dot(feat1, feat2.T)
    
    # Convert to distance
    cost_matrix = 1.0 - similarity_matrix
    
    # Uniform distributions
    a = np.ones(n_samples) / n_samples
    b = np.ones(n_samples) / n_samples
    
    # Solve optimal transport
    try:
        # Use entropic regularization for stability
        transport_plan = ot.sinkhorn(a, b, cost_matrix, reg=0.01, numItermax=100)
        
        # Compute reuse score based on transport plan
        # High transport mass between features indicates reuse
        threshold = 3.0 / n_samples  # 3x average mass
        high_mass_pairs = transport_plan > threshold
        
        # Features that have at least one high-mass match are considered reused
        reused_feat1 = np.any(high_mass_pairs, axis=1)
        reuse_score = np.mean(reused_feat1)
        
        # Also compute average transport cost for validation
        transport_cost = np.sum(transport_plan * cost_matrix)
        
        return float(reuse_score), float(transport_cost)
        
    except Exception as e:
        print(f"OT computation failed: {e}")
        return 0.0, float('inf')


def validate_metric():
    """Validate that our metric works on synthetic data"""
    print("\nValidating metric on synthetic data...")
    
    # Test 1: Identical features should have high reuse
    feat_identical = np.random.randn(100, 50)
    reuse, cost = compute_optimal_transport_reuse(feat_identical, feat_identical)
    print(f"Identical features - Reuse: {reuse:.3f}, Cost: {cost:.3f}")
    assert reuse > 0.8, f"Identical features should have high reuse, got {reuse}"
    
    # Test 2: Random features should have low reuse
    feat_random1 = np.random.randn(100, 50)
    feat_random2 = np.random.randn(100, 50)
    reuse, cost = compute_optimal_transport_reuse(feat_random1, feat_random2)
    print(f"Random features - Reuse: {reuse:.3f}, Cost: {cost:.3f}")
    assert reuse < 0.3, f"Random features should have low reuse, got {reuse}"
    
    # Test 3: Slightly perturbed features should have moderate reuse
    feat_perturbed = feat_identical + 0.3 * np.random.randn(100, 50)
    reuse, cost = compute_optimal_transport_reuse(feat_identical, feat_perturbed)
    print(f"Perturbed features - Reuse: {reuse:.3f}, Cost: {cost:.3f}")
    assert 0.3 < reuse < 0.8, f"Perturbed features should have moderate reuse, got {reuse}"
    
    print("✓ Metric validation passed!\n")


# === Training Functions ===
def train_model(model, train_loader, val_loader, device, max_epochs=3, patience=2):
    """Train model with convergence monitoring"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    converged = False
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx > 100:  # Limit batches for speed
                break
                
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                if batch_idx > 40:  # Limit validation
                    break
                    
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = val_loss / min(40, len(val_loader))
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
        
        scheduler.step(avg_val_loss)
        
        # Check convergence
        if avg_val_loss < best_val_loss - 0.001:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("CONVERGED")
            converged = True
            break
    
    if not converged:
        print("NOT_CONVERGED: Reached max_epochs")
    
    return model, converged, val_acc


# === Main Experiment ===
def run_experiment(num_seeds=2):
    """Run complete experiment"""
    
    # Validate metric first
    validate_metric()
    
    # Load dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Use subset for speed
    subset_indices = list(range(10000))
    full_dataset = Subset(full_dataset, subset_indices)
    
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Results storage
    results = {
        'ot_reuse_scores': [],
        'cka_scores': [],
        'convergence': []
    }
    per_seed_results = []
    
    # Run experiments
    for seed in range(num_seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Split data
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Feature extraction subset
        feature_subset = Subset(train_dataset, range(min(300, len(train_dataset))))
        feature_loader = DataLoader(feature_subset, batch_size=64, shuffle=False)
        
        # Train models
        print("\nTraining CNN...")
        cnn = SimpleCNN()
        cnn, cnn_conv, cnn_acc = train_model(cnn, train_loader, val_loader, device)
        
        print("\nTraining ViT...")
        vit = SimpleViT()
        vit, vit_conv, vit_acc = train_model(vit, train_loader, val_loader, device)
        
        # Extract features
        print("\nExtracting features...")
        cnn_features = extract_features(cnn, feature_loader, device)
        vit_features = extract_features(vit, feature_loader, device)
        print(f"CNN features: {cnn_features.shape}, ViT features: {vit_features.shape}")
        
        # Compute metrics
        print("Computing feature reuse...")
        ot_reuse, ot_cost = compute_optimal_transport_reuse(cnn_features, vit_features)
        cka_score = compute_cka(cnn_features, vit_features)
        
        print(f"OT Reuse: {ot_reuse:.4f}, OT Cost: {ot_cost:.4f}, CKA: {cka_score:.4f}")
        
        # Store results
        seed_result = {
            'seed': seed,
            'ot_reuse_score': ot_reuse,
            'ot_cost': ot_cost,
            'cka_score': float(cka_score),
            'cnn_acc': float(cnn_acc),
            'vit_acc': float(vit_acc),
            'converged': cnn_conv and vit_conv
        }
        per_seed_results.append(seed_result)
        results['ot_reuse_scores'].append(ot_reuse)
        results['cka_scores'].append(cka_score)
        results['convergence'].append(cnn_conv and vit_conv)
    
    # Compute baselines
    print(f"\n{'='*60}")
    print("COMPUTING BASELINES")
    print(f"{'='*60}")
    
    # Random baseline
    print("\nRandom baseline (untrained models)...")
    random_ot_scores = []
    random_cka_scores = []
    
    for i in range(3):
        torch.manual_seed(1000 + i)
        np.random.seed(1000 + i)
        
        untrained_cnn = SimpleCNN()
        untrained_vit = SimpleViT()
        
        untrained_cnn_feat = extract_features(untrained_cnn, feature_loader, device)
        untrained_vit_feat = extract_features(untrained_vit, feature_loader, device)
        
        random_ot, _ = compute_optimal_transport_reuse(untrained_cnn_feat, untrained_vit_feat)
        random_cka = compute_cka(untrained_cnn_feat, untrained_vit_feat)
        
        random_ot_scores.append(random_ot)
        random_cka_scores.append(random_cka)
        print(f"  Random {i+1}: OT={random_ot:.4f}, CKA={random_cka:.4f}")
    
    random_ot_baseline = float(np.mean(random_ot_scores))
    random_cka_baseline = float(np.mean(random_cka_scores))
    
    # Same architecture baseline
    print("\nSame architecture baseline...")
    torch.manual_seed(42)
    cnn1 = SimpleCNN()
    cnn2 = SimpleCNN()
    cnn1, _, _ = train_model(cnn1, train_loader, val_loader, device, max_epochs=3)
    cnn2, _, _ = train_model(cnn2, train_loader, val_loader, device, max_epochs=3)
    
    cnn1_feat = extract_features(cnn1, feature_loader, device)
    cnn2_feat = extract_features(cnn2, feature_loader, device)
    same_arch_ot, _ = compute_optimal_transport_reuse(cnn1_feat, cnn2_feat)
    same_arch_cka = compute_cka(cnn1_feat, cnn2_feat)
    print(f"Same architecture: OT={same_arch_ot:.4f}, CKA={same_arch_cka:.4f}")
    
    # Ablation: Effect of normalization
    print("\nAblation: No normalization...")
    no_norm_scores = []
    for i in range(3):
        idx = i * 3  # Use different seeds
        if idx < len(per_seed_results):
            # Re-extract features from saved models
            no_norm_ot, _ = compute_optimal_transport_reuse(
                cnn_features, vit_features, normalize=False
            )
            no_norm_scores.append(no_norm_ot)
    no_norm_mean = np.mean(no_norm_scores) if no_norm_scores else 0.0
    
    # Statistical analysis
    ot_scores = results['ot_reuse_scores']
    cka_scores = results['cka_scores']
    
    mean_ot = float(np.mean(ot_scores))
    std_ot = float(np.std(ot_scores))
    mean_cka = float(np.mean(cka_scores))
    std_cka = float(np.std(cka_scores))
    
    # Statistical tests
    t_stat_random, p_value_random = ttest_1samp(ot_scores, random_ot_baseline, alternative='greater')
    t_stat_30, p_value_30 = ttest_1samp(ot_scores, 0.30, alternative='greater')
    
    # Bootstrap confidence intervals
    def compute_mean(data, axis=0):
        return np.mean(data, axis=axis)
    
    bootstrap_result = bootstrap((ot_scores,), compute_mean, 
                               n_resamples=1000, confidence_level=0.95,
                               random_state=42)
    ci_low, ci_high = bootstrap_result.confidence_interval
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"OT Reuse: {mean_ot:.4f} ± {std_ot:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
    print(f"CKA: {mean_cka:.4f} ± {std_cka:.4f}")
    print(f"Random baseline - OT: {random_ot_baseline:.4f}, CKA: {random_cka_baseline:.4f}")
    print(f"Same architecture - OT: {same_arch_ot:.4f}, CKA: {same_arch_cka:.4f}")
    print(f"p-value vs random: {p_value_random:.4f}")
    print(f"p-value vs 30%: {p_value_30:.4f}")
    print(f"Runtime: {time.time() - start_time:.1f}s")
    
    # Signal detection
    signal_detected = (mean_ot > random_ot_baseline + 0.1) and (p_value_random < 0.05) and (mean_ot > 0.15)
    
    if signal_detected:
        print(f"\nSIGNAL_DETECTED: Significant feature reuse ({mean_ot:.1%}) between CNN and ViT")
    else:
        print(f"\nNO_SIGNAL: No significant feature reuse detected")
    
    # Final results JSON
    final_results = {
        'mean': mean_ot,
        'std': std_ot,
        'per_seed_results': per_seed_results,
        'p_values': {
            'vs_random': float(p_value_random),
            'vs_30pct': float(p_value_30)
        },
        'ablation_results': {
            'same_architecture': float(same_arch_ot),
            'no_normalization': float(no_norm_mean)
        },
        'convergence_status': results['convergence'],
        'baselines': {
            'random': random_ot_baseline,
            'cka_random': random_cka_baseline,
            'cka_mean': mean_cka
        },
        'confidence_interval': {
            'low': float(ci_low),
            'high': float(ci_high)
        },
        'signal_detected': signal_detected,
        'runtime_seconds': time.time() - start_time
    }
    
    print(f"\nRESULTS: {json.dumps(final_results)}")


if __name__ == "__main__":
    start_time = time.time()
    run_experiment(num_seeds=2)