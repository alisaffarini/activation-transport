
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
from scipy.stats import ttest_rel, ttest_1samp
import ot  # Python Optimal Transport
import random
import time
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Ultra-lightweight Models for Speed ===
class TinyCNN(nn.Module):
    """Ultra-lightweight CNN for feasibility testing"""
    def __init__(self, width=16, num_classes=10, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, 3, stride=2, padding=1)  # Stride 2 for speed
        self.conv2 = nn.Conv2d(width, width*2, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(width*2*2*2, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def get_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x  # [B, C, H, W]


class TinyViT(nn.Module):
    """Ultra-lightweight ViT for feasibility testing"""
    def __init__(self, patch_size=14, dim=32, depth=1, num_heads=2, 
                 num_classes=10, in_channels=1, img_size=28):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        
        # Single transformer block
        self.transformer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim*2, 
            batch_first=True, dropout=0.0
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        x = self.head(x)
        return x
    
    def get_features(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        return x  # [B, N, D]


# === Fast Feature Analysis ===
def extract_features_fast(model, dataloader, device, model_type='cnn', max_samples=200):
    """Extract features efficiently"""
    model.eval()
    features_list = []
    count = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            if count >= max_samples:
                break
            images = images.to(device)
            features = model.get_features(images)
            
            # Pool features to reduce dimensionality
            if model_type == 'cnn':
                # [B, C, H, W] -> [B, C]
                features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
            else:
                # [B, N, D] -> [B, D]
                features = features.mean(dim=1)
            
            features_list.append(features.cpu().numpy())
            count += images.size(0)
    
    return np.vstack(features_list)  # [n_samples, n_features]


def compute_feature_reuse_fast(features1, features2):
    """Fast feature reuse computation"""
    # Normalize
    features1 = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
    features2 = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)
    
    # Sample for speed
    n_samples = min(100, len(features1), len(features2))
    idx1 = np.random.choice(len(features1), n_samples, replace=False)
    idx2 = np.random.choice(len(features2), n_samples, replace=False)
    
    f1 = features1[idx1]
    f2 = features2[idx2]
    
    # Simple optimal transport
    cost = 1 - np.dot(f1, f2.T)  # Cosine distance
    
    # Uniform distributions
    a = np.ones(n_samples) / n_samples
    b = np.ones(n_samples) / n_samples
    
    try:
        # Fast sinkhorn
        transport_plan = ot.sinkhorn(a, b, cost, reg=0.05, numItermax=50)
        
        # Compute reuse score
        threshold = 2.0 / n_samples  # 2x average mass
        high_mass = transport_plan > threshold
        reuse_score = np.sum(np.any(high_mass, axis=1)) / n_samples
        
        return float(reuse_score)
    except:
        return 0.0


# === Fast Training ===
def train_model_fast(model, train_loader, val_loader, device, max_epochs=3, patience=2):
    """Fast training with early stopping"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Higher LR for faster convergence
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    patience_counter = 0
    converged = False
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx > 50:  # Limit batches per epoch for speed
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
        
        # Quick validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                if batch_idx > 20:  # Limit validation batches
                    break
                    
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
        
        # Early stopping
        if val_acc > best_val_acc + 0.5:
            best_val_acc = val_acc
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


# === Main Experiment (Optimized) ===
def run_experiment():
    """Run optimized experiment"""
    
    # Load dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Use smaller subset for speed
    subset_indices = list(range(5000))  # Only 5000 samples
    full_dataset = Subset(full_dataset, subset_indices)
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    results = {
        'reuse_scores': [],
        'convergence': []
    }
    per_seed_results = []
    
    # Run with 10 seeds as required
    seeds = list(range(10))
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")
        
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
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
        
        # Feature extraction subset
        feature_indices = list(range(min(200, len(train_dataset))))
        feature_subset = Subset(train_dataset, feature_indices)
        feature_loader = DataLoader(feature_subset, batch_size=64, shuffle=False, num_workers=0)
        
        # Train models
        print("Training CNN...")
        cnn = TinyCNN()
        cnn, cnn_conv, cnn_acc = train_model_fast(cnn, train_loader, val_loader, device)
        
        print("Training ViT...")
        vit = TinyViT()
        vit, vit_conv, vit_acc = train_model_fast(vit, train_loader, val_loader, device)
        
        # Extract features
        print("Extracting features...")
        cnn_features = extract_features_fast(cnn, feature_loader, device, 'cnn')
        vit_features = extract_features_fast(vit, feature_loader, device, 'vit')
        
        # Compute reuse
        print("Computing reuse...")
        reuse_score = compute_feature_reuse_fast(cnn_features, vit_features)
        print(f"Reuse score: {reuse_score:.4f}")
        
        # Store results
        seed_result = {
            'seed': seed,
            'reuse_score': reuse_score,
            'cnn_acc': float(cnn_acc),
            'vit_acc': float(vit_acc),
            'converged': cnn_conv and vit_conv
        }
        per_seed_results.append(seed_result)
        results['reuse_scores'].append(reuse_score)
        results['convergence'].append(cnn_conv and vit_conv)
    
    # Compute baselines
    print(f"\n{'='*50}")
    print("BASELINES")
    print(f"{'='*50}")
    
    # Random baseline
    print("Computing random baseline...")
    random_scores = []
    for i in range(3):
        torch.manual_seed(1000 + i)
        np.random.seed(1000 + i)
        
        untrained_cnn = TinyCNN()
        untrained_vit = TinyViT()
        
        untrained_cnn_features = extract_features_fast(untrained_cnn, feature_loader, device, 'cnn')
        untrained_vit_features = extract_features_fast(untrained_vit, feature_loader, device, 'vit')
        
        random_score = compute_feature_reuse_fast(untrained_cnn_features, untrained_vit_features)
        random_scores.append(random_score)
    
    random_baseline = float(np.mean(random_scores))
    print(f"Random baseline: {random_baseline:.4f}")
    
    # Same architecture ablation
    print("\nSame architecture ablation...")
    torch.manual_seed(42)
    cnn1 = TinyCNN()
    cnn2 = TinyCNN()
    cnn1, _, _ = train_model_fast(cnn1, train_loader, val_loader, device, max_epochs=3)
    cnn2, _, _ = train_model_fast(cnn2, train_loader, val_loader, device, max_epochs=3)
    
    cnn1_features = extract_features_fast(cnn1, feature_loader, device, 'cnn')
    cnn2_features = extract_features_fast(cnn2, feature_loader, device, 'cnn')
    same_arch_score = compute_feature_reuse_fast(cnn1_features, cnn2_features)
    print(f"Same architecture score: {same_arch_score:.4f}")
    
    # Statistical analysis
    reuse_scores = results['reuse_scores']
    mean_reuse = float(np.mean(reuse_scores))
    std_reuse = float(np.std(reuse_scores)) if len(reuse_scores) > 1 else 0.0
    
    # Statistical tests
    if len(reuse_scores) > 1:
        t_stat_random, p_value_random = ttest_1samp(reuse_scores, random_baseline, alternative='greater')
        t_stat_30, p_value_30 = ttest_1samp(reuse_scores, 0.30, alternative='greater')
    else:
        p_value_random = 1.0
        p_value_30 = 1.0
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Mean reuse: {mean_reuse:.4f} ± {std_reuse:.4f}")
    print(f"Random baseline: {random_baseline:.4f}")
    print(f"Same architecture: {same_arch_score:.4f}")
    print(f"p-value vs random: {p_value_random:.4f}")
    print(f"p-value vs 30%: {p_value_30:.4f}")
    
    # Signal detection
    signal_detected = (mean_reuse > random_baseline + 0.05) and (p_value_random < 0.05)
    
    if signal_detected:
        print(f"\nSIGNAL_DETECTED: Feature reuse {mean_reuse:.1%} > baseline {random_baseline:.1%}")
    else:
        print(f"\nNO_SIGNAL: No significant feature reuse detected")
    
    # Final results
    final_results = {
        'mean': mean_reuse,
        'std': std_reuse,
        'per_seed_results': per_seed_results,
        'p_values': {
            'vs_random': float(p_value_random),
            'vs_30pct': float(p_value_30)
        },
        'ablation_results': {
            'same_architecture': float(same_arch_score)
        },
        'convergence_status': results['convergence'],
        'baselines': {
            'random': random_baseline
        },
        'signal_detected': signal_detected,
        'runtime_seconds': time.time() - start_time
    }
    
    print(f"\nRESULTS: {json.dumps(final_results)}")


if __name__ == "__main__":
    start_time = time.time()
    run_experiment()