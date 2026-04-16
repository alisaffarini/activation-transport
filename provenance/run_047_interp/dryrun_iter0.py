
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
from collections import defaultdict
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Model Definitions ===
class SimpleCNN(nn.Module):
    """Lightweight CNN for fast experiments"""
    def __init__(self, width=32, num_classes=10, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width*2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(width*2)
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4)
        
        self.fc1 = nn.Linear(width*2*4*4, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.width = width
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_features(self, x):
        """Extract features from conv2 layer"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        return x  # [B, C, H, W]


class SimpleViT(nn.Module):
    """Lightweight ViT for fast experiments"""
    def __init__(self, patch_size=7, dim=64, depth=3, num_heads=4, 
                 num_classes=10, in_channels=1, img_size=28):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.dropout = nn.Dropout(0.1)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, 
                                      dim_feedforward=dim*4, batch_first=True,
                                      dropout=0.1)
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
        x = self.dropout(x)
        
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
        
        # Apply first block only
        x = self.blocks[0](x)
        return x[:, 1:]  # Remove CLS token, keep patches [B, N-1, D]


# === Feature Extraction and Analysis ===
def extract_features(model, dataloader, device, model_type='cnn', max_samples=500):
    """Extract features from model"""
    model.eval()
    all_features = []
    all_labels = []
    sample_count = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            if sample_count >= max_samples:
                break
            images = images.to(device)
            features = model.get_features(images)
            
            if model_type == 'cnn':
                # CNN: [B, C, H, W] -> flatten spatial, keep channels separate
                B, C, H, W = features.shape
                features = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
            else:
                # ViT: [B, N, D] already in right format
                pass
            
            all_features.append(features.cpu())
            all_labels.append(labels)
            sample_count += len(images)
    
    if len(all_features) == 0:
        raise ValueError("No features extracted!")
    
    features = torch.cat(all_features, dim=0)  # [n_samples, n_positions, n_features]
    labels = torch.cat(all_labels, dim=0)
    return features, labels


def compute_optimal_transport_matching(features1, features2, n_samples=100):
    """Compute feature matching using optimal transport"""
    # Sample random subset for computational efficiency
    n1, s1, d1 = features1.shape
    n2, s2, d2 = features2.shape
    
    idx1 = np.random.choice(n1, min(n_samples, n1), replace=False)
    idx2 = np.random.choice(n2, min(n_samples, n2), replace=False)
    
    feat1 = features1[idx1].reshape(-1, d1)  # [n_samples * s1, d1]
    feat2 = features2[idx2].reshape(-1, d2)  # [n_samples * s2, d2]
    
    # Normalize features
    feat1 = F.normalize(feat1, p=2, dim=1).numpy()
    feat2 = F.normalize(feat2, p=2, dim=1).numpy()
    
    # Compute cost matrix using cosine distance
    cost = 1 - np.dot(feat1, feat2.T)  # [s1*n, s2*n]
    
    # Solve optimal transport
    a = np.ones(len(feat1)) / len(feat1)
    b = np.ones(len(feat2)) / len(feat2)
    
    try:
        # Use entropic regularization for stability
        transport_plan = ot.sinkhorn(a, b, cost, reg=0.01)
        
        # Find matched features (those with high transport mass)
        threshold = 1.0 / min(len(feat1), len(feat2)) * 5  # 5x average mass
        matches = np.sum(transport_plan > threshold, axis=1)
        reuse_score = np.sum(matches > 0) / len(feat1)
        
        # Also compute transport distance
        transport_cost = np.sum(transport_plan * cost)
        
        return float(reuse_score), float(transport_cost)
    except:
        return 0.0, float('inf')


def compute_feature_reuse(model1, model2, dataloader, device, model1_type='cnn', model2_type='vit'):
    """Main function to compute feature reuse between two models"""
    print("Extracting features from model 1...")
    features1, labels1 = extract_features(model1, dataloader, device, model1_type)
    print(f"Model 1 features shape: {features1.shape}")
    
    print("Extracting features from model 2...")
    features2, labels2 = extract_features(model2, dataloader, device, model2_type)
    print(f"Model 2 features shape: {features2.shape}")
    
    print("Computing optimal transport matching...")
    reuse_score, transport_cost = compute_optimal_transport_matching(features1, features2)
    print(f"Reuse score: {reuse_score:.4f}, Transport cost: {transport_cost:.4f}")
    
    return reuse_score


# === Training Functions ===
def train_model(model, train_loader, val_loader, device, max_epochs=3, patience=2, lr=1e-3):
    """Train model with convergence-based stopping"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        
        for images, labels in train_loader:
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
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{max_epochs}: Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_acc:.1f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.1f}%")
        
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
def run_experiment(seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """Run full experiment with multiple seeds"""
    
    # Load dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    
    # Split train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    results = defaultdict(list)
    per_seed_results = []
    
    # Run experiments
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Create train/val split with seed
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Use subset for feature extraction
        feature_subset = Subset(train_dataset, range(500))
        feature_loader = DataLoader(feature_subset, batch_size=64, shuffle=False)
        
        # Train CNN
        print("\nTraining CNN...")
        cnn = SimpleCNN()
        cnn, cnn_converged, cnn_acc = train_model(cnn, train_loader, val_loader, device)
        
        # Train ViT
        print("\nTraining ViT...")
        vit = SimpleViT()
        vit, vit_converged, vit_acc = train_model(vit, train_loader, val_loader, device)
        
        # Compute feature reuse
        print("\nComputing feature reuse...")
        reuse_score = compute_feature_reuse(cnn, vit, feature_loader, device, 'cnn', 'vit')
        
        # Store results
        seed_result = {
            'seed': seed,
            'reuse_score': reuse_score,
            'cnn_acc': cnn_acc,
            'vit_acc': vit_acc,
            'converged': cnn_converged and vit_converged
        }
        per_seed_results.append(seed_result)
        results['reuse_scores'].append(reuse_score)
        results['convergence'].append(cnn_converged and vit_converged)
    
    # Compute baselines
    print(f"\n{'='*60}")
    print("COMPUTING BASELINES")
    print(f"{'='*60}")
    
    # Random baseline (untrained models)
    print("\n=== BASELINE: Untrained models ===")
    random_scores = []
    for i in range(3):
        torch.manual_seed(1000 + i)
        untrained_cnn = SimpleCNN()
        untrained_vit = SimpleViT()
        random_score = compute_feature_reuse(untrained_cnn, untrained_vit, feature_loader, 
                                           device, 'cnn', 'vit')
        random_scores.append(random_score)
    random_baseline = np.mean(random_scores)
    print(f"Random baseline: {random_baseline:.4f}")
    
    # Ablation: Same architecture
    print("\n=== ABLATION: Same Architecture ===")
    torch.manual_seed(42)
    cnn1 = SimpleCNN()
    cnn2 = SimpleCNN()
    cnn1, _, _ = train_model(cnn1, train_loader, val_loader, device, max_epochs=3)
    cnn2, _, _ = train_model(cnn2, train_loader, val_loader, device, max_epochs=3)
    same_arch_score = compute_feature_reuse(cnn1, cnn2, feature_loader, device, 'cnn', 'cnn')
    print(f"Same architecture baseline: {same_arch_score:.4f}")
    
    # Statistical analysis
    reuse_scores = results['reuse_scores']
    
    # T-test vs random baseline
    if len(reuse_scores) > 1:
        t_stat, p_value_random = ttest_1samp(reuse_scores, random_baseline, alternative='greater')
    else:
        p_value_random = 1.0
    
    # T-test vs 30% (hypothesis threshold)
    t_stat_30, p_value_30 = ttest_1samp(reuse_scores, 0.3, alternative='greater')
    
    # Bootstrap confidence intervals
    def compute_mean(data, axis=0):
        return np.mean(data, axis=axis)
    
    if len(reuse_scores) > 1:
        bootstrap_result = bootstrap((reuse_scores,), compute_mean, 
                                   n_resamples=1000, confidence_level=0.95, 
                                   random_state=42)
        ci_low, ci_high = bootstrap_result.confidence_interval
    else:
        ci_low = ci_high = reuse_scores[0]
    
    # Summary
    mean_reuse = np.mean(reuse_scores)
    std_reuse = np.std(reuse_scores) if len(reuse_scores) > 1 else 0.0
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Mean reuse: {mean_reuse:.4f} ± {std_reuse:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Random baseline: {random_baseline:.4f}")
    print(f"p-value vs random: {p_value_random:.4f}")
    print(f"p-value vs 30%: {p_value_30:.4f}")
    print(f"Runtime: {time.time() - start_time:.1f}s")
    
    # Signal detection
    signal_detected = (mean_reuse > random_baseline + 0.1) and (p_value_random < 0.05)
    
    if signal_detected:
        print(f"\nSIGNAL_DETECTED: Significant feature reuse ({mean_reuse:.1%}) detected between architectures")
    else:
        print(f"\nNO_SIGNAL: No significant feature reuse detected")
    
    # Final results
    final_results = {
        'mean': float(mean_reuse),
        'std': float(std_reuse),
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
            'random': float(random_baseline)
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
    
    # Run with 10 seeds for publication quality
    run_experiment(seeds=list(range(10)))