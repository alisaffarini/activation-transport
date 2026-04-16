# pip install torch torchvision scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision import transforms
import numpy as np
import json
from scipy.stats import wasserstein_distance, ttest_rel, ttest_ind
from scipy.optimize import linear_sum_assignment
import random
from typing import Dict, List, Tuple
import warnings
import time
from collections import defaultdict
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Model Definitions (Scaled up but still reasonable) ===
class CNN(nn.Module):
    """CNN for experiments - slightly larger than tiny version"""
    def __init__(self, width=32, num_classes=10, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width*2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(width*2)
        self.conv3 = nn.Conv2d(width*2, width*4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(width*4)
        self.pool = nn.MaxPool2d(2)
        
        # Calculate flattened size
        if in_channels == 1:  # MNIST
            self.flat_size = width*4*3*3
        else:  # CIFAR
            self.flat_size = width*4*4*4
            
        self.fc1 = nn.Linear(self.flat_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.width = width
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_features(self, x, layer='conv2'):
        """Extract features from specified layer"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        if layer == 'conv1':
            return x
        x = F.relu(self.bn2(self.conv2(x)))
        if layer == 'conv2':
            return x
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class ViT(nn.Module):
    """Vision Transformer for experiments"""
    def __init__(self, patch_size=7, dim=64, depth=4, num_heads=8, num_classes=10, in_channels=1, img_size=28):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, dim, h, w]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, dim]
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # CLS token
        x = self.head(x)
        return x
    
    def get_features(self, x, layer=2):
        """Extract features from specified transformer layer"""
        B = x.shape[0]
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == layer:
                return x[:, 1:]  # Remove CLS token for feature analysis
                
        return x[:, 1:]


class TransformerBlock(nn.Module):
    """Transformer block for ViT"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


# === Feature Analysis Functions ===
def compute_feature_distributions(activations: List[torch.Tensor], n_bins=50) -> np.ndarray:
    """
    Compute histogram distributions for each feature channel.
    """
    all_acts = torch.cat(activations, dim=0)  # [total_samples, spatial, channels]
    n_channels = all_acts.shape[-1]
    
    distributions = []
    
    for c in range(n_channels):
        channel_acts = all_acts[:, :, c].flatten().cpu().numpy()
        
        # Robust histogram with adaptive range
        p5, p95 = np.percentile(channel_acts, [5, 95])
        hist_range = (p5 - 0.1 * abs(p5), p95 + 0.1 * abs(p95))
        
        hist, _ = np.histogram(channel_acts, bins=n_bins, range=hist_range)
        hist = hist.astype(np.float32) + 1e-10
        hist = hist / hist.sum()
        distributions.append(hist)
    
    return np.array(distributions)


def compute_pairwise_distances(dist1: np.ndarray, dist2: np.ndarray, metric='wasserstein') -> np.ndarray:
    """
    Compute pairwise distances between distributions using specified metric.
    """
    n1, n2 = dist1.shape[0], dist2.shape[0]
    n_bins = dist1.shape[1]
    
    if metric == 'wasserstein':
        # Pre-compute bin locations
        bins = np.arange(n_bins, dtype=np.float32)
        
        cost_matrix = np.zeros((n1, n2), dtype=np.float32)
        for i in range(n1):
            for j in range(n2):
                # Compute CDF
                cdf1 = np.cumsum(dist1[i])
                cdf2 = np.cumsum(dist2[j])
                # L1 Wasserstein
                cost_matrix[i, j] = np.sum(np.abs(cdf1 - cdf2))
                
    elif metric == 'l2':
        # Simple L2 distance between distributions
        cost_matrix = np.zeros((n1, n2), dtype=np.float32)
        for i in range(n1):
            for j in range(n2):
                cost_matrix[i, j] = np.sqrt(np.sum((dist1[i] - dist2[j]) ** 2))
                
    elif metric == 'kl':
        # KL divergence (symmetrized)
        cost_matrix = np.zeros((n1, n2), dtype=np.float32)
        for i in range(n1):
            for j in range(n2):
                kl_ij = np.sum(dist1[i] * np.log(dist1[i] / (dist2[j] + 1e-10) + 1e-10))
                kl_ji = np.sum(dist2[j] * np.log(dist2[j] / (dist1[i] + 1e-10) + 1e-10))
                cost_matrix[i, j] = (kl_ij + kl_ji) / 2
    
    return cost_matrix


def measure_feature_reuse(model1, model2, data_loader, device, metric='wasserstein', 
                         n_bins=50, layer1='conv2', layer2=2):
    """
    Measure feature reuse between two models.
    """
    model1.eval()
    model2.eval()
    
    acts1, acts2 = [], []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= 50:  # Use reasonable subset
                break
                
            data = data.to(device)
            
            # Extract features
            feat1 = model1.get_features(data, layer=layer1)
            feat2 = model2.get_features(data, layer=layer2)
            
            # Reshape to [batch, spatial, channels]
            if len(feat1.shape) == 4:  # CNN: [B, C, H, W]
                feat1 = feat1.permute(0, 2, 3, 1).reshape(feat1.shape[0], -1, feat1.shape[1])
            # ViT already in [B, patches, dim]
            
            acts1.append(feat1.cpu())
            acts2.append(feat2.cpu())
    
    # Compute feature distributions
    dist1 = compute_feature_distributions(acts1, n_bins=n_bins)
    dist2 = compute_feature_distributions(acts2, n_bins=n_bins)
    
    # Compute cost matrix
    cost_matrix = compute_pairwise_distances(dist1, dist2, metric=metric)
    
    # Solve optimal assignment
    n1, n2 = cost_matrix.shape
    
    # Pad to square
    if n1 < n2:
        padded_cost = np.pad(cost_matrix, ((0, n2-n1), (0, 0)), 
                           constant_values=np.max(cost_matrix)*10)
    elif n1 > n2:
        padded_cost = np.pad(cost_matrix, ((0, 0), (0, n1-n2)), 
                           constant_values=np.max(cost_matrix)*10)
    else:
        padded_cost = cost_matrix
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(padded_cost)
    
    # Extract valid matches
    valid_matches = []
    for i, j in zip(row_ind, col_ind):
        if i < n1 and j < n2:
            valid_matches.append((i, j))
    
    # Identify reused features (below threshold)
    costs = [cost_matrix[i, j] for i, j in valid_matches]
    threshold = np.percentile(costs, 50)  # Use median
    reused_pairs = [(i, j) for i, j in valid_matches if cost_matrix[i, j] < threshold]
    
    # Compute metrics
    reuse_score = len(reused_pairs) / min(n1, n2)
    avg_cost = np.mean([cost_matrix[i, j] for i, j in reused_pairs]) if reused_pairs else 0
    
    return float(reuse_score), reused_pairs, float(avg_cost), cost_matrix


def measure_consistency(model, feature_idx, data_loader, device, layer=None, model_type='cnn'):
    """
    Measure activation consistency of a feature under augmentations.
    Fixed to handle both CNN and ViT models properly.
    """
    model.eval()
    
    # Set appropriate default layer if not specified
    if layer is None:
        layer = 'conv2' if model_type == 'cnn' else 2
    
    augment = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    
    original_acts = []
    augmented_acts = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= 20:
                break
                
            # Original activations
            data_orig = data.to(device)
            feat_orig = model.get_features(data_orig, layer=layer)
            
            if len(feat_orig.shape) == 4:  # CNN
                feat_orig = feat_orig[:, feature_idx, :, :].mean(dim=(1, 2))
            else:  # ViT
                feat_orig = feat_orig[:, :, feature_idx].mean(dim=1)
                
            original_acts.append(feat_orig.cpu())
            
            # Augmented activations
            data_aug = augment(data).to(device)
            feat_aug = model.get_features(data_aug, layer=layer)
                
            if len(feat_aug.shape) == 4:  # CNN
                feat_aug = feat_aug[:, feature_idx, :, :].mean(dim=(1, 2))
            else:  # ViT
                feat_aug = feat_aug[:, :, feature_idx].mean(dim=1)
                
            augmented_acts.append(feat_aug.cpu())
    
    # Compute correlation between original and augmented
    original_acts = torch.cat(original_acts).numpy()
    augmented_acts = torch.cat(augmented_acts).numpy()
    
    # Handle edge case where variance is zero
    if np.std(original_acts) == 0 or np.std(augmented_acts) == 0:
        return 0.0
    
    correlation = np.corrcoef(original_acts, augmented_acts)[0, 1]
    return float(correlation) if not np.isnan(correlation) else 0.0


def train_model(model, train_loader, val_loader, device, max_epochs=100, patience=10, lr=0.001):
    """
    Train model with convergence-based stopping and LR scheduling.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     patience=5, factor=0.5, min_lr=1e-6)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                val_loss += loss.item() * target.size(0)
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / val_total
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # LR scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("CONVERGED")
            break
    else:
        print("NOT_CONVERGED: Reached max epochs")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def bootstrap_confidence_interval(data, n_bootstrap=1000, ci=95):
    """Compute bootstrap confidence interval."""
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, (100-ci)/2)
    upper = np.percentile(bootstrap_means, (100+ci)/2)
    
    return float(lower), float(upper)


def run_single_experiment(seed, dataset_name='mnist'):
    """Run single seed experiment."""
    # Set all random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n{'='*50}")
    print(f"Running experiment - Dataset: {dataset_name.upper()}, Seed: {seed}")
    print(f"{'='*50}")
    
    # Load dataset
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, 
                                                   transform=transform)
        test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, 
                                                  transform=transform)
        in_channels = 1
        num_classes = 10
        img_size = 28
        
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                               std=[0.2023, 0.1994, 0.2010])
        ])
        
        train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, 
                                                     transform=transform)
        test_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True, 
                                                    transform=transform)
        in_channels = 3
        num_classes = 10
        img_size = 32
    
    # Train/val split
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Initialize models
    cnn = CNN(width=32, num_classes=num_classes, in_channels=in_channels)
    vit = ViT(patch_size=7 if dataset_name == 'mnist' else 8, dim=64, depth=4, 
              num_heads=8, num_classes=num_classes, in_channels=in_channels, img_size=img_size)
    
    # Train models
    print("\nTraining CNN...")
    cnn = train_model(cnn, train_loader, val_loader, device)
    
    print("\nTraining ViT...")
    vit = train_model(vit, train_loader, val_loader, device)
    
    results = defaultdict(dict)
    
    # Main experiment - measure feature reuse
    print("\nMeasuring feature reuse (main method)...")
    reuse_score, reused_pairs, avg_cost, cost_matrix = measure_feature_reuse(
        cnn, vit, test_loader, device, metric='wasserstein', n_bins=50
    )
    
    results['main']['reuse_score'] = reuse_score
    results['main']['num_reused'] = len(reused_pairs)
    results['main']['avg_cost'] = avg_cost
    
    print(f"Feature reuse score: {reuse_score:.3f}")
    print(f"Number of reused pairs: {len(reused_pairs)}")
    
    # Validate with consistency check
    if len(reused_pairs) > 0:
        print("\nValidating with consistency check...")
        
        # Sample some reused and random features
        n_samples = min(10, len(reused_pairs))
        sampled_pairs = random.sample(reused_pairs, n_samples)
        
        reused_consistencies = []
        for cnn_idx, vit_idx in sampled_pairs:
            # Fixed: specify layer and model_type for consistency measurement
            cnn_cons = measure_consistency(cnn, cnn_idx, test_loader, device, 
                                         layer='conv2', model_type='cnn')
            vit_cons = measure_consistency(vit, vit_idx, test_loader, device, 
                                         layer=2, model_type='vit')
            reused_consistencies.append((cnn_cons + vit_cons) / 2)
        
        # Random features
        random_consistencies = []
        for _ in range(n_samples):
            random_cnn_idx = random.randint(0, cost_matrix.shape[0]-1)
            random_vit_idx = random.randint(0, cost_matrix.shape[1]-1)
            cnn_cons = measure_consistency(cnn, random_cnn_idx, test_loader, device, 
                                         layer='conv2', model_type='cnn')
            vit_cons = measure_consistency(vit, random_vit_idx, test_loader, device, 
                                         layer=2, model_type='vit')
            random_consistencies.append((cnn_cons + vit_cons) / 2)
        
        results['validation']['reused_consistency'] = float(np.mean(reused_consistencies))
        results['validation']['random_consistency'] = float(np.mean(random_consistencies))
        results['validation']['improvement'] = results['validation']['reused_consistency'] - \
                                              results['validation']['random_consistency']
        
        print(f"Reused features consistency: {results['validation']['reused_consistency']:.3f}")
        print(f"Random features consistency: {results['validation']['random_consistency']:.3f}")
    
    # Baselines
    print("\nComputing baselines...")
    
    # 1. Random baseline
    results['baselines']['random'] = 0.1
    
    # 2. Same architecture baseline
    print("Training second CNN for same-architecture baseline...")
    cnn2 = CNN(width=32, num_classes=num_classes, in_channels=in_channels)
    cnn2 = train_model(cnn2, train_loader, val_loader, device)
    
    same_arch_reuse, _, _, _ = measure_feature_reuse(
        cnn, cnn2, test_loader, device, metric='wasserstein', n_bins=50
    )
    results['baselines']['same_architecture'] = float(same_arch_reuse)
    
    # 3. Untrained models baseline
    print("Computing untrained models baseline...")
    cnn_untrained = CNN(width=32, num_classes=num_classes, in_channels=in_channels).to(device)
    vit_untrained = ViT(patch_size=7 if dataset_name == 'mnist' else 8, dim=64, depth=4,
                       num_heads=8, num_classes=num_classes, in_channels=in_channels, 
                       img_size=img_size).to(device)
    
    untrained_reuse, _, _, _ = measure_feature_reuse(
        cnn_untrained, vit_untrained, test_loader, device, metric='wasserstein', n_bins=50
    )
    results['baselines']['untrained'] = float(untrained_reuse)
    
    # 4. Different task baseline (if MNIST, train on rotated version)
    if dataset_name == 'mnist':
        print("Computing different task baseline...")
        
        # Create rotated dataset
        rotate_transform = transforms.Compose([
            transforms.RandomRotation((90, 90)),  # Always rotate 90 degrees
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        rotated_train = torchvision.datasets.MNIST('./data', train=True, transform=rotate_transform)
        rotated_train, rotated_val = random_split(rotated_train, [train_size, val_size])
        
        rotated_train_loader = DataLoader(rotated_train, batch_size=128, shuffle=True)
        rotated_val_loader = DataLoader(rotated_val, batch_size=128, shuffle=False)
        
        cnn_rotated = CNN(width=32, num_classes=num_classes, in_channels=in_channels)
        cnn_rotated = train_model(cnn_rotated, rotated_train_loader, rotated_val_loader, device)
        
        diff_task_reuse, _, _, _ = measure_feature_reuse(
            cnn, cnn_rotated, test_loader, device, metric='wasserstein', n_bins=50
        )
        results['baselines']['different_task'] = float(diff_task_reuse)
    
    # Ablations
    print("\nRunning ablation studies...")
    
    # 1. Different distance metrics
    print("Testing different distance metrics...")
    for metric in ['l2', 'kl']:
        metric_reuse, _, _, _ = measure_feature_reuse(
            cnn, vit, test_loader, device, metric=metric, n_bins=50
        )
        results['ablations'][f'metric_{metric}'] = float(metric_reuse)
    
    # 2. Different bin counts
    print("Testing different histogram bin counts...")
    for n_bins in [20, 100]:
        bins_reuse, _, _, _ = measure_feature_reuse(
            cnn, vit, test_loader, device, metric='wasserstein', n_bins=n_bins
        )
        results['ablations'][f'bins_{n_bins}'] = float(bins_reuse)
    
    # 3. Different layers
    print("Testing different layer depths...")
    # CNN layer 1, ViT layer 1
    early_reuse, _, _, _ = measure_feature_reuse(
        cnn, vit, test_loader, device, metric='wasserstein', n_bins=50,
        layer1='conv1', layer2=1
    )
    results['ablations']['early_layers'] = float(early_reuse)
    
    # Print summary
    print(f"\n{'='*30}")
    print(f"EXPERIMENT SUMMARY - Seed {seed}")
    print(f"{'='*30}")
    print(f"Main result: {results['main']['reuse_score']:.3f}")
    print(f"Baselines - Random: {results['baselines']['random']:.3f}, "
          f"Same-arch: {results['baselines']['same_architecture']:.3f}, "
          f"Untrained: {results['baselines']['untrained']:.3f}")
    
    return dict(results)


def main():
    """Main experiment runner."""
    start_time = time.time()
    
    # Experiment configuration
    n_seeds = 10
    dataset = 'mnist'  # Can change to 'cifar10'
    
    all_results = []
    
    # Run experiments
    for seed in range(n_seeds):
        seed_results = run_single_experiment(seed, dataset)
        seed_results['seed'] = seed
        all_results.append(seed_results)
        
        # Print progress
        elapsed = time.time() - start_time
        print(f"\nCompleted {seed+1}/{n_seeds} seeds. Elapsed time: {elapsed/60:.1f} minutes")
    
    # Aggregate results
    print("\n" + "="*50)
    print("AGGREGATING RESULTS")
    print("="*50)
    
    # Extract main metrics
    reuse_scores = [r['main']['reuse_score'] for r in all_results]
    same_arch_scores = [r['baselines']['same_architecture'] for r in all_results]
    untrained_scores = [r['baselines']['untrained'] for r in all_results]
    
    # Validation metrics (if available)
    consistency_improvements = []
    for r in all_results:
        if 'validation' in r and 'improvement' in r['validation']:
            consistency_improvements.append(r['validation']['improvement'])
    
    # Statistical tests
    # 1. Test if cross-arch reuse > random baseline
    random_baseline = 0.1
    t_stat_random, p_val_random = ttest_rel(reuse_scores, 
                                           [random_baseline] * len(reuse_scores))
    
    # 2. Test if cross-arch reuse > untrained baseline
    t_stat_untrained, p_val_untrained = ttest_rel(reuse_scores, untrained_scores)
    
    # 3. Test if same-arch > cross-arch (expected)
    t_stat_same, p_val_same = ttest_rel(same_arch_scores, reuse_scores)
    
    # Bootstrap confidence intervals
    reuse_ci_low, reuse_ci_high = bootstrap_confidence_interval(reuse_scores)
    
    # Check convergence status
    convergence_status = "CONVERGED" if all('CONVERGED' in str(r) for r in all_results) else "PARTIAL"
    
    # Determine if signal detected
    signal_detected = (np.mean(reuse_scores) > random_baseline * 1.5 and 
                      p_val_random < 0.05 and p_val_untrained < 0.05)
    
    if signal_detected:
        print(f"\nSIGNAL_DETECTED: Cross-architecture feature reuse of {np.mean(reuse_scores):.1%} "
              f"significantly above random (p={p_val_random:.3f}) and untrained (p={p_val_untrained:.3f}) baselines")
    else:
        print(f"\nNO_SIGNAL: Feature reuse {np.mean(reuse_scores):.1%} not significantly above baselines")
    
    # Prepare final results
    final_results = {
        'experiment_config': {
            'dataset': dataset,
            'n_seeds': n_seeds,
            'models': ['CNN', 'ViT'],
            'metric': 'wasserstein',
            'n_bins': 50
        },
        'per_seed_results': all_results,
        'aggregate_results': {
            'mean': {
                'reuse_score': float(np.mean(reuse_scores)),
                'same_architecture': float(np.mean(same_arch_scores)),
                'untrained': float(np.mean(untrained_scores)),
                'consistency_improvement': float(np.mean(consistency_improvements)) if consistency_improvements else 0.0
            },
            'std': {
                'reuse_score': float(np.std(reuse_scores)),
                'same_architecture': float(np.std(same_arch_scores)),
                'untrained': float(np.std(untrained_scores)),
                'consistency_improvement': float(np.std(consistency_improvements)) if consistency_improvements else 0.0
            },
            'confidence_intervals': {
                'reuse_score_95ci': [reuse_ci_low, reuse_ci_high]
            }
        },
        'statistical_tests': {
            'vs_random': {
                't_stat': float(t_stat_random),
                'p_value': float(p_val_random)
            },
            'vs_untrained': {
                't_stat': float(t_stat_untrained),
                'p_value': float(p_val_untrained)
            },
            'same_vs_cross': {
                't_stat': float(t_stat_same),
                'p_value': float(p_val_same)
            }
        },
        'ablation_summary': {
            'metrics': defaultdict(list),
            'bins': defaultdict(list),
            'layers': []
        },
        'convergence_status': convergence_status,
        'signal_detected': signal_detected,
        'total_runtime_minutes': float((time.time() - start_time) / 60)
    }
    
    # Aggregate ablation results
    for r in all_results:
        if 'ablations' in r:
            for key, value in r['ablations'].items():
                if 'metric_' in key:
                    metric_name = key.split('_')[1]
                    final_results['ablation_summary']['metrics'][metric_name].append(value)
                elif 'bins_' in key:
                    bin_count = key.split('_')[1]
                    final_results['ablation_summary']['bins'][bin_count].append(value)
                elif key == 'early_layers':
                    final_results['ablation_summary']['layers'].append(value)
    
    # Average ablation results
    for metric, values in final_results['ablation_summary']['metrics'].items():
        final_results['ablation_summary']['metrics'][metric] = float(np.mean(values))
    
    for bins, values in final_results['ablation_summary']['bins'].items():
        final_results['ablation_summary']['bins'][bins] = float(np.mean(values))
    
    if final_results['ablation_summary']['layers']:
        final_results['ablation_summary']['layers'] = float(np.mean(final_results['ablation_summary']['layers']))
    
    # Print final summary
    print(f"\n{'='*50}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Cross-architecture reuse: {final_results['aggregate_results']['mean']['reuse_score']:.3f} "
          f"± {final_results['aggregate_results']['std']['reuse_score']:.3f}")
    print(f"95% CI: [{reuse_ci_low:.3f}, {reuse_ci_high:.3f}]")
    print(f"Same-architecture reuse: {final_results['aggregate_results']['mean']['same_architecture']:.3f} "
          f"± {final_results['aggregate_results']['std']['same_architecture']:.3f}")
    print(f"Statistical significance vs random: p = {p_val_random:.4f}")
    print(f"Statistical significance vs untrained: p = {p_val_untrained:.4f}")
    print(f"Total runtime: {final_results['total_runtime_minutes']:.1f} minutes")
    
    # Output final JSON
    print(f"\nRESULTS: {json.dumps(final_results)}")


if __name__ == "__main__":
    main()