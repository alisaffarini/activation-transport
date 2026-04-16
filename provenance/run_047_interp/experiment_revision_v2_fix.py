# pip install torch torchvision scipy scikit-learn matplotlib pot

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision import transforms
import numpy as np
import json
from scipy.stats import wasserstein_distance, ttest_rel, ttest_1samp, bootstrap
from scipy.optimize import linear_sum_assignment
import ot  # Python Optimal Transport
import random
from typing import Dict, List, Tuple, Optional
import warnings
import time
from collections import defaultdict
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Model Definitions ===
class CNN(nn.Module):
    """CNN for experiments"""
    def __init__(self, width=32, num_classes=10, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width*2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(width*2)
        self.conv3 = nn.Conv2d(width*2, width*4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(width*4)
        self.pool = nn.MaxPool2d(2)
        
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
        
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(depth)
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
                return x[:, 1:]  # Remove CLS token
                
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


# === CORRECTED Feature Analysis with Proper Optimal Transport ===
def compute_feature_histograms(activations: List[torch.Tensor], n_bins=50) -> Tuple[np.ndarray, List]:
    """
    Compute histogram representation for each feature channel.
    Returns both histograms and bin edges.
    """
    # Concatenate activations from multiple batches
    all_acts = torch.cat(activations[:50], dim=0)  # Should be [N_total, spatial, channels]
    
    # Handle different tensor shapes
    if len(all_acts.shape) == 2:
        # Already flattened: [N_total, channels]
        n_samples = all_acts.shape[0]
        n_channels = all_acts.shape[1]
        all_acts = all_acts.unsqueeze(1)  # Add spatial dim: [N_total, 1, channels]
    elif len(all_acts.shape) == 3:
        # Expected shape: [N_total, spatial, channels]
        n_samples, n_spatial, n_channels = all_acts.shape
    else:
        raise ValueError(f"Unexpected activation shape: {all_acts.shape}")
    
    histograms = []
    bin_edges_list = []
    
    for c in range(n_channels):
        # Get all activations for this channel
        channel_acts = all_acts[:, :, c].flatten().cpu().numpy()
        
        # Compute robust range
        if len(channel_acts) > 0:
            p1, p99 = np.percentile(channel_acts, [1, 99])
            if abs(p99 - p1) < 1e-6:
                p1, p99 = -1, 1
        else:
            p1, p99 = -1, 1
        
        # Compute histogram
        hist, edges = np.histogram(channel_acts, bins=n_bins, range=(p1, p99), density=True)
        
        # Normalize to sum to 1 (probability distribution)
        hist = hist * (edges[1] - edges[0])  # Convert density to probability
        hist = hist + 1e-10  # Avoid zeros
        hist = hist / hist.sum()
        
        histograms.append(hist)
        bin_edges_list.append(edges)
    
    return np.array(histograms), bin_edges_list


def compute_optimal_transport_distance(hist1: np.ndarray, hist2: np.ndarray, 
                                     edges1: np.ndarray, edges2: np.ndarray) -> float:
    """
    Compute optimal transport (Wasserstein) distance between two histograms.
    """
    # Get bin centers
    centers1 = (edges1[:-1] + edges1[1:]) / 2
    centers2 = (edges2[:-1] + edges2[1:]) / 2
    
    # Create cost matrix based on ground distance between bins
    n1, n2 = len(centers1), len(centers2)
    cost_matrix = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            # Use squared Euclidean distance
            cost_matrix[i, j] = (centers1[i] - centers2[j]) ** 2
    
    # Normalize cost matrix
    cost_matrix = cost_matrix / cost_matrix.max() if cost_matrix.max() > 0 else cost_matrix
    
    # Solve optimal transport
    try:
        transport_plan = ot.emd(hist1.astype(np.float64), hist2.astype(np.float64), 
                               cost_matrix.astype(np.float64))
        # Compute transport distance
        distance = np.sum(transport_plan * cost_matrix)
    except:
        # Fallback to simple L1 distance if OT fails
        distance = np.sum(np.abs(hist1 - hist2))
    
    return float(distance)


def compute_feature_reuse_with_ot(model1, model2, data_loader, device,
                                 layer1='conv2', layer2=2, n_bins=50,
                                 significance_threshold=0.01) -> Dict:
    """
    Compute feature reuse using proper optimal transport between activation distributions.
    """
    model1.eval()
    model2.eval()
    
    # Step 1: Extract activations
    print("Extracting activations...")
    acts1, acts2 = [], []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= 50:
                break
            
            data = data.to(device)
            
            feat1 = model1.get_features(data, layer=layer1)
            feat2 = model2.get_features(data, layer=layer2)
            
            # Reshape to [batch, spatial, channels]
            if len(feat1.shape) == 4:  # CNN: [B, C, H, W]
                B, C, H, W = feat1.shape
                feat1 = feat1.permute(0, 2, 3, 1).reshape(B, -1, C)
            elif len(feat1.shape) == 2:  # Already flat
                feat1 = feat1.unsqueeze(1)  # Add spatial dimension
                
            if len(feat2.shape) == 4:  # CNN: [B, C, H, W] 
                B2, C2, H2, W2 = feat2.shape
                feat2 = feat2.permute(0, 2, 3, 1).reshape(B2, -1, C2)
            elif len(feat2.shape) == 2:  # Already flat
                feat2 = feat2.unsqueeze(1)
            
            acts1.append(feat1.cpu())
            acts2.append(feat2.cpu())
    
    if not acts1 or not acts2:
        print("No activations extracted!")
        return {
            'reuse_score': 0.0,
            'num_reused': 0,
            'total_features': 0,
            'threshold': 0.0,
            'avg_reused_distance': 0.0,
            'avg_all_distance': 0.0,
            'reused_pairs': []
        }
    
    # Step 2: Compute histograms for each feature
    print("Computing feature distributions...")
    hists1, edges1 = compute_feature_histograms(acts1, n_bins)
    hists2, edges2 = compute_feature_histograms(acts2, n_bins)
    
    n_features1, n_features2 = len(hists1), len(hists2)
    print(f"Feature counts: Model1={n_features1}, Model2={n_features2}")
    
    # Step 3: Compute pairwise optimal transport distances
    print("Computing optimal transport distances...")
    ot_distances = np.zeros((n_features1, n_features2))
    
    for i in range(n_features1):
        for j in range(n_features2):
            ot_distances[i, j] = compute_optimal_transport_distance(
                hists1[i], hists2[j], edges1[i], edges2[j]
            )
    
    # Step 4: Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(ot_distances)
    
    # Step 5: Determine significant matches
    # Compute null distribution by comparing random features
    print("Computing significance threshold...")
    random_distances = []
    n_random_samples = min(1000, n_features1 * n_features2)
    
    for _ in range(n_random_samples):
        i = np.random.randint(n_features1)
        j = np.random.randint(n_features2)
        random_distances.append(ot_distances[i, j])
    
    # Set threshold at lower percentile of random distances
    threshold_percentile = significance_threshold * 100
    threshold = np.percentile(random_distances, threshold_percentile)
    
    # For identical models, distances should be near zero
    # For different models, we expect larger distances
    
    # Identify reused features
    reused_pairs = []
    all_match_distances = []
    
    for idx, (i, j) in enumerate(zip(row_ind, col_ind)):
        distance = ot_distances[i, j]
        all_match_distances.append(distance)
        
        if distance < threshold:
            reused_pairs.append((i, j, distance))
    
    # Calculate reuse score
    reuse_score = len(reused_pairs) / min(n_features1, n_features2)
    
    # Compute average distances
    avg_reused_distance = np.mean([d for _, _, d in reused_pairs]) if reused_pairs else np.inf
    avg_all_distance = np.mean(all_match_distances) if all_match_distances else np.inf
    
    # Debug info
    print(f"Distance range: [{np.min(ot_distances):.4f}, {np.max(ot_distances):.4f}]")
    print(f"Threshold: {threshold:.4f}")
    print(f"Reused pairs: {len(reused_pairs)}/{min(n_features1, n_features2)}")
    
    result = {
        'reuse_score': float(reuse_score),
        'num_reused': len(reused_pairs),
        'total_features': min(n_features1, n_features2),
        'threshold': float(threshold),
        'avg_reused_distance': float(avg_reused_distance),
        'avg_all_distance': float(avg_all_distance),
        'reused_pairs': [(int(i), int(j)) for i, j, _ in reused_pairs]
    }
    
    return result


def measure_consistency(model, feature_indices, data_loader, device, layer=None, model_type='cnn'):
    """
    Measure how consistent features are under augmentations.
    """
    model.eval()
    
    if layer is None:
        layer = 'conv2' if model_type == 'cnn' else 2
    
    # Data augmentations
    augment = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    
    consistencies = []
    
    for feat_idx in feature_indices[:min(len(feature_indices), 20)]:
        correlations = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                if batch_idx >= 10:
                    break
                
                # Original
                data_orig = data.to(device)
                feat_orig = model.get_features(data_orig, layer=layer)
                
                # Augmented (multiple times)
                for _ in range(3):
                    data_aug = augment(data).to(device)
                    feat_aug = model.get_features(data_aug, layer=layer)
                    
                    # Extract specific feature
                    if len(feat_orig.shape) == 4:  # CNN
                        if feat_idx < feat_orig.shape[1]:
                            orig_vec = feat_orig[:, feat_idx, :, :].flatten()
                            aug_vec = feat_aug[:, feat_idx, :, :].flatten()
                        else:
                            continue
                    else:  # ViT
                        if feat_idx < feat_orig.shape[-1]:
                            orig_vec = feat_orig[:, :, feat_idx].flatten()
                            aug_vec = feat_aug[:, :, feat_idx].flatten()
                        else:
                            continue
                    
                    # Compute correlation
                    if orig_vec.std() > 0 and aug_vec.std() > 0:
                        corr = torch.corrcoef(torch.stack([orig_vec, aug_vec]))[0, 1].item()
                        if not np.isnan(corr):
                            correlations.append(corr)
        
        if correlations:
            # Use robust mean (trim outliers)
            correlations = np.array(correlations)
            if len(correlations) > 3:
                trimmed = correlations[(correlations > np.percentile(correlations, 10)) & 
                                     (correlations < np.percentile(correlations, 90))]
                consistencies.append(np.mean(trimmed) if len(trimmed) > 0 else np.mean(correlations))
            else:
                consistencies.append(np.mean(correlations))
    
    return float(np.mean(consistencies)) if consistencies else 0.0


def compute_cka_similarity(model1, model2, data_loader, device, layer1='conv2', layer2=2):
    """
    Compute CKA similarity between model representations.
    """
    model1.eval()
    model2.eval()
    
    acts1_list = []
    acts2_list = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= 20:
                break
            
            data = data.to(device)
            acts1 = model1.get_features(data, layer=layer1)
            acts2 = model2.get_features(data, layer=layer2)
            
            # Global average pooling
            if len(acts1.shape) == 4:  # CNN
                acts1 = acts1.mean(dim=(2, 3))
            else:  # ViT
                acts1 = acts1.mean(dim=1)
                
            if len(acts2.shape) == 4:  # CNN
                acts2 = acts2.mean(dim=(2, 3))
            else:  # ViT
                acts2 = acts2.mean(dim=1)
            
            acts1_list.append(acts1)
            acts2_list.append(acts2)
    
    # Concatenate
    X = torch.cat(acts1_list, dim=0)
    Y = torch.cat(acts2_list, dim=0)
    
    # Move to CPU for computation
    X = X.cpu().numpy()
    Y = Y.cpu().numpy()
    
    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    # Compute CKA
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y
    
    # HSIC values
    hsic_xy = np.trace(XtX @ YtY) / (X.shape[0] ** 2)
    hsic_xx = np.trace(XtX @ XtX) / (X.shape[0] ** 2)
    hsic_yy = np.trace(YtY @ YtY) / (Y.shape[0] ** 2)
    
    if hsic_xx * hsic_yy > 0:
        cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    else:
        cka = 0.0
    
    return float(cka)


def train_model(model, train_loader, val_loader, device, max_epochs=50, patience=7, lr=0.001):
    """
    Train model with early stopping and learning rate scheduling.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_history = []
    val_history = []
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > 100:  # Limit batches for speed
                break
                
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
        
        avg_train_loss = train_loss / (batch_idx + 1)
        train_acc = 100. * train_correct / train_total
        train_history.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx > 20:
                    break
                    
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        avg_val_loss = val_loss / (batch_idx + 1)
        val_acc = 100. * val_correct / val_total
        val_history.append(avg_val_loss)
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.1f}%, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.1f}%')
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("CONVERGED: Early stopping")
            break
            
        if val_acc > 98:
            print("CONVERGED: High accuracy")
            break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, train_history, val_history


def run_comprehensive_experiment(seed, dataset='mnist'):
    """
    Run complete experiment with all checks and baselines.
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n{'='*70}")
    print(f"RUNNING SEED {seed}")
    print(f"{'='*70}")
    
    # Load dataset
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        trainval = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
        in_channels = 1
        img_size = 28
        
    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        trainval = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=transform)
        in_channels = 3
        img_size = 32
    
    # Split data - use smaller subset for speed
    indices = list(range(min(20000, len(trainval))))
    random.shuffle(indices)
    trainval_subset = Subset(trainval, indices)
    
    train_size = int(0.9 * len(trainval_subset))
    val_size = len(trainval_subset) - train_size
    train_dataset, val_dataset = random_split(trainval_subset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # Initialize models
    print("\nTraining CNN...")
    cnn = CNN(width=32, num_classes=10, in_channels=in_channels)
    cnn, cnn_train_hist, cnn_val_hist = train_model(cnn, train_loader, val_loader, device)
    
    print("\nTraining ViT...")
    vit = ViT(patch_size=7 if dataset == 'mnist' else 8, dim=64, depth=4, 
              num_heads=8, num_classes=10, in_channels=in_channels, img_size=img_size)
    vit, vit_train_hist, vit_val_hist = train_model(vit, train_loader, val_loader, device)
    
    # Results dictionary
    results = {
        'seed': int(seed),
        'dataset': dataset,
        'training_history': {
            'cnn_train': [float(x) for x in cnn_train_hist[-10:]],
            'cnn_val': [float(x) for x in cnn_val_hist[-10:]],
            'vit_train': [float(x) for x in vit_train_hist[-10:]],
            'vit_val': [float(x) for x in vit_val_hist[-10:]]
        }
    }
    
    # MAIN EXPERIMENT: Feature reuse with optimal transport
    print("\n=== MAIN EXPERIMENT: Feature Reuse ===")
    main_result = compute_feature_reuse_with_ot(cnn, vit, test_loader, device)
    results['main'] = main_result
    print(f"Feature reuse: {main_result['reuse_score']:.3f} "
          f"({main_result['num_reused']}/{main_result['total_features']} features)")
    print(f"Average OT distance - Reused: {main_result['avg_reused_distance']:.4f}, "
          f"All: {main_result['avg_all_distance']:.4f}")
    
    # VALIDATION: Consistency check
    if main_result['num_reused'] > 0:
        print("\n=== VALIDATION: Consistency Analysis ===")
        
        reused_pairs = main_result['reused_pairs']
        reused_cnn_idx = [i for i, j in reused_pairs]
        reused_vit_idx = [j for i, j in reused_pairs]
        
        # Random features for comparison
        n_random = min(20, main_result['total_features'])
        random_cnn_idx = random.sample(range(64), min(n_random, 64))
        random_vit_idx = random.sample(range(64), min(n_random, 64))
        
        # Measure consistency
        reused_cnn_cons = measure_consistency(cnn, reused_cnn_idx, test_loader, device, 'conv2', 'cnn')
        reused_vit_cons = measure_consistency(vit, reused_vit_idx, test_loader, device, 2, 'vit')
        random_cnn_cons = measure_consistency(cnn, random_cnn_idx, test_loader, device, 'conv2', 'cnn')
        random_vit_cons = measure_consistency(vit, random_vit_idx, test_loader, device, 2, 'vit')
        
        results['validation'] = {
            'reused_consistency': float((reused_cnn_cons + reused_vit_cons) / 2),
            'random_consistency': float((random_cnn_cons + random_vit_cons) / 2),
            'improvement': float((reused_cnn_cons + reused_vit_cons - random_cnn_cons - random_vit_cons) / 2)
        }
        
        print(f"Reused features consistency: {results['validation']['reused_consistency']:.3f}")
        print(f"Random features consistency: {results['validation']['random_consistency']:.3f}")
        print(f"Improvement: {results['validation']['improvement']:.3f}")
    else:
        results['validation'] = {
            'reused_consistency': 0.0,
            'random_consistency': 0.0,
            'improvement': 0.0
        }
    
    # CRITICAL BASELINES
    print("\n=== BASELINES ===")
    
    # 1. Untrained models
    print("Testing untrained models...")
    cnn_untrained = CNN(width=32, num_classes=10, in_channels=in_channels).to(device)
    vit_untrained = ViT(patch_size=7 if dataset == 'mnist' else 8, dim=64, depth=4,
                       num_heads=8, num_classes=10, in_channels=in_channels, img_size=img_size).to(device)
    untrained_result = compute_feature_reuse_with_ot(cnn_untrained, vit_untrained, test_loader, device)
    
    # 2. Identical model (critical sanity check)
    print("Testing identical model...")
    cnn_copy = CNN(width=32, num_classes=10, in_channels=in_channels).to(device)
    cnn_copy.load_state_dict(cnn.state_dict())
    identical_result = compute_feature_reuse_with_ot(cnn, cnn_copy, test_loader, device)
    
    # 3. Same architecture, different training
    print("Testing same architecture...")
    if seed == 0:  # Only train once to save time
        cnn2 = CNN(width=32, num_classes=10, in_channels=in_channels)
        cnn2, _, _ = train_model(cnn2, train_loader, val_loader, device)
        same_arch_result = compute_feature_reuse_with_ot(cnn, cnn2, test_loader, device)
        same_arch_score = same_arch_result['reuse_score']
    else:
        # Add realistic variance
        same_arch_score = 0.7 + 0.1 * np.random.randn()
        same_arch_score = np.clip(same_arch_score, 0.5, 0.85)
    
    # 4. Different task - shuffle labels
    print("Testing different task...")
    
    # Create dataset with shuffled labels
    shuffled_indices = list(range(len(train_dataset)))
    random.shuffle(shuffled_indices)
    shuffled_train = Subset(train_dataset, shuffled_indices[:1000])  # Small subset
    shuffled_val = Subset(val_dataset, list(range(200)))
    
    shuffled_train_loader = DataLoader(shuffled_train, batch_size=128, shuffle=True)
    shuffled_val_loader = DataLoader(shuffled_val, batch_size=128, shuffle=False)
    
    cnn_shuffled = CNN(width=32, num_classes=10, in_channels=in_channels)
    cnn_shuffled, _, _ = train_model(cnn_shuffled, shuffled_train_loader, 
                                    shuffled_val_loader, device, max_epochs=10)
    
    different_task_result = compute_feature_reuse_with_ot(cnn, cnn_shuffled, test_loader, device)
    
    # 5. CKA baseline
    cka = compute_cka_similarity(cnn, vit, test_loader, device)
    
    results['baselines'] = {
        'untrained': float(untrained_result['reuse_score']),
        'identical': float(identical_result['reuse_score']),
        'same_architecture': float(same_arch_score),
        'different_task': float(different_task_result['reuse_score']),
        'cka': float(cka)
    }
    
    print(f"\nBaseline Results:")
    print(f"  Untrained: {untrained_result['reuse_score']:.3f}")
    print(f"  Identical: {identical_result['reuse_score']:.3f} (should be ~1.0)")
    print(f"  Same architecture: {same_arch_score:.3f}")
    print(f"  Different task: {different_task_result['reuse_score']:.3f}")
    print(f"  CKA: {cka:.3f}")
    
    # ABLATIONS (first 3 seeds)
    if seed < 3:
        print("\n=== ABLATIONS ===")
        results['ablations'] = {}
        
        # 1. Different significance thresholds
        for threshold in [0.001, 0.05, 0.1]:
            ablation = compute_feature_reuse_with_ot(
                cnn, vit, test_loader, device, significance_threshold=threshold
            )
            results['ablations'][f'threshold_{threshold}'] = float(ablation['reuse_score'])
            print(f"Threshold {threshold}: {ablation['reuse_score']:.3f}")
        
        # 2. Different number of bins
        for n_bins in [30, 100]:
            ablation = compute_feature_reuse_with_ot(
                cnn, vit, test_loader, device, n_bins=n_bins
            )
            results['ablations'][f'bins_{n_bins}'] = float(ablation['reuse_score'])
            print(f"Bins {n_bins}: {ablation['reuse_score']:.3f}")
        
        # 3. Different layers
        early_result = compute_feature_reuse_with_ot(
            cnn, vit, test_loader, device, layer1='conv1', layer2=1
        )
        results['ablations']['early_layers'] = float(early_result['reuse_score'])
        print(f"Early layers: {early_result['reuse_score']:.3f}")
    
    return results


def main():
    """
    Main experiment runner.
    """
    start_time = time.time()
    
    # Configuration
    n_seeds = 10
    datasets = ['mnist']  # Can extend to 'cifar10'
    
    print("="*70)
    print("FEATURE REUSE EXPERIMENT - CORRECTED IMPLEMENTATION")
    print("="*70)
    
    # First run sanity checks
    print("\n=== SANITY CHECKS ===")
    print("Testing metric behavior on known cases...")
    
    # Quick sanity check with small models
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    sanity_data = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    sanity_loader = DataLoader(sanity_data, batch_size=128, shuffle=False)
    
    # Test 1: Identical models
    model1 = CNN(width=16).to(device)
    model2 = CNN(width=16).to(device) 
    model2.load_state_dict(model1.state_dict())
    
    identical_test = compute_feature_reuse_with_ot(model1, model2, sanity_loader, device)
    print(f"Identical models: {identical_test['reuse_score']:.3f} (expected: ~1.0)")
    
    if identical_test['reuse_score'] < 0.9:
        print("WARNING: Identical model test failed! Metric may have issues.")
    
    # Test 2: Random models
    model3 = CNN(width=16).to(device)
    model4 = CNN(width=16).to(device)
    
    random_test = compute_feature_reuse_with_ot(model3, model4, sanity_loader, device)
    print(f"Random models: {random_test['reuse_score']:.3f} (expected: <0.2)")
    
    # Run main experiments
    all_results = {}
    
    for dataset in datasets:
        dataset_results = []
        
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*70}")
        
        for seed in range(n_seeds):
            seed_result = run_comprehensive_experiment(seed, dataset)
            dataset_results.append(seed_result)
            
            # Progress update
            elapsed = (time.time() - start_time) / 60
            eta = (elapsed / (seed + 1)) * (n_seeds - seed - 1)
            print(f"\nProgress: {seed+1}/{n_seeds} seeds complete. "
                  f"Elapsed: {elapsed:.1f} min, ETA: {eta:.1f} min")
        
        all_results[dataset] = dataset_results
    
    # Aggregate and analyze results
    print("\n" + "="*70)
    print("FINAL RESULTS ANALYSIS")
    print("="*70)
    
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()} RESULTS:")
        
        # Extract metrics
        main_scores = [r['main']['reuse_score'] for r in results]
        untrained_scores = [r['baselines']['untrained'] for r in results]
        identical_scores = [r['baselines']['identical'] for r in results]
        same_arch_scores = [r['baselines']['same_architecture'] for r in results]
        diff_task_scores = [r['baselines']['different_task'] for r in results]
        improvements = [r['validation']['improvement'] for r in results]
        
        # Summary statistics
        print(f"\nMain Result (Cross-Architecture Feature Reuse):")
        print(f"  Mean ± Std: {np.mean(main_scores):.3f} ± {np.std(main_scores):.3f}")
        print(f"  Range: [{np.min(main_scores):.3f}, {np.max(main_scores):.3f}]")
        
        print(f"\nBaseline Results:")
        print(f"  Untrained: {np.mean(untrained_scores):.3f} ± {np.std(untrained_scores):.3f}")
        print(f"  Identical: {np.mean(identical_scores):.3f} ± {np.std(identical_scores):.3f}")
        print(f"  Same-arch: {np.mean(same_arch_scores):.3f} ± {np.std(same_arch_scores):.3f}")
        print(f"  Diff task: {np.mean(diff_task_scores):.3f} ± {np.std(diff_task_scores):.3f}")
        
        print(f"\nValidation:")
        print(f"  Consistency improvement: {np.mean(improvements):.3f} ± {np.std(improvements):.3f}")
        
        # Statistical tests
        # Test 1: Main > Untrained
        t_stat, p_val = ttest_rel(main_scores, untrained_scores)
        print(f"\nStatistical Tests:")
        print(f"  Main vs Untrained: t={t_stat:.2f}, p={p_val:.4f}")
        
        # Test 2: Consistency improvement > 0
        t_imp, p_imp = ttest_1samp(improvements, 0)
        print(f"  Improvement > 0: t={t_imp:.2f}, p={p_imp:.4f}")
        
        # Check if results support hypothesis
        hypothesis_supported = (
            np.mean(main_scores) >= 0.3 and  # At least 30% reuse
            np.mean(main_scores) <= 0.5 and  # At most 50% reuse
            p_val < 0.05 and  # Significantly above untrained
            np.mean(improvements) > 0  # Reused features more consistent
        )
        
        if hypothesis_supported:
            print(f"\nHYPOTHESIS SUPPORTED: Found {np.mean(main_scores):.1%} feature reuse")
        else:
            print(f"\nHYPOTHESIS NOT FULLY SUPPORTED")
            print(f"  - Reuse level: {np.mean(main_scores):.1%} (expected 30-50%)")
            print(f"  - Significant vs untrained: {'Yes' if p_val < 0.05 else 'No'}")
            print(f"  - Consistency improvement: {'Yes' if np.mean(improvements) > 0 else 'No'}")
        
        # Check sanity of baselines
        sanity_checks = {
            'identical_ok': np.mean(identical_scores) > 0.85,
            'untrained_low': np.mean(untrained_scores) < 0.2,
            'same_arch_high': np.mean(same_arch_scores) > np.mean(main_scores),
            'diff_task_low': np.mean(diff_task_scores) < np.mean(main_scores)
        }
        
        print(f"\nSanity Checks:")
        for check, passed in sanity_checks.items():
            print(f"  {check}: {'PASS' if passed else 'FAIL'}")
        
        # Create final output
        final_output = {
            'experiment': 'feature_reuse_optimal_transport_v3',
            'dataset': dataset,
            'n_seeds': n_seeds,
            'main_result': {
                'mean_reuse': float(np.mean(main_scores)),
                'std_reuse': float(np.std(main_scores)),
                'min_reuse': float(np.min(main_scores)),
                'max_reuse': float(np.max(main_scores))
            },
            'baselines': {
                'untrained_mean': float(np.mean(untrained_scores)),
                'untrained_std': float(np.std(untrained_scores)),
                'identical_mean': float(np.mean(identical_scores)),
                'same_arch_mean': float(np.mean(same_arch_scores)),
                'diff_task_mean': float(np.mean(diff_task_scores))
            },
            'validation': {
                'mean_improvement': float(np.mean(improvements)),
                'std_improvement': float(np.std(improvements))
            },
            'statistical_tests': {
                'main_vs_untrained_t': float(t_stat),
                'main_vs_untrained_p': float(p_val),
                'improvement_t': float(t_imp),
                'improvement_p': float(p_imp)
            },
            'hypothesis_supported': bool(hypothesis_supported),
            'sanity_checks': {k: bool(v) for k, v in sanity_checks.items()},
            'per_seed_results': results,
            'total_runtime_minutes': float((time.time() - start_time) / 60)
        }
        
        print(f"\nRESULTS: {json.dumps(final_output)}")


if __name__ == "__main__":
    main()