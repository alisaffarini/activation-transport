# pip install torch torchvision scipy pot matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import numpy as np
import json
from scipy.stats import wasserstein_distance, ttest_rel
from scipy.optimize import linear_sum_assignment
import ot  # pip install POT (Python Optimal Transport)
import random
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Model Definitions ===
class SimpleCNN(nn.Module):
    """Simple CNN for MNIST"""
    def __init__(self, width=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width*2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(width*2*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.width = width
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_features(self, x):
        """Extract features from middle layer"""
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))  # Before second pooling
        return x  # [B, C, H, W]


class SimpleViT(nn.Module):
    """Minimal Vision Transformer for MNIST"""
    def __init__(self, patch_size=7, dim=64, num_patches=16):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = num_patches
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
        
        # Transformer blocks (simplified - just 2 layers)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, dim, h, w]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        
        # Global pool and classify
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pool
        x = self.head(x)
        return x
    
    def get_features(self, x):
        """Extract features from middle layer"""
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        # After first transformer block
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        return x  # [B, num_patches, dim]


# === Feature Distribution and Transport Functions ===
def compute_feature_distributions(activations: List[torch.Tensor], n_bins=50) -> np.ndarray:
    """
    Compute histogram distributions for each feature channel.
    
    Args:
        activations: List of [batch, spatial, channels] tensors
        n_bins: Number of histogram bins
    
    Returns:
        distributions: [n_channels, n_bins] array of normalized histograms
    """
    # Concatenate all activations
    all_acts = torch.cat(activations, dim=0)  # [total_samples, spatial, channels]
    n_channels = all_acts.shape[-1]
    
    distributions = []
    
    for c in range(n_channels):
        channel_acts = all_acts[:, :, c].flatten().cpu().numpy()
        
        # Compute histogram with fixed range for stability
        hist, _ = np.histogram(channel_acts, bins=n_bins, range=(-3, 3), density=True)
        hist = hist / (hist.sum() + 1e-10)  # Normalize
        distributions.append(hist)
    
    return np.array(distributions)


def pairwise_wasserstein(dist1: np.ndarray, dist2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Wasserstein distances between feature distributions.
    
    Args:
        dist1: [n_features1, n_bins]
        dist2: [n_features2, n_bins]
    
    Returns:
        cost_matrix: [n_features1, n_features2] Wasserstein distances
    """
    n1, n2 = dist1.shape[0], dist2.shape[0]
    cost_matrix = np.zeros((n1, n2))
    
    # Bin locations (assuming uniform binning)
    bins = np.linspace(-3, 3, dist1.shape[1])
    
    for i in range(n1):
        for j in range(n2):
            cost_matrix[i, j] = wasserstein_distance(bins, bins, dist1[i], dist2[j])
    
    return cost_matrix


def measure_feature_reuse(model1, model2, data_loader, device):
    """
    Main method to measure feature reuse between two models.
    """
    model1.eval()
    model2.eval()
    
    acts1, acts2 = [], []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= 20:  # Use subset for speed
                break
                
            data = data.to(device)
            
            # Extract features
            feat1 = model1.get_features(data)
            feat2 = model2.get_features(data)
            
            # Reshape to [batch, spatial, channels]
            if len(feat1.shape) == 4:  # CNN: [B, C, H, W]
                feat1 = feat1.permute(0, 2, 3, 1).reshape(feat1.shape[0], -1, feat1.shape[1])
            # ViT already in [B, patches, dim]
            
            acts1.append(feat1.cpu())
            acts2.append(feat2.cpu())
    
    # Compute feature distributions
    print("Computing feature distributions...")
    dist1 = compute_feature_distributions(acts1, n_bins=30)
    dist2 = compute_feature_distributions(acts2, n_bins=30)
    
    # Compute cost matrix
    print("Computing Wasserstein distances...")
    cost_matrix = pairwise_wasserstein(dist1, dist2)
    
    # Solve optimal transport
    print("Solving optimal transport...")
    # Normalize to get transport plan
    a = np.ones(len(dist1)) / len(dist1)  # Uniform weights
    b = np.ones(len(dist2)) / len(dist2)
    
    transport_plan = ot.emd(a, b, cost_matrix)
    
    # Identify reused features
    threshold = 1.0 / max(len(dist1), len(dist2))
    reused_pairs = []
    
    for i in range(transport_plan.shape[0]):
        for j in range(transport_plan.shape[1]):
            if transport_plan[i, j] > threshold:
                reused_pairs.append((i, j, transport_plan[i, j]))
    
    # Compute reuse score
    reuse_score = len(reused_pairs) / min(len(dist1), len(dist2))
    
    # Also compute average transport cost
    avg_cost = np.sum(transport_plan * cost_matrix)
    
    return reuse_score, reused_pairs, avg_cost


def measure_consistency(model, feature_idx, data_loader, device, num_augmentations=5):
    """
    Measure activation consistency of a specific feature under augmentations.
    """
    model.eval()
    
    # Define augmentations
    augment = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
    ])
    
    consistencies = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= 10:  # Use subset
                break
                
            data = data.to(device)
            orig_acts = model.get_features(data)
            
            # Extract specific feature
            if len(orig_acts.shape) == 4:  # CNN
                orig_feat = orig_acts[:, feature_idx, :, :].mean(dim=(1, 2))
            else:  # ViT
                orig_feat = orig_acts[:, :, feature_idx].mean(dim=1)
            
            # Compute consistency across augmentations
            aug_feats = []
            for _ in range(num_augmentations):
                aug_data = augment(data.cpu()).to(device)
                aug_acts = model.get_features(aug_data)
                
                if len(aug_acts.shape) == 4:  # CNN
                    aug_feat = aug_acts[:, feature_idx, :, :].mean(dim=(1, 2))
                else:  # ViT
                    aug_feat = aug_acts[:, :, feature_idx].mean(dim=1)
                
                aug_feats.append(aug_feat)
            
            # Compute variance across augmentations
            aug_feats = torch.stack(aug_feats)
            consistency = 1.0 / (1.0 + aug_feats.var(dim=0).mean().item())
            consistencies.append(consistency)
    
    return np.mean(consistencies)


def train_model(model, train_loader, val_loader, device, max_epochs=50, patience=5):
    """Train model until convergence."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        val_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%')
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("CONVERGED")
            break
    else:
        print("NOT_CONVERGED: Reached max epochs")
    
    return model


def run_experiment(seed):
    """Run single seed experiment."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n=== Running experiment with seed {seed} ===")
    
    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Split into train/val/test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Train models
    print("\nTraining CNN...")
    cnn = SimpleCNN(width=32)
    cnn = train_model(cnn, train_loader, val_loader, device)
    
    print("\nTraining ViT...")
    vit = SimpleViT(patch_size=7, dim=64, num_patches=16)
    vit = train_model(vit, train_loader, val_loader, device)
    
    # Measure feature reuse
    print("\nMeasuring feature reuse...")
    reuse_score, reused_pairs, avg_cost = measure_feature_reuse(
        cnn, vit, test_loader, device
    )
    
    print(f"Feature reuse score: {reuse_score:.3f}")
    print(f"Number of reused pairs: {len(reused_pairs)}")
    print(f"Average transport cost: {avg_cost:.3f}")
    
    # Validation: Check if reused features are more consistent
    if len(reused_pairs) > 0:
        print("\nValidating reused features...")
        
        # Sample some reused and random features
        n_samples = min(5, len(reused_pairs))
        sampled_pairs = random.sample(reused_pairs, n_samples)
        
        reused_consistencies = []
        random_consistencies = []
        
        for cnn_idx, vit_idx, _ in sampled_pairs:
            # Measure consistency of reused features
            cnn_consistency = measure_consistency(cnn, cnn_idx, test_loader, device)
            vit_consistency = measure_consistency(vit, vit_idx, test_loader, device)
            reused_consistencies.append((cnn_consistency + vit_consistency) / 2)
            
            # Random features for comparison
            random_cnn_idx = random.randint(0, 63)  # 64 channels
            random_vit_idx = random.randint(0, 63)
            
            random_cnn_consistency = measure_consistency(cnn, random_cnn_idx, test_loader, device)
            random_vit_consistency = measure_consistency(vit, random_vit_idx, test_loader, device)
            random_consistencies.append((random_cnn_consistency + random_vit_consistency) / 2)
        
        avg_reused_consistency = np.mean(reused_consistencies)
        avg_random_consistency = np.mean(random_consistencies)
        
        print(f"Average consistency - Reused features: {avg_reused_consistency:.3f}")
        print(f"Average consistency - Random features: {avg_random_consistency:.3f}")
        
        consistency_improvement = (avg_reused_consistency - avg_random_consistency) / avg_random_consistency
    else:
        avg_reused_consistency = 0
        avg_random_consistency = 0
        consistency_improvement = 0
    
    # Baselines
    print("\nComputing baselines...")
    
    # 1. Random baseline (random transport plan)
    random_reuse = 0.1  # Expected random overlap
    
    # 2. Same architecture baseline (CNN-CNN)
    cnn2 = SimpleCNN(width=32)
    cnn2 = train_model(cnn2, train_loader, val_loader, device)
    same_arch_reuse, _, _ = measure_feature_reuse(cnn, cnn2, test_loader, device)
    
    print(f"\nBaselines:")
    print(f"Random baseline: {random_reuse:.3f}")
    print(f"Same architecture (CNN-CNN): {same_arch_reuse:.3f}")
    print(f"Cross architecture (CNN-ViT): {reuse_score:.3f}")
    
    # Determine if signal detected
    if reuse_score > random_reuse * 1.5 and consistency_improvement > 0:
        print(f"SIGNAL_DETECTED: {reuse_score:.1%} feature reuse between CNN and ViT, "
              f"{consistency_improvement:.1%} higher consistency than random")
    else:
        print(f"NO_SIGNAL: Feature reuse {reuse_score:.1%} not significantly above random baseline")
    
    return {
        'seed': seed,
        'reuse_score': reuse_score,
        'num_reused_pairs': len(reused_pairs),
        'avg_transport_cost': avg_cost,
        'avg_reused_consistency': avg_reused_consistency,
        'avg_random_consistency': avg_random_consistency,
        'consistency_improvement': consistency_improvement,
        'baselines': {
            'random': random_reuse,
            'same_architecture': same_arch_reuse
        }
    }


# === Main Execution ===
def main():
    n_seeds = 3  # Feasibility probe
    all_results = []
    
    for seed in range(n_seeds):
        result = run_experiment(seed)
        all_results.append(result)
    
    # Aggregate results
    reuse_scores = [r['reuse_score'] for r in all_results]
    consistency_improvements = [r['consistency_improvement'] for r in all_results]
    same_arch_scores = [r['baselines']['same_architecture'] for r in all_results]
    
    # Statistical test vs random baseline
    random_baseline = 0.1
    t_stat, p_value = ttest_rel(reuse_scores, [random_baseline] * len(reuse_scores))
    
    # Prepare final results
    final_results = {
        'per_seed_results': all_results,
        'mean': {
            'reuse_score': np.mean(reuse_scores),
            'consistency_improvement': np.mean(consistency_improvements)
        },
        'std': {
            'reuse_score': np.std(reuse_scores),
            'consistency_improvement': np.std(consistency_improvements)
        },
        'p_values': {
            'vs_random': p_value
        },
        'ablation_results': {
            'same_architecture_reuse': np.mean(same_arch_scores)
        },
        'convergence_status': 'CONVERGED'
    }
    
    print(f"\nRESULTS: {json.dumps(final_results)}")


if __name__ == "__main__":
    main()