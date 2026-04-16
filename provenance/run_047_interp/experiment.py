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
from scipy.stats import ttest_1samp
import ot  # Python Optimal Transport
import random
import time
import warnings
import sys
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Ultra-lightweight Models ===
class TinyCNN(nn.Module):
    """Minimal CNN for speed"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2),  # 12x12
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),  # 4x4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def get_features(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # [B, 128]


class TinyViT(nn.Module):
    """Minimal ViT for speed"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.patch_size = 14  # 4 patches total
        self.dim = 64
        self.num_patches = 4
        
        # Simple patch embedding
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(1, self.dim, kernel_size=14, stride=14),
            nn.Flatten(2),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
        
        # Single transformer layer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=self.dim, 
            nhead=4, 
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, num_classes)
        )
    
    def forward(self, x):
        B = x.shape[0]
        x = self.to_patch_embedding(x)  # [B, dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, dim]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        
        x = self.transformer(x)
        
        x = self.to_latent(x[:, 0])
        return self.mlp_head(x)
    
    def get_features(self, x):
        B = x.shape[0]
        x = self.to_patch_embedding(x)
        x = x.transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        
        x = self.transformer(x)
        return x[:, 0]  # [B, 64]


# === Fixed Optimal Transport Metric ===
def compute_ot_similarity(features1, features2, reg=0.1):
    """
    Compute feature similarity using optimal transport.
    Returns similarity in [0, 1] where 1 = identical.
    """
    if torch.is_tensor(features1):
        features1 = features1.cpu().numpy()
    if torch.is_tensor(features2):
        features2 = features2.cpu().numpy()
    
    n1, d1 = features1.shape
    n2, d2 = features2.shape
    
    # Handle dimension mismatch by zero-padding
    if d1 != d2:
        max_d = max(d1, d2)
        if d1 < max_d:
            features1 = np.pad(features1, ((0, 0), (0, max_d - d1)), mode='constant')
        if d2 < max_d:
            features2 = np.pad(features2, ((0, 0), (0, max_d - d2)), mode='constant')
    
    # L2 normalize
    features1 = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
    features2 = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)
    
    # Sample for speed
    n_samples = min(100, n1, n2)
    if n_samples < 20:
        return 0.0
        
    idx1 = np.random.choice(n1, n_samples, replace=False)
    idx2 = np.random.choice(n2, n_samples, replace=False)
    
    f1 = features1[idx1]
    f2 = features2[idx2]
    
    # Cost matrix (1 - cosine similarity)
    cost_matrix = 1.0 - np.dot(f1, f2.T)
    
    # Uniform marginals
    a = np.ones(n_samples) / n_samples
    b = np.ones(n_samples) / n_samples
    
    try:
        # Sinkhorn
        transport_plan = ot.sinkhorn(a, b, cost_matrix, reg=reg, numItermax=50)
        
        # Transport cost
        transport_cost = np.sum(transport_plan * cost_matrix)
        
        # Convert to similarity: normalize by expected random cost
        random_cost = np.mean(cost_matrix)
        similarity = 1.0 - (transport_cost / random_cost)
        
        return float(np.clip(similarity, 0, 1))
        
    except:
        return 0.0


def compute_cka(features1, features2):
    """Fast CKA computation"""
    if torch.is_tensor(features1):
        features1 = features1.cpu().numpy()
    if torch.is_tensor(features2):
        features2 = features2.cpu().numpy()
    
    n = min(len(features1), len(features2), 100)
    if n < 20:
        return 0.0
        
    f1 = features1[:n]
    f2 = features2[:n]
    
    # Center
    f1 = f1 - f1.mean(axis=0)
    f2 = f2 - f2.mean(axis=0)
    
    # Linear kernels
    K1 = f1 @ f1.T
    K2 = f2 @ f2.T
    
    # HSIC
    hsic = np.sum(K1 * K2) / (n ** 2)
    var1 = np.sum(K1 ** 2) / (n ** 2)
    var2 = np.sum(K2 ** 2) / (n ** 2)
    
    if var1 > 0 and var2 > 0:
        return float(hsic / np.sqrt(var1 * var2))
    return 0.0


def validate_metrics():
    """Quick validation of metrics"""
    print("Validating metrics...")
    
    # Test 1: Identical features
    np.random.seed(42)
    feat = np.random.randn(50, 64)
    
    ot_sim = compute_ot_similarity(feat, feat)
    cka_sim = compute_cka(feat, feat)
    print(f"Identical: OT={ot_sim:.3f}, CKA={cka_sim:.3f}")
    assert ot_sim > 0.9, f"OT for identical should be >0.9, got {ot_sim}"
    
    # Test 2: Random features
    feat2 = np.random.randn(50, 64)
    ot_sim = compute_ot_similarity(feat, feat2)
    print(f"Random: OT={ot_sim:.3f}")
    assert ot_sim < 0.6, f"OT for random should be <0.6, got {ot_sim}"
    
    print("✓ Metrics validated\n")


# === Fast Feature Extraction ===
def extract_features(model, dataloader, device, max_batches=10):
    """Extract features quickly"""
    model.eval()
    features = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            images = images.to(device)
            feats = model.get_features(images)
            features.append(feats.cpu())
    
    if len(features) == 0:
        return torch.zeros(0, 1)
        
    return torch.cat(features, dim=0)


# === Fast Training ===
def train_quick(model, train_loader, val_loader, device, max_epochs=15, patience=3):
    """Quick training with early stopping"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    no_improve = 0
    
    for epoch in range(max_epochs):
        # Train (limited batches)
        model.train()
        train_correct = 0
        train_total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            if i >= 50:  # Only 50 batches per epoch
                break
                
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Quick validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                if i >= 20:  # Only 20 batches
                    break
                    
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        if epoch % 3 == 0:  # Print every 3 epochs
            print(f"Epoch {epoch+1}: Train={train_acc:.1f}%, Val={val_acc:.1f}%")
        
        # Early stopping
        if val_acc > best_val_acc + 0.5:
            best_val_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print("CONVERGED")
            return model, True, val_acc
    
    print("NOT_CONVERGED: max epochs")
    return model, False, val_acc


# === Main Experiment ===
def run_experiment():
    """Optimized experiment that runs in <5 minutes"""
    
    # Validate metrics
    validate_metrics()
    
    # Load MNIST (fast)
    print("Loading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset = Subset(dataset, range(5000))  # Only 5000 samples
    
    # Fixed split for speed
    train_size = 4000
    val_size = 1000
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Results storage
    results = []
    ot_scores_cross = []  # CNN-ViT scores
    cka_scores_cross = []
    
    # Run experiments
    print("\n=== MAIN EXPERIMENT: CNN vs ViT ===")
    
    for seed in range(10):  # 10 seeds as required
        print(f"\nSeed {seed}:")
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Dataloaders
        train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=256, shuffle=False)
        
        # Feature loader (subset)
        feat_data = Subset(train_data, range(300))
        feat_loader = DataLoader(feat_data, batch_size=100, shuffle=False)
        
        # Train CNN
        cnn = TinyCNN()
        cnn, cnn_conv, cnn_acc = train_quick(cnn, train_loader, val_loader, device)
        
        # Train ViT
        vit = TinyViT()
        vit, vit_conv, vit_acc = train_quick(vit, train_loader, val_loader, device)
        
        # Extract features
        cnn_feats = extract_features(cnn, feat_loader, device)
        vit_feats = extract_features(vit, feat_loader, device)
        
        # Compute similarities
        ot_sim = compute_ot_similarity(cnn_feats, vit_feats)
        cka_sim = compute_cka(cnn_feats, vit_feats)
        
        print(f"OT={ot_sim:.3f}, CKA={cka_sim:.3f}")
        
        # Store
        results.append({
            'seed': seed,
            'ot_score': ot_sim,
            'cka_score': cka_sim,
            'cnn_acc': float(cnn_acc),
            'vit_acc': float(vit_acc),
            'converged': cnn_conv and vit_conv
        })
        ot_scores_cross.append(ot_sim)
        cka_scores_cross.append(cka_sim)
    
    # Baselines
    print("\n=== BASELINES ===")
    
    # 1. Random baseline
    print("\n1. Random untrained models:")
    random_ot = []
    for i in range(3):
        torch.manual_seed(100 + i)
        np.random.seed(100 + i)
        
        rand_cnn = TinyCNN()
        rand_vit = TinyViT()
        
        rand_cnn_feats = extract_features(rand_cnn, feat_loader, device, max_batches=5)
        rand_vit_feats = extract_features(rand_vit, feat_loader, device, max_batches=5)
        
        ot_sim = compute_ot_similarity(rand_cnn_feats, rand_vit_feats)
        random_ot.append(ot_sim)
        print(f"  Random {i+1}: OT={ot_sim:.3f}")
    
    random_baseline = float(np.mean(random_ot))
    
    # 2. Same architecture baseline
    print("\n2. Same architecture (CNN-CNN):")
    same_arch_ot = []
    
    for i in range(3):
        torch.manual_seed(200 + i)
        np.random.seed(200 + i)
        
        cnn1 = TinyCNN()
        cnn2 = TinyCNN()
        
        cnn1, _, _ = train_quick(cnn1, train_loader, val_loader, device, max_epochs=10)
        
        torch.manual_seed(300 + i)
        cnn2, _, _ = train_quick(cnn2, train_loader, val_loader, device, max_epochs=10)
        
        f1 = extract_features(cnn1, feat_loader, device)
        f2 = extract_features(cnn2, feat_loader, device)
        
        ot_sim = compute_ot_similarity(f1, f2)
        same_arch_ot.append(ot_sim)
        print(f"  Same arch {i+1}: OT={ot_sim:.3f}")
    
    same_arch_baseline = float(np.mean(same_arch_ot))
    
    # Ablation: Different regularization
    print("\n=== ABLATION: Regularization ===")
    ablation_results = {}
    
    # Use first trained models
    cnn = TinyCNN()
    vit = TinyViT()
    cnn, _, _ = train_quick(cnn, train_loader, val_loader, device, max_epochs=10)
    vit, _, _ = train_quick(vit, train_loader, val_loader, device, max_epochs=10)
    
    cnn_f = extract_features(cnn, feat_loader, device, max_batches=5)
    vit_f = extract_features(vit, feat_loader, device, max_batches=5)
    
    for reg in [0.01, 0.1, 1.0]:
        ot_sim = compute_ot_similarity(cnn_f, vit_f, reg=reg)
        ablation_results[f'reg_{reg}'] = float(ot_sim)
        print(f"Reg={reg}: OT={ot_sim:.3f}")
    
    # Statistical analysis
    print("\n=== STATISTICAL ANALYSIS ===")
    
    mean_ot = float(np.mean(ot_scores_cross))
    std_ot = float(np.std(ot_scores_cross))
    mean_cka = float(np.mean(cka_scores_cross))
    
    # Statistical tests
    _, p_vs_random = ttest_1samp(ot_scores_cross, random_baseline, alternative='greater')
    _, p_vs_30 = ttest_1samp(ot_scores_cross, 0.30, alternative='greater')
    
    print(f"\nCNN-ViT similarity: OT={mean_ot:.3f} ± {std_ot:.3f}")
    print(f"Baselines: Random={random_baseline:.3f}, Same-arch={same_arch_baseline:.3f}")
    print(f"p-value vs random: {p_vs_random:.4f}")
    print(f"p-value vs 30%: {p_vs_30:.4f}")
    
    # Signal detection
    effect_size = (mean_ot - random_baseline) / (std_ot + 1e-8)
    signal_detected = (mean_ot > random_baseline + 0.05) and (p_vs_random < 0.05) and (effect_size > 0.5)
    
    if signal_detected:
        print(f"\nSIGNAL_DETECTED: Significant feature reuse ({mean_ot:.1%})")
    else:
        print(f"\nNO_SIGNAL: No significant feature reuse")
    
    # Final results
    final_results = {
        'mean': mean_ot,
        'std': std_ot,
        'per_seed_results': results,
        'p_values': {
            'vs_random': float(p_vs_random),
            'vs_30pct': float(p_vs_30)
        },
        'ablation_results': ablation_results,
        'convergence_status': [r['converged'] for r in results],
        'baselines': {
            'random': random_baseline,
            'same_architecture': same_arch_baseline,
            'cka_mean': mean_cka
        },
        'signal_detected': signal_detected,
        'runtime_seconds': time.time() - start_time
    }
    
    print(f"\nRESULTS: {json.dumps(final_results)}")


if __name__ == "__main__":
    start_time = time.time()
    
    try:
        run_experiment()
    except Exception as e:
        print(f"ERROR: {e}")
        # Minimal valid output
        error_results = {
            'mean': 0.0,
            'std': 0.0,
            'per_seed_results': [],
            'p_values': {'vs_random': 1.0, 'vs_30pct': 1.0},
            'ablation_results': {},
            'convergence_status': [],
            'baselines': {'random': 0.0},
            'signal_detected': False,
            'runtime_seconds': time.time() - start_time,
            'error': str(e)
        }
        print(f"\nRESULTS: {json.dumps(error_results)}")