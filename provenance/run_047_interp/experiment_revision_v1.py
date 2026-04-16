# pip install torch torchvision scipy scikit-learn pot matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision import transforms
import numpy as np
import json
from scipy.stats import ttest_rel, ttest_1samp, ttest_ind, bootstrap
import ot  # Python Optimal Transport
import random
import time
import warnings
import sys
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Model Definitions (More realistic) ===
class ResNetBlock(nn.Module):
    """Basic ResNet block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MiniResNet(nn.Module):
    """Smaller ResNet for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = nn.Sequential(
            ResNetBlock(64, 64),
            ResNetBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(128, 256, stride=2),
            ResNetBlock(256, 256)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def get_features(self, x, layer='layer3'):
        """Extract features from specified layer"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling to get feature vector
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x  # [B, 256]


class MiniViT(nn.Module):
    """Smaller ViT for CIFAR-10"""
    def __init__(self, img_size=32, patch_size=4, num_classes=10, 
                 dim=192, depth=6, heads=3):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, 
                                     dim_feedforward=dim*4, batch_first=True),
            num_layers=depth
        )
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        B = x.shape[0]
        
        # Create patches
        x = self.create_patches(x)
        x = self.patch_to_embedding(x)
        
        # Add cls token and positional embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        
        # Transformer
        x = self.transformer(x)
        
        # Classification
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
    
    def create_patches(self, x):
        """Convert image to patches"""
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        
        h, w = H // self.patch_size, W // self.patch_size
        x = x.reshape(B, C, h, self.patch_size, w, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, -1)
        return x
    
    def get_features(self, x):
        """Extract cls token features after transformer"""
        B = x.shape[0]
        
        x = self.create_patches(x)
        x = self.patch_to_embedding(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        
        x = self.transformer(x)
        
        # Return cls token features
        return x[:, 0]  # [B, 192]


# === Fixed Optimal Transport Implementation ===
def compute_optimal_transport_similarity(features1, features2, metric='cosine', reg=0.1):
    """
    Compute similarity between feature sets using optimal transport.
    Returns value in [0, 1] where 1 means identical features.
    """
    # Ensure numpy arrays
    if torch.is_tensor(features1):
        features1 = features1.cpu().numpy()
    if torch.is_tensor(features2):
        features2 = features2.cpu().numpy()
    
    n1, d1 = features1.shape
    n2, d2 = features2.shape
    
    # For different dimensions, project to common space
    if d1 != d2:
        common_dim = min(d1, d2)
        # Use PCA to project to common dimensionality
        from sklearn.decomposition import PCA
        if d1 > common_dim:
            pca1 = PCA(n_components=common_dim, random_state=42)
            features1 = pca1.fit_transform(features1)
        if d2 > common_dim:
            pca2 = PCA(n_components=common_dim, random_state=42)
            features2 = pca2.fit_transform(features2)
    
    # L2 normalize
    features1 = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
    features2 = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)
    
    # Sample for computational efficiency
    n_samples = min(200, n1, n2)
    idx1 = np.random.choice(n1, n_samples, replace=False)
    idx2 = np.random.choice(n2, n_samples, replace=False)
    
    f1 = features1[idx1]
    f2 = features2[idx2]
    
    if metric == 'cosine':
        # Cost matrix: 1 - cosine_similarity
        cost_matrix = 1.0 - np.dot(f1, f2.T)
    else:
        # Euclidean distance
        cost_matrix = np.sum(f1**2, axis=1)[:, None] + np.sum(f2**2, axis=1)[None, :] - 2*np.dot(f1, f2.T)
        cost_matrix = np.sqrt(np.maximum(cost_matrix, 0))
        cost_matrix /= cost_matrix.max() + 1e-8  # Normalize to [0, 1]
    
    # Uniform marginals
    a = np.ones(n_samples) / n_samples
    b = np.ones(n_samples) / n_samples
    
    try:
        # Compute optimal transport plan
        transport_plan = ot.sinkhorn(a, b, cost_matrix, reg=reg, numItermax=100)
        
        # Compute similarity as 1 - normalized transport cost
        transport_cost = np.sum(transport_plan * cost_matrix)
        max_cost = np.mean(cost_matrix)  # Expected cost for random matching
        
        # Normalize to [0, 1] where 1 = identical, 0 = maximally different
        similarity = 1.0 - (transport_cost / max_cost)
        similarity = np.clip(similarity, 0, 1)
        
        return float(similarity)
        
    except Exception as e:
        print(f"OT computation failed: {e}")
        return 0.0


def compute_cka(features1, features2):
    """Compute CKA similarity between feature sets"""
    if torch.is_tensor(features1):
        features1 = features1.cpu().numpy()
    if torch.is_tensor(features2):
        features2 = features2.cpu().numpy()
    
    n = min(len(features1), len(features2))
    f1 = features1[:n]
    f2 = features2[:n]
    
    # Center
    f1 = f1 - np.mean(f1, axis=0, keepdims=True)
    f2 = f2 - np.mean(f2, axis=0, keepdims=True)
    
    # Compute Gram matrices
    K1 = np.dot(f1, f1.T)
    K2 = np.dot(f2, f2.T)
    
    # Center Gram matrices
    K1_centered = K1 - K1.mean(axis=0, keepdims=True) - K1.mean(axis=1, keepdims=True) + K1.mean()
    K2_centered = K2 - K2.mean(axis=0, keepdims=True) - K2.mean(axis=1, keepdims=True) + K2.mean()
    
    # Compute HSIC
    hsic = np.sum(K1_centered * K2_centered) / ((n-1)**2)
    var1 = np.sum(K1_centered**2) / ((n-1)**2)
    var2 = np.sum(K2_centered**2) / ((n-1)**2)
    
    if var1 * var2 > 0:
        cka = hsic / np.sqrt(var1 * var2)
        return float(np.clip(cka, 0, 1))
    return 0.0


def validate_metrics():
    """Comprehensive validation of similarity metrics"""
    print("Validating metrics...")
    
    # Test 1: Identical features should have high similarity
    np.random.seed(42)
    feat1 = np.random.randn(100, 64)
    
    ot_sim = compute_optimal_transport_similarity(feat1, feat1)
    cka_sim = compute_cka(feat1, feat1)
    print(f"Identical features - OT: {ot_sim:.3f}, CKA: {cka_sim:.3f}")
    assert ot_sim > 0.95, f"OT similarity for identical features should be >0.95, got {ot_sim}"
    assert cka_sim > 0.99, f"CKA for identical features should be >0.99, got {cka_sim}"
    
    # Test 2: Random features should have low similarity
    feat2 = np.random.randn(100, 64)
    ot_sim = compute_optimal_transport_similarity(feat1, feat2)
    cka_sim = compute_cka(feat1, feat2)
    print(f"Random features - OT: {ot_sim:.3f}, CKA: {cka_sim:.3f}")
    assert ot_sim < 0.6, f"OT similarity for random features should be <0.6, got {ot_sim}"
    
    # Test 3: Slightly perturbed features should have moderate similarity
    feat3 = feat1 + 0.5 * np.random.randn(100, 64)
    ot_sim = compute_optimal_transport_similarity(feat1, feat3)
    print(f"Perturbed features - OT: {ot_sim:.3f}")
    assert 0.6 < ot_sim < 0.9, f"OT similarity for perturbed features should be in (0.6, 0.9), got {ot_sim}"
    
    # Test 4: Different dimensions should work
    feat4 = np.random.randn(100, 128)
    ot_sim = compute_optimal_transport_similarity(feat1, feat4)
    print(f"Different dimensions - OT: {ot_sim:.3f}")
    
    print("✓ All metric validations passed!\n")


# === Comprehensive Baselines ===
def compute_comprehensive_baselines(model_class1, model_class2, train_loader, val_loader, 
                                  feat_loader, device):
    """Compute multiple baselines for proper comparison"""
    baselines = {}
    
    print("\n=== COMPUTING COMPREHENSIVE BASELINES ===")
    
    # 1. Random untrained networks
    print("\n1. Random untrained networks...")
    rand_scores_ot = []
    rand_scores_cka = []
    
    for i in range(5):
        torch.manual_seed(2000 + i)
        m1 = model_class1().to(device)
        m2 = model_class2().to(device)
        
        f1 = extract_features(m1, feat_loader, device, max_samples=200)
        f2 = extract_features(m2, feat_loader, device, max_samples=200)
        
        rand_scores_ot.append(compute_optimal_transport_similarity(f1, f2))
        rand_scores_cka.append(compute_cka(f1, f2))
    
    baselines['random_untrained'] = {
        'ot_mean': float(np.mean(rand_scores_ot)),
        'ot_std': float(np.std(rand_scores_ot)),
        'cka_mean': float(np.mean(rand_scores_cka)),
        'cka_std': float(np.std(rand_scores_cka))
    }
    print(f"Random untrained - OT: {baselines['random_untrained']['ot_mean']:.3f} ± {baselines['random_untrained']['ot_std']:.3f}")
    
    # 2. Same architecture, different seeds
    print("\n2. Same architecture, different training runs...")
    same_arch_ot = []
    same_arch_cka = []
    
    for i in range(3):
        torch.manual_seed(3000 + i)
        m1 = model_class1().to(device)
        m2 = model_class1().to(device)
        
        # Quick training
        m1, _, _ = train_model(m1, train_loader, val_loader, device, max_epochs=10)
        
        torch.manual_seed(3100 + i)
        m2, _, _ = train_model(m2, train_loader, val_loader, device, max_epochs=10)
        
        f1 = extract_features(m1, feat_loader, device, max_samples=200)
        f2 = extract_features(m2, feat_loader, device, max_samples=200)
        
        same_arch_ot.append(compute_optimal_transport_similarity(f1, f2))
        same_arch_cka.append(compute_cka(f1, f2))
    
    baselines['same_architecture'] = {
        'ot_mean': float(np.mean(same_arch_ot)),
        'ot_std': float(np.std(same_arch_ot)),
        'cka_mean': float(np.mean(same_arch_cka))
    }
    print(f"Same architecture - OT: {baselines['same_architecture']['ot_mean']:.3f} ± {baselines['same_architecture']['ot_std']:.3f}")
    
    # 3. Permuted weights baseline
    print("\n3. Permuted weights (breaks function while keeping statistics)...")
    torch.manual_seed(4000)
    m1 = model_class1().to(device)
    m1, _, _ = train_model(m1, train_loader, val_loader, device, max_epochs=10)
    
    # Create permuted version
    m1_perm = model_class1().to(device)
    m1_perm.load_state_dict(m1.state_dict())
    
    # Permute conv weights
    for name, param in m1_perm.named_parameters():
        if 'conv' in name and len(param.shape) == 4:
            perm = torch.randperm(param.shape[0])
            param.data = param.data[perm]
    
    f1 = extract_features(m1, feat_loader, device, max_samples=200)
    f1_perm = extract_features(m1_perm, feat_loader, device, max_samples=200)
    
    baselines['permuted_weights'] = {
        'ot': float(compute_optimal_transport_similarity(f1, f1_perm)),
        'cka': float(compute_cka(f1, f1_perm))
    }
    print(f"Permuted weights - OT: {baselines['permuted_weights']['ot']:.3f}")
    
    return baselines


def extract_features(model, dataloader, device, max_samples=500):
    """Extract features from model"""
    model.eval()
    features_list = []
    
    with torch.no_grad():
        count = 0
        for images, _ in dataloader:
            if count >= max_samples:
                break
            images = images.to(device)
            feats = model.get_features(images)
            features_list.append(feats.cpu())
            count += images.size(0)
    
    return torch.cat(features_list, dim=0)


# === Training ===
def train_model(model, train_loader, val_loader, device, max_epochs=30, patience=5):
    """Train model with early stopping"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    no_improve = 0
    
    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
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
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                if batch_idx > 40:
                    break
                    
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        train_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print("CONVERGED")
            return model, True, val_acc
    
    print("NOT_CONVERGED: Max epochs")
    return model, False, val_acc


# === Main Experiment ===
def run_experiment():
    """Run complete experiment with all baselines and proper statistics"""
    
    # Validate metrics first
    validate_metrics()
    
    # Load CIFAR-10 (more realistic than MNIST)
    print("Loading CIFAR-10 dataset...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Use subset for feasibility
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    dataset = Subset(full_train, range(10000))  # 10k samples
    
    # Storage for results
    all_results = []
    ot_scores = []
    cka_scores = []
    
    # Architecture pairs to test
    arch_pairs = [
        ('ResNet', 'ViT', MiniResNet, MiniViT),
        ('ResNet', 'ResNet', MiniResNet, MiniResNet),  # Same arch control
    ]
    
    # Run main experiments
    num_seeds = 10
    
    for arch1_name, arch2_name, arch1_class, arch2_class in arch_pairs:
        print(f"\n{'='*60}")
        print(f"Testing {arch1_name} vs {arch2_name}")
        print(f"{'='*60}")
        
        pair_ot_scores = []
        pair_cka_scores = []
        
        for seed in range(num_seeds):
            print(f"\nSeed {seed}...")
            
            # Set seeds
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            # Data splits
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_data, val_data = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=2)
            
            # Feature extraction subset
            feat_data = Subset(train_data, range(min(500, len(train_data))))
            feat_loader = DataLoader(feat_data, batch_size=50, shuffle=False)
            
            # Train models
            print(f"Training {arch1_name}...")
            model1 = arch1_class().to(device)
            model1, conv1, acc1 = train_model(model1, train_loader, val_loader, device)
            
            print(f"Training {arch2_name}...")
            model2 = arch2_class().to(device)
            model2, conv2, acc2 = train_model(model2, train_loader, val_loader, device)
            
            # Extract features
            print("Computing similarities...")
            feat1 = extract_features(model1, feat_loader, device)
            feat2 = extract_features(model2, feat_loader, device)
            
            ot_sim = compute_optimal_transport_similarity(feat1, feat2)
            cka_sim = compute_cka(feat1, feat2)
            
            print(f"OT: {ot_sim:.3f}, CKA: {cka_sim:.3f}")
            
            # Store results
            result = {
                'arch_pair': f"{arch1_name}-{arch2_name}",
                'seed': seed,
                'ot_similarity': ot_sim,
                'cka_similarity': cka_sim,
                'acc1': float(acc1),
                'acc2': float(acc2),
                'converged': conv1 and conv2
            }
            
            all_results.append(result)
            pair_ot_scores.append(ot_sim)
            pair_cka_scores.append(cka_sim)
            
            # Only do first 3 seeds if same architecture (control)
            if arch1_name == arch2_name and seed >= 2:
                break
        
        if arch1_name != arch2_name:  # Main experiment
            ot_scores = pair_ot_scores
            cka_scores = pair_cka_scores
    
    # Compute comprehensive baselines (only once)
    train_loader_baseline = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    val_loader_baseline = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=2)
    feat_loader_baseline = DataLoader(feat_data, batch_size=50, shuffle=False)
    
    baselines = compute_comprehensive_baselines(
        MiniResNet, MiniViT, 
        train_loader_baseline, val_loader_baseline, 
        feat_loader_baseline, device
    )
    
    # Statistical analysis with bootstrap
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    
    mean_ot = float(np.mean(ot_scores))
    std_ot = float(np.std(ot_scores))
    mean_cka = float(np.mean(cka_scores))
    std_cka = float(np.std(cka_scores))
    
    # Bootstrap confidence intervals
    def compute_mean(x):
        return np.mean(x)
    
    boot_result = bootstrap((ot_scores,), compute_mean, n_resamples=1000, 
                           confidence_level=0.95, random_state=42)
    ci_low, ci_high = boot_result.confidence_interval
    
    print(f"\nCNN-ViT similarity:")
    print(f"OT: {mean_ot:.3f} ± {std_ot:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])")
    print(f"CKA: {mean_cka:.3f} ± {std_cka:.3f}")
    
    # Statistical tests against baselines
    random_baseline = baselines['random_untrained']['ot_mean']
    _, p_vs_random = ttest_1samp(ot_scores, random_baseline, alternative='greater')
    _, p_vs_50pct = ttest_1samp(ot_scores, 0.5, alternative='greater')
    _, p_vs_30pct = ttest_1samp(ot_scores, 0.3, alternative='greater')
    
    print(f"\nStatistical tests:")
    print(f"p-value vs random ({random_baseline:.3f}): {p_vs_random:.4f}")
    print(f"p-value vs 50%: {p_vs_50pct:.4f}")
    print(f"p-value vs 30%: {p_vs_30pct:.4f}")
    
    # Effect size (Cohen's d)
    cohens_d = (mean_ot - random_baseline) / std_ot
    print(f"Effect size (Cohen's d): {cohens_d:.2f}")
    
    # Ablation: Effect of regularization parameter
    print(f"\n{'='*60}")
    print("ABLATION: OT Regularization Parameter")
    print(f"{'='*60}")
    
    reg_values = [0.001, 0.01, 0.1, 1.0]
    ablation_results = {}
    
    # Use first trained models for ablation
    torch.manual_seed(42)
    model1 = MiniResNet().to(device)
    model2 = MiniViT().to(device)
    model1, _, _ = train_model(model1, train_loader_baseline, val_loader_baseline, device, max_epochs=15)
    model2, _, _ = train_model(model2, train_loader_baseline, val_loader_baseline, device, max_epochs=15)
    
    feat1 = extract_features(model1, feat_loader_baseline, device, max_samples=300)
    feat2 = extract_features(model2, feat_loader_baseline, device, max_samples=300)
    
    for reg in reg_values:
        ot_sim = compute_optimal_transport_similarity(feat1, feat2, reg=reg)
        ablation_results[f'reg_{reg}'] = float(ot_sim)
        print(f"Reg={reg}: OT similarity = {ot_sim:.3f}")
    
    # Signal detection
    signal_detected = (mean_ot > random_baseline + 0.1) and (p_vs_random < 0.01) and (cohens_d > 0.5)
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    hypothesis_supported = mean_ot >= 0.3 and mean_ot <= 0.5 and p_vs_random < 0.05
    
    if signal_detected:
        print(f"SIGNAL_DETECTED: CNN-ViT show significant feature alignment ({mean_ot:.1%})")
    else:
        print(f"NO_SIGNAL: No significant feature alignment detected")
    
    if hypothesis_supported:
        print(f"✓ Hypothesis SUPPORTED: Feature reuse is {mean_ot:.1%} (expected 30-50%)")
    else:
        print(f"✗ Hypothesis NOT supported: Feature reuse is {mean_ot:.1%} (expected 30-50%)")
    
    print(f"\nKey findings:")
    print(f"- CNN-ViT similarity: {mean_ot:.1%} (p<{p_vs_random:.3f} vs random)")
    print(f"- Same architecture: {baselines['same_architecture']['ot_mean']:.1%}")
    print(f"- Random baseline: {random_baseline:.1%}")
    print(f"- Effect size: {cohens_d:.2f} (large)" if cohens_d > 0.8 else f"- Effect size: {cohens_d:.2f} (medium)")
    
    # Final JSON results
    final_results = {
        'mean': mean_ot,
        'std': std_ot,
        'confidence_interval': {'low': float(ci_low), 'high': float(ci_high)},
        'per_seed_results': [r for r in all_results if r['arch_pair'] == 'ResNet-ViT'],
        'p_values': {
            'vs_random': float(p_vs_random),
            'vs_30pct': float(p_vs_30pct),
            'vs_50pct': float(p_vs_50pct)
        },
        'ablation_results': ablation_results,
        'baselines': baselines,
        'convergence_status': [r['converged'] for r in all_results if r['arch_pair'] == 'ResNet-ViT'],
        'effect_size_cohens_d': float(cohens_d),
        'signal_detected': signal_detected,
        'hypothesis_supported': hypothesis_supported,
        'runtime_seconds': time.time() - start_time
    }
    
    print(f"\nRESULTS: {json.dumps(final_results)}")


if __name__ == "__main__":
    start_time = time.time()
    
    try:
        run_experiment()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Emergency results
        emergency_results = {
            'mean': 0.0,
            'std': 0.0,
            'per_seed_results': [],
            'p_values': {'vs_random': 1.0, 'vs_30pct': 1.0},
            'ablation_results': {},
            'convergence_status': [],
            'baselines': {'random_untrained': {'ot_mean': 0.0}},
            'signal_detected': False,
            'runtime_seconds': time.time() - start_time,
            'error': str(e)
        }
        print(f"\nRESULTS: {json.dumps(emergency_results)}")