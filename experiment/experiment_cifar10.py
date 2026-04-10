#!/usr/bin/env python3
"""Run 082: Activation Transport on CIFAR-10 — Expanded from Run 047.

Tests whether the AT metric finding (CNNs and ViTs learn different features)
holds on a harder dataset with proper architectures.

Architectures: ResNet-18, ViT-Small (custom), MLP-Mixer (custom)
Dataset: CIFAR-10
Seeds: 10
Metrics: AT score (Sinkhorn OT), CKA, feature correlation
Comparisons: All pairs (CNN-ViT, CNN-Mixer, ViT-Mixer, CNN-CNN, ViT-ViT)
"""

# pip install torch torchvision scipy pot

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import numpy as np
import json
import random
import time
import warnings
from scipy.stats import ttest_ind, spearmanr
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() 
                       else 'mps' if torch.backends.mps.is_available() 
                       else 'cpu')
print(f"Device: {device}")

N_SEEDS = 10
EPOCHS = 50
PATIENCE = 8
LR = 0.001
BATCH_SIZE = 128
N_CLASSES = 10
AT_BINS = 25

# ─── Model Definitions ───

class ResNet18Features(nn.Module):
    """ResNet-18 with feature extraction hooks."""
    def __init__(self, num_classes=10):
        super().__init__()
        # Build from scratch (no pretrained) for fair comparison
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = []
        # First block (may downsample)
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        layers.append(BasicBlock(in_ch, out_ch, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)
    
    def get_features(self, x, layer='final'):
        """Extract features at various layers."""
        x = F.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        
        if layer == 'layer1': return f1
        if layer == 'layer2': return f2
        if layer == 'layer3': return f3
        if layer == 'layer4': return f4
        # final = pooled
        return self.avgpool(f4).flatten(1)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample: identity = self.downsample(x)
        return F.relu(out + identity)


class SmallViT(nn.Module):
    """Small Vision Transformer for CIFAR-10 (trained from scratch)."""
    def __init__(self, img_size=32, patch_size=4, dim=192, depth=6, heads=6, 
                 mlp_dim=384, num_classes=10):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2  # 64 patches
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, n_patches, dim]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0])  # CLS token
        return self.head(x)
    
    def get_features(self, x, layer='final'):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if layer == f'block{i}':
                return x[:, 0]  # CLS token at this block
        
        if layer == 'final':
            return self.norm(x[:, 0])
        return self.norm(x[:, 0])


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
    
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MLPMixer(nn.Module):
    """MLP-Mixer for CIFAR-10."""
    def __init__(self, img_size=32, patch_size=4, dim=192, depth=6, 
                 token_mlp_dim=96, channel_mlp_dim=384, num_classes=10):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        
        self.blocks = nn.ModuleList([
            MixerBlock(n_patches, dim, token_mlp_dim, channel_mlp_dim) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, patches, dim]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)
    
    def get_features(self, x, layer='final'):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if layer == f'block{i}':
                return x.mean(dim=1)
        return self.norm(x.mean(dim=1))


class MixerBlock(nn.Module):
    def __init__(self, n_patches, dim, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = nn.Sequential(
            nn.Linear(n_patches, token_mlp_dim),
            nn.GELU(),
            nn.Linear(token_mlp_dim, n_patches)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, channel_mlp_dim),
            nn.GELU(),
            nn.Linear(channel_mlp_dim, dim)
        )
    
    def forward(self, x):
        # Token mixing
        y = self.norm1(x).transpose(1, 2)  # [B, dim, patches]
        y = self.token_mix(y).transpose(1, 2)
        x = x + y
        # Channel mixing
        x = x + self.channel_mix(self.norm2(x))
        return x


# ─── Activation Transport (AT) Metric ───

def compute_at_score(feats1, feats2, n_bins=AT_BINS, reg=0.1):
    """Compute Activation Transport score between two feature matrices.
    
    feats1: [N, d1] numpy array
    feats2: [N, d2] numpy array
    Returns: AT score (lower = more similar features)
    """
    d1, d2 = feats1.shape[1], feats2.shape[1]
    
    # Normalize features
    feats1 = (feats1 - feats1.mean(0)) / (feats1.std(0) + 1e-8)
    feats2 = (feats2 - feats2.mean(0)) / (feats2.std(0) + 1e-8)
    
    # Compute per-channel histograms
    def channel_histograms(feats, n_bins):
        hists = []
        for c in range(feats.shape[1]):
            h, _ = np.histogram(feats[:, c], bins=n_bins, range=(-3, 3))
            h = h.astype(np.float64) + 1e-10
            h /= h.sum()
            hists.append(h)
        return np.array(hists)
    
    hist1 = channel_histograms(feats1, n_bins)
    hist2 = channel_histograms(feats2, n_bins)
    
    # Compute cost matrix using L1 Wasserstein between histograms
    bins = np.linspace(-3, 3, n_bins)
    cost = np.zeros((d1, d2), dtype=np.float64)
    for i in range(d1):
        cdf_i = np.cumsum(hist1[i])
        for j in range(d2):
            cdf_j = np.cumsum(hist2[j])
            cost[i, j] = np.sum(np.abs(cdf_i - cdf_j)) * (bins[1] - bins[0])
    
    # Pad to square
    d_max = max(d1, d2)
    if d1 < d_max:
        cost = np.pad(cost, ((0, d_max - d1), (0, 0)), constant_values=cost.max() * 10)
    elif d2 < d_max:
        cost = np.pad(cost, ((0, 0), (0, d_max - d2)), constant_values=cost.max() * 10)
    
    # Hungarian assignment (optimal matching)
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Average matched cost (normalized)
    valid_cost = [cost[i, j] for i, j in zip(row_ind, col_ind) if i < d1 and j < d2]
    at_score = np.mean(valid_cost) if valid_cost else 0.0
    
    return float(at_score)


def compute_cka(feats1, feats2):
    """Linear CKA between two feature matrices [N, d]."""
    feats1 = feats1 - feats1.mean(0)
    feats2 = feats2 - feats2.mean(0)
    
    hsic_xy = np.linalg.norm(feats1.T @ feats2, 'fro') ** 2
    hsic_xx = np.linalg.norm(feats1.T @ feats1, 'fro') ** 2
    hsic_yy = np.linalg.norm(feats2.T @ feats2, 'fro') ** 2
    
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


# ─── Training ───

def get_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_set = torchvision.datasets.CIFAR10('./data', train=True, download=True, 
                                              transform=transform_train)
    test_set = torchvision.datasets.CIFAR10('./data', train=False, download=True,
                                             transform=transform_test)
    
    # Split train into train/val
    n_val = 5000
    n_train = len(train_set) - n_val
    train_sub = Subset(train_set, range(n_train))
    val_sub = Subset(train_set, range(n_train, n_train + n_val))
    
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    no_improve = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        
        scheduler.step()
        train_acc = correct / total
        
        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    print(f"  Best val_acc: {best_val_acc:.4f}")
    return model, best_val_acc


def extract_features(model, data_loader, layer='final'):
    """Extract features from a model for all test samples."""
    model.eval()
    all_feats = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            f = model.get_features(x, layer=layer)
            # Flatten spatial dims if present
            if len(f.shape) == 4:  # [B, C, H, W]
                f = f.mean(dim=(2, 3))  # Global avg pool -> [B, C]
            all_feats.append(f.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


# ─── Main Experiment ───

def run_seed(seed, train_loader, val_loader, test_loader):
    """Run one seed: train all architectures, compute all AT/CKA pairs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n{'='*50}")
    print(f"Seed {seed}")
    print(f"{'='*50}")
    
    # Train each architecture
    models = {}
    accuracies = {}
    
    print("\n  Training ResNet-18...")
    resnet = ResNet18Features(num_classes=N_CLASSES)
    resnet, acc = train_model(resnet, train_loader, val_loader)
    models['resnet'] = resnet
    accuracies['resnet'] = acc
    
    print("\n  Training ViT...")
    vit = SmallViT(num_classes=N_CLASSES)
    vit, acc = train_model(vit, train_loader, val_loader)
    models['vit'] = vit
    accuracies['vit'] = acc
    
    print("\n  Training MLP-Mixer...")
    mixer = MLPMixer(num_classes=N_CLASSES)
    mixer, acc = train_model(mixer, train_loader, val_loader)
    models['mixer'] = mixer
    accuracies['mixer'] = acc
    
    # Train a second ResNet for same-architecture baseline
    print("\n  Training ResNet-18 (baseline)...")
    torch.manual_seed(seed + 1000)
    resnet2 = ResNet18Features(num_classes=N_CLASSES)
    resnet2, acc2 = train_model(resnet2, train_loader, val_loader)
    models['resnet2'] = resnet2
    accuracies['resnet2'] = acc2
    
    # Extract features (final layer)
    print("\n  Extracting features...")
    features = {}
    for name, model in models.items():
        features[name] = extract_features(model, test_loader, layer='final')
        print(f"    {name}: shape {features[name].shape}")
    
    # Compute all pairwise AT and CKA scores
    pairs = [
        ('resnet', 'vit'),      # Cross-arch
        ('resnet', 'mixer'),    # Cross-arch
        ('vit', 'mixer'),       # Cross-arch
        ('resnet', 'resnet2'),  # Same-arch baseline
    ]
    
    at_scores = {}
    cka_scores = {}
    for n1, n2 in pairs:
        key = f"{n1}_vs_{n2}"
        at_scores[key] = compute_at_score(features[n1], features[n2])
        cka_scores[key] = compute_cka(features[n1], features[n2])
        print(f"    {key}: AT={at_scores[key]:.4f}, CKA={cka_scores[key]:.4f}")
    
    # Layer-wise AT analysis (ResNet vs ViT only, to keep runtime sane)
    print("\n  Layer-wise AT analysis (ResNet vs ViT)...")
    layer_at = {}
    resnet_layers = ['layer1', 'layer2', 'layer3', 'layer4', 'final']
    vit_layers = ['block0', 'block2', 'block4', 'block5', 'final']
    
    for rl, vl in zip(resnet_layers, vit_layers):
        rf = extract_features(models['resnet'], test_loader, layer=rl)
        vf = extract_features(models['vit'], test_loader, layer=vl)
        at = compute_at_score(rf, vf)
        layer_at[f"resnet_{rl}_vs_vit_{vl}"] = at
        print(f"    {rl} vs {vl}: AT={at:.4f}")
    
    return {
        'seed': seed,
        'accuracies': {k: float(v) for k, v in accuracies.items()},
        'at_scores': at_scores,
        'cka_scores': cka_scores,
        'layer_at': layer_at,
    }


def main():
    print("=" * 60)
    print("Run 082: Activation Transport on CIFAR-10")
    print("=" * 60)
    
    train_loader, val_loader, test_loader = get_data()
    
    all_results = []
    start = time.time()
    
    for seed in range(42, 42 + N_SEEDS):
        result = run_seed(seed, train_loader, val_loader, test_loader)
        all_results.append(result)
        
        # Print running summary after each seed
        elapsed = time.time() - start
        print(f"\n  [Seed {seed} done | {elapsed/60:.1f}min elapsed]")
        
        # SANITY CHECK after first seed
        if seed == 42:
            at = result['at_scores']
            cross_at = np.mean([at['resnet_vs_vit'], at['resnet_vs_mixer'], at['vit_vs_mixer']])
            same_at = at['resnet_vs_resnet2']
            print(f"\n  SANITY CHECK (seed 42):")
            print(f"    Cross-arch avg AT: {cross_at:.4f}")
            print(f"    Same-arch AT:      {same_at:.4f}")
            if cross_at == 0.0 or same_at == 0.0:
                print("  SANITY_ABORT: AT scores are zero — metric is broken")
                return
            print(f"    Ratio: {cross_at/same_at:.2f}x")
            print("  Sanity passed — continuing remaining seeds...")
    
    # ─── Aggregate Results ───
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)
    
    # Collect per-pair scores across seeds
    pair_keys = ['resnet_vs_vit', 'resnet_vs_mixer', 'vit_vs_mixer', 'resnet_vs_resnet2']
    
    at_by_pair = {k: [r['at_scores'][k] for r in all_results] for k in pair_keys}
    cka_by_pair = {k: [r['cka_scores'][k] for r in all_results] for k in pair_keys}
    
    print("\nActivation Transport scores (mean ± std):")
    for k in pair_keys:
        vals = at_by_pair[k]
        print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    print("\nCKA scores (mean ± std):")
    for k in pair_keys:
        vals = cka_by_pair[k]
        print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    # Statistical tests: cross-arch vs same-arch
    cross_arch_at = [np.mean([r['at_scores']['resnet_vs_vit'], 
                               r['at_scores']['resnet_vs_mixer'],
                               r['at_scores']['vit_vs_mixer']]) for r in all_results]
    same_arch_at = [r['at_scores']['resnet_vs_resnet2'] for r in all_results]
    
    t_stat, p_val = ttest_ind(cross_arch_at, same_arch_at)
    print(f"\nCross-arch AT vs Same-arch AT:")
    print(f"  Cross: {np.mean(cross_arch_at):.4f} ± {np.std(cross_arch_at):.4f}")
    print(f"  Same:  {np.mean(same_arch_at):.4f} ± {np.std(same_arch_at):.4f}")
    print(f"  t={t_stat:.3f}, p={p_val:.6f}")
    
    # Layer-wise analysis
    layer_keys = [k for k in all_results[0]['layer_at'].keys()]
    print("\nLayer-wise AT (ResNet vs ViT, mean ± std):")
    for lk in layer_keys:
        vals = [r['layer_at'][lk] for r in all_results]
        print(f"  {lk}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    # Accuracies
    for arch in ['resnet', 'vit', 'mixer']:
        accs = [r['accuracies'][arch] for r in all_results]
        print(f"\n{arch} accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    
    total_time = time.time() - start
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    
    # Final JSON
    final = {
        'experiment': 'run_082_at_cifar10',
        'n_seeds': N_SEEDS,
        'at_scores': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} 
                      for k, v in at_by_pair.items()},
        'cka_scores': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} 
                       for k, v in cka_by_pair.items()},
        'cross_vs_same_at': {
            'cross_mean': float(np.mean(cross_arch_at)),
            'same_mean': float(np.mean(same_arch_at)),
            't_stat': float(t_stat),
            'p_value': float(p_val),
        },
        'layer_wise_at': {lk: {'mean': float(np.mean([r['layer_at'][lk] for r in all_results])),
                                'std': float(np.std([r['layer_at'][lk] for r in all_results]))}
                          for lk in layer_keys},
        'accuracies': {arch: {'mean': float(np.mean([r['accuracies'][arch] for r in all_results])),
                              'std': float(np.std([r['accuracies'][arch] for r in all_results]))}
                       for arch in ['resnet', 'vit', 'mixer']},
        'per_seed': all_results,
        'total_time_minutes': total_time / 60,
    }
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(final, f, indent=2)
    
    print(f"\nRESULTS: {json.dumps({k: v for k, v in final.items() if k != 'per_seed'})}")


if __name__ == "__main__":
    main()
