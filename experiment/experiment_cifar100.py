#!/usr/bin/env python3
"""Run 083: Activation Transport on CIFAR-100 — does the AT finding hold with more classes?

Same methodology as run 082 but on CIFAR-100 (100 classes).
Hypothesis: With more classes, architectures may be forced to share features
(or diverge even more due to harder task).
"""

# pip install torch torchvision scipy

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
from scipy.stats import ttest_ind
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() 
                       else 'mps' if torch.backends.mps.is_available() 
                       else 'cpu')
print(f"Device: {device}")

N_SEEDS = 10
EPOCHS = 80  # More epochs for harder task
PATIENCE = 10
LR = 0.001
BATCH_SIZE = 128
N_CLASSES = 100
AT_BINS = 25

# Import architectures — same as run 082 but with 100 classes
# (Duplicating here for standalone execution)

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


class ResNet18Features(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
        layers.append(BasicBlock(in_ch, out_ch, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.avgpool(x).flatten(1))
    
    def get_features(self, x, layer='final'):
        x = F.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(x); f2 = self.layer2(f1); f3 = self.layer3(f2); f4 = self.layer4(f3)
        if layer == 'layer1': return f1
        if layer == 'layer2': return f2
        if layer == 'layer3': return f3
        if layer == 'layer4': return f4
        return self.avgpool(f4).flatten(1)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))
    
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class SmallViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, dim=256, depth=8, heads=8, 
                 mlp_dim=512, num_classes=100):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm(x[:, 0]))
    
    def get_features(self, x, layer='final'):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if layer == f'block{i}': return x[:, 0]
        return self.norm(x[:, 0])


class MixerBlock(nn.Module):
    def __init__(self, n_patches, dim, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = nn.Sequential(nn.Linear(n_patches, token_mlp_dim), nn.GELU(), nn.Linear(token_mlp_dim, n_patches))
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(nn.Linear(dim, channel_mlp_dim), nn.GELU(), nn.Linear(channel_mlp_dim, dim))
    
    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        x = x + self.token_mix(y).transpose(1, 2)
        return x + self.channel_mix(self.norm2(x))


class MLPMixer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, dim=256, depth=8,
                 token_mlp_dim=128, channel_mlp_dim=512, num_classes=100):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([MixerBlock(n_patches, dim, token_mlp_dim, channel_mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm(x.mean(dim=1)))
    
    def get_features(self, x, layer='final'):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if layer == f'block{i}': return x.mean(dim=1)
        return self.norm(x.mean(dim=1))


# ─── Metrics ───

def compute_at_score(feats1, feats2, n_bins=AT_BINS):
    d1, d2 = feats1.shape[1], feats2.shape[1]
    feats1 = (feats1 - feats1.mean(0)) / (feats1.std(0) + 1e-8)
    feats2 = (feats2 - feats2.mean(0)) / (feats2.std(0) + 1e-8)
    
    def hist(feats, n):
        return np.array([np.histogram(feats[:, c], bins=n, range=(-3,3))[0].astype(np.float64) + 1e-10 
                        for c in range(feats.shape[1])]) 
    
    h1 = hist(feats1, n_bins); h2 = hist(feats2, n_bins)
    h1 /= h1.sum(1, keepdims=True); h2 /= h2.sum(1, keepdims=True)
    
    dx = 6.0 / n_bins
    cost = np.zeros((d1, d2))
    for i in range(d1):
        c1 = np.cumsum(h1[i])
        for j in range(d2):
            cost[i,j] = np.sum(np.abs(c1 - np.cumsum(h2[j]))) * dx
    
    d_max = max(d1, d2)
    if d1 < d_max: cost = np.pad(cost, ((0,d_max-d1),(0,0)), constant_values=cost.max()*10)
    elif d2 < d_max: cost = np.pad(cost, ((0,0),(0,d_max-d2)), constant_values=cost.max()*10)
    
    ri, ci = linear_sum_assignment(cost)
    valid = [cost[i,j] for i,j in zip(ri,ci) if i < d1 and j < d2]
    return float(np.mean(valid)) if valid else 0.0


def compute_cka(f1, f2):
    f1 = f1 - f1.mean(0); f2 = f2 - f2.mean(0)
    xy = np.linalg.norm(f1.T @ f2, 'fro')**2
    xx = np.linalg.norm(f1.T @ f1, 'fro')**2
    yy = np.linalg.norm(f2.T @ f2, 'fro')**2
    return float(xy / (np.sqrt(xx * yy) + 1e-10))


# ─── Training ───

def get_data():
    t_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    
    train = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=t_train)
    test = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=t_test)
    
    n_val = 5000
    return (DataLoader(Subset(train, range(len(train)-n_val)), batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
            DataLoader(Subset(train, range(len(train)-n_val, len(train))), batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
            DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2))


def train_model(model, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc = 0; no_imp = 0; best_state = None
    
    for ep in range(epochs):
        model.train()
        c = t = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); out = model(x); F.cross_entropy(out, y).backward(); opt.step()
            c += (out.argmax(1)==y).sum().item(); t += y.size(0)
        sched.step()
        
        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vc += (model(x).argmax(1)==y).sum().item(); vt += y.size(0)
        va = vc/vt
        
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1}: train={c/t:.4f}, val={va:.4f}")
        
        if va > best_acc:
            best_acc = va; best_state = {k:v.clone() for k,v in model.state_dict().items()}; no_imp = 0
        else: no_imp += 1
        if no_imp >= patience:
            print(f"  Early stop at {ep+1}"); break
    
    if best_state: model.load_state_dict(best_state)
    print(f"  Best val: {best_acc:.4f}")
    return model, best_acc


def extract_features(model, loader, layer='final'):
    model.eval()
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            f = model.get_features(x.to(device), layer=layer)
            if len(f.shape) == 4: f = f.mean(dim=(2,3))
            feats.append(f.cpu().numpy())
    return np.concatenate(feats)


def run_seed(seed, train_loader, val_loader, test_loader):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
    print(f"\n{'='*50}\nSeed {seed}\n{'='*50}")
    
    models = {}; accs = {}
    for name, cls in [('resnet', ResNet18Features), ('vit', SmallViT), ('mixer', MLPMixer)]:
        print(f"\n  Training {name}...")
        m, a = train_model(cls(num_classes=N_CLASSES), train_loader, val_loader)
        models[name] = m; accs[name] = a
    
    # Baseline
    print(f"\n  Training resnet2 (baseline)...")
    torch.manual_seed(seed + 1000)
    m2, a2 = train_model(ResNet18Features(num_classes=N_CLASSES), train_loader, val_loader)
    models['resnet2'] = m2; accs['resnet2'] = a2
    
    features = {n: extract_features(m, test_loader) for n, m in models.items()}
    
    pairs = [('resnet','vit'), ('resnet','mixer'), ('vit','mixer'), ('resnet','resnet2')]
    at = {f"{a}_vs_{b}": compute_at_score(features[a], features[b]) for a,b in pairs}
    cka = {f"{a}_vs_{b}": compute_cka(features[a], features[b]) for a,b in pairs}
    
    for k in at: print(f"    {k}: AT={at[k]:.4f}, CKA={cka[k]:.4f}")
    
    return {'seed': seed, 'accuracies': {k:float(v) for k,v in accs.items()}, 
            'at_scores': at, 'cka_scores': cka}


def main():
    print("="*60 + "\nRun 083: Activation Transport on CIFAR-100\n" + "="*60)
    train_loader, val_loader, test_loader = get_data()
    
    results = []; start = time.time()
    for seed in range(42, 42 + N_SEEDS):
        results.append(run_seed(seed, train_loader, val_loader, test_loader))
        
        if seed == 42:  # Sanity check
            at = results[0]['at_scores']
            cross = np.mean([at['resnet_vs_vit'], at['resnet_vs_mixer'], at['vit_vs_mixer']])
            same = at['resnet_vs_resnet2']
            print(f"\n  SANITY: cross={cross:.4f}, same={same:.4f}, ratio={cross/same:.2f}x")
            if cross == 0 or same == 0:
                print("  SANITY_ABORT"); return
    
    # Aggregate
    pairs = ['resnet_vs_vit', 'resnet_vs_mixer', 'vit_vs_mixer', 'resnet_vs_resnet2']
    at_agg = {k: [r['at_scores'][k] for r in results] for k in pairs}
    
    cross = [np.mean([r['at_scores'][p] for p in pairs[:3]]) for r in results]
    same = [r['at_scores']['resnet_vs_resnet2'] for r in results]
    t, p = ttest_ind(cross, same)
    
    print(f"\n{'='*60}\nAGGREGATED (CIFAR-100)")
    for k in pairs: print(f"  {k}: {np.mean(at_agg[k]):.4f} ± {np.std(at_agg[k]):.4f}")
    print(f"\nCross vs Same: t={t:.3f}, p={p:.6f}")
    
    final = {'experiment': 'run_083_at_cifar100', 'n_seeds': N_SEEDS,
             'at_scores': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k,v in at_agg.items()},
             'cross_vs_same': {'t': float(t), 'p': float(p), 
                               'cross_mean': float(np.mean(cross)), 'same_mean': float(np.mean(same))},
             'per_seed': results, 'time_min': (time.time()-start)/60}
    
    with open('results.json', 'w') as f: json.dump(final, f, indent=2)
    print(f"\nRESULTS: {json.dumps({k:v for k,v in final.items() if k != 'per_seed'})}")

if __name__ == "__main__":
    main()
