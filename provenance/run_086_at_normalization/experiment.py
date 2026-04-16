#!/usr/bin/env python3
"""Run 086: How Normalization Layers Affect Cross-Architecture Feature Similarity.

Connects our BN expertise with the AT metric work.
Question: Does the choice of normalization (BN, LN, GN, none) affect 
how similar features are across architectures?

Hypothesis: Normalization creates "feature attractors" — networks with the same 
normalization type may converge to more similar features regardless of architecture.
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
from scipy.stats import ttest_ind, f_oneway
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() 
                       else 'mps' if torch.backends.mps.is_available() 
                       else 'cpu')
print(f"Device: {device}")

N_SEEDS = 8
EPOCHS = 50
PATIENCE = 8
LR = 0.001
BATCH_SIZE = 128
N_CLASSES = 10

NORM_TYPES = ['batchnorm', 'layernorm', 'groupnorm', 'none']

# ─── Flexible Architecture with Swappable Norm ───

def get_norm(norm_type, channels):
    if norm_type == 'batchnorm': return nn.BatchNorm2d(channels)
    elif norm_type == 'layernorm': return nn.GroupNorm(1, channels)  # LN for conv = GN with 1 group
    elif norm_type == 'groupnorm': return nn.GroupNorm(min(8, channels), channels)
    elif norm_type == 'none': return nn.Identity()
    raise ValueError(f"Unknown norm: {norm_type}")

def get_norm_1d(norm_type, features):
    if norm_type == 'batchnorm': return nn.BatchNorm1d(features)
    elif norm_type == 'layernorm': return nn.LayerNorm(features)
    elif norm_type == 'groupnorm': return nn.GroupNorm(min(8, features), features)
    elif norm_type == 'none': return nn.Identity()
    raise ValueError(f"Unknown norm: {norm_type}")


class FlexCNN(nn.Module):
    """CNN with swappable normalization."""
    def __init__(self, norm_type='batchnorm', nc=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,1,1,bias=False), get_norm(norm_type, 64), nn.ReLU(),
            nn.Conv2d(64,64,3,1,1,bias=False), get_norm(norm_type, 64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1,bias=False), get_norm(norm_type, 128), nn.ReLU(),
            nn.Conv2d(128,128,3,1,1,bias=False), get_norm(norm_type, 128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,1,1,bias=False), get_norm(norm_type, 256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(256, nc)
    
    def forward(self, x):
        return self.fc(self.features(x).flatten(1))
    
    def get_features(self, x):
        return self.features(x).flatten(1)  # [B, 256]


class FlexViT(nn.Module):
    """ViT with swappable normalization in attention blocks."""
    def __init__(self, norm_type='layernorm', nc=10, ps=4, d=128, dep=4, h=4, m=256):
        super().__init__()
        np_ = (32//ps)**2
        self.pe = nn.Conv2d(3,d,ps,ps)
        self.cls = nn.Parameter(torch.randn(1,1,d)*0.02)
        self.pos = nn.Parameter(torch.randn(1,np_+1,d)*0.02)
        
        self.blocks = nn.ModuleList()
        for _ in range(dep):
            if norm_type == 'batchnorm':
                # BN doesn't work well in transformers, use BN1d on token dim
                n1 = nn.BatchNorm1d(np_+1)
                n2 = nn.BatchNorm1d(np_+1)
            elif norm_type == 'layernorm':
                n1 = nn.LayerNorm(d); n2 = nn.LayerNorm(d)
            elif norm_type == 'groupnorm':
                n1 = nn.GroupNorm(min(8, d), d); n2 = nn.GroupNorm(min(8, d), d)
            elif norm_type == 'none':
                n1 = nn.Identity(); n2 = nn.Identity()
            
            self.blocks.append(nn.ModuleDict({
                'n1': n1, 'attn': nn.MultiheadAttention(d, h, batch_first=True),
                'n2': n2, 'mlp': nn.Sequential(nn.Linear(d,m), nn.GELU(), nn.Linear(m,d))
            }))
        
        self.norm = nn.LayerNorm(d) if norm_type != 'none' else nn.Identity()
        self.head = nn.Linear(d, nc)
        self.norm_type = norm_type
    
    def _apply_norm(self, norm, x):
        """Handle different norm shapes."""
        if isinstance(norm, nn.BatchNorm1d):
            return norm(x)  # [B, seq, dim] — BN1d on seq dim
        elif isinstance(norm, nn.GroupNorm):
            # GN expects [B, C, ...], transpose
            return norm(x.transpose(1,2)).transpose(1,2)
        return norm(x)
    
    def forward(self, x):
        B = x.shape[0]; x = self.pe(x).flatten(2).transpose(1,2)
        x = torch.cat([self.cls.expand(B,-1,-1), x], 1) + self.pos
        for blk in self.blocks:
            xn = self._apply_norm(blk['n1'], x)
            x = x + blk['attn'](xn, xn, xn)[0]
            x = x + blk['mlp'](self._apply_norm(blk['n2'], x))
        return self.head(self.norm(x[:,0]) if not isinstance(self.norm, nn.Identity) else x[:,0])
    
    def get_features(self, x):
        B = x.shape[0]; x = self.pe(x).flatten(2).transpose(1,2)
        x = torch.cat([self.cls.expand(B,-1,-1), x], 1) + self.pos
        for blk in self.blocks:
            xn = self._apply_norm(blk['n1'], x)
            x = x + blk['attn'](xn, xn, xn)[0]
            x = x + blk['mlp'](self._apply_norm(blk['n2'], x))
        out = self.norm(x[:,0]) if not isinstance(self.norm, nn.Identity) else x[:,0]
        return out


# ─── AT Metric ───

def compute_at(f1, f2, nb=25):
    d1, d2 = f1.shape[1], f2.shape[1]
    f1 = (f1-f1.mean(0))/(f1.std(0)+1e-8); f2 = (f2-f2.mean(0))/(f2.std(0)+1e-8)
    h = lambda f,n: np.array([np.histogram(f[:,c],n,(-3,3))[0].astype(np.float64)+1e-10 for c in range(f.shape[1])])
    h1 = h(f1,nb); h2 = h(f2,nb)
    h1 /= h1.sum(1,keepdims=True); h2 /= h2.sum(1,keepdims=True)
    dx = 6.0/nb
    cost = np.zeros((d1,d2))
    for i in range(d1):
        c1 = np.cumsum(h1[i])
        for j in range(d2): cost[i,j] = np.sum(np.abs(c1-np.cumsum(h2[j])))*dx
    dm = max(d1,d2)
    if d1<dm: cost = np.pad(cost,((0,dm-d1),(0,0)),constant_values=cost.max()*10)
    elif d2<dm: cost = np.pad(cost,((0,0),(0,dm-d2)),constant_values=cost.max()*10)
    ri,ci = linear_sum_assignment(cost)
    v = [cost[i,j] for i,j in zip(ri,ci) if i<d1 and j<d2]
    return float(np.mean(v)) if v else 0.0


# ─── Training ───

def get_data():
    tt = transforms.Compose([transforms.RandomCrop(32,4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    te = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    tr = torchvision.datasets.CIFAR10('./data',True,download=True,transform=tt)
    ts = torchvision.datasets.CIFAR10('./data',False,download=True,transform=te)
    return (DataLoader(Subset(tr,range(45000)),BATCH_SIZE,True,num_workers=2),
            DataLoader(Subset(tr,range(45000,50000)),BATCH_SIZE,False,num_workers=2),
            DataLoader(ts,BATCH_SIZE,False,num_workers=2))


def train_model(model, tl, vl, epochs=EPOCHS, patience=PATIENCE):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = 0; ni = 0; bs = None
    for ep in range(epochs):
        model.train()
        for x,y in tl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x),y).backward(); opt.step()
        sched.step()
        model.eval(); c=t=0
        with torch.no_grad():
            for x,y in vl: x,y=x.to(device),y.to(device); c+=(model(x).argmax(1)==y).sum().item(); t+=y.size(0)
        va = c/t
        if va > best: best=va; bs={k:v.clone() for k,v in model.state_dict().items()}; ni=0
        else: ni+=1
        if ni>=patience: break
    if bs: model.load_state_dict(bs)
    return model, best


def extract_features(model, loader):
    model.eval(); feats = []
    with torch.no_grad():
        for x,_ in loader:
            f = model.get_features(x.to(device))
            if len(f.shape)==4: f = f.mean(dim=(2,3))
            feats.append(f.cpu().numpy())
    return np.concatenate(feats)


# ─── Main ───

def run_seed(seed, tl, vl, ts):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
    print(f"\n{'='*50}\nSeed {seed}\n{'='*50}")
    
    models = {}; accs = {}
    
    for norm in NORM_TYPES:
        for arch_type in ['cnn', 'vit']:
            key = f"{arch_type}_{norm}"
            print(f"  Training {key}...")
            if arch_type == 'cnn':
                m = FlexCNN(norm_type=norm)
            else:
                m = FlexViT(norm_type=norm)
            m, a = train_model(m, tl, vl)
            models[key] = m; accs[key] = a
            print(f"    acc={a:.4f}")
    
    # Extract features
    features = {k: extract_features(m, ts) for k, m in models.items()}
    
    # Compute AT scores for key comparisons
    at_scores = {}
    
    # 1. Cross-arch with SAME norm
    for norm in NORM_TYPES:
        k = f"cnn_{norm}_vs_vit_{norm}"
        at_scores[k] = compute_at(features[f"cnn_{norm}"], features[f"vit_{norm}"])
    
    # 2. Same-arch with DIFFERENT norm
    for n1 in NORM_TYPES:
        for n2 in NORM_TYPES:
            if n1 < n2:
                k = f"cnn_{n1}_vs_cnn_{n2}"
                at_scores[k] = compute_at(features[f"cnn_{n1}"], features[f"cnn_{n2}"])
    
    # 3. Cross-arch with DIFFERENT norm
    at_scores['cnn_bn_vs_vit_ln'] = compute_at(features['cnn_batchnorm'], features['vit_layernorm'])
    at_scores['cnn_ln_vs_vit_bn'] = compute_at(features['cnn_layernorm'], features['vit_batchnorm'])
    
    for k, v in sorted(at_scores.items()):
        print(f"    {k}: AT={v:.4f}")
    
    return {'seed': seed, 'accuracies': {k:float(v) for k,v in accs.items()}, 'at_scores': at_scores}


def main():
    print("="*60 + "\nRun 086: Normalization Effects on Cross-Architecture AT\n" + "="*60)
    tl, vl, ts = get_data()
    
    results = []; start = time.time()
    for seed in range(42, 42 + N_SEEDS):
        results.append(run_seed(seed, tl, vl, ts))
    
    # Aggregate
    print(f"\n{'='*60}\nAGGREGATED RESULTS\n{'='*60}")
    
    all_at_keys = list(results[0]['at_scores'].keys())
    
    print(f"\n{'Comparison':<40} {'AT (mean±std)':>20}")
    print("-"*62)
    for k in sorted(all_at_keys):
        vals = [r['at_scores'][k] for r in results]
        print(f"{k:<40} {np.mean(vals):>8.4f} ± {np.std(vals):.4f}")
    
    # Key test: Does same-norm cross-arch < different-norm cross-arch?
    same_norm_cross = []
    diff_norm_cross = []
    for r in results:
        for norm in NORM_TYPES:
            same_norm_cross.append(r['at_scores'][f"cnn_{norm}_vs_vit_{norm}"])
        diff_norm_cross.append(r['at_scores']['cnn_bn_vs_vit_ln'])
        diff_norm_cross.append(r['at_scores']['cnn_ln_vs_vit_bn'])
    
    t, p = ttest_ind(same_norm_cross, diff_norm_cross)
    print(f"\nSame-norm cross-arch AT: {np.mean(same_norm_cross):.4f} ± {np.std(same_norm_cross):.4f}")
    print(f"Diff-norm cross-arch AT: {np.mean(diff_norm_cross):.4f} ± {np.std(diff_norm_cross):.4f}")
    print(f"t={t:.3f}, p={p:.6f}")
    
    # ANOVA: does norm type significantly affect cross-arch AT?
    groups = [[r['at_scores'][f"cnn_{n}_vs_vit_{n}"] for r in results] for n in NORM_TYPES]
    f_stat, anova_p = f_oneway(*groups)
    print(f"\nANOVA (norm type effect on cross-arch AT): F={f_stat:.3f}, p={anova_p:.6f}")
    
    final = {
        'experiment': 'run_086_at_normalization',
        'n_seeds': N_SEEDS,
        'norm_types': NORM_TYPES,
        'at_scores': {k: {'mean': float(np.mean([r['at_scores'][k] for r in results])),
                          'std': float(np.std([r['at_scores'][k] for r in results]))}
                     for k in all_at_keys},
        'same_vs_diff_norm': {'t': float(t), 'p': float(p)},
        'anova': {'f': float(f_stat), 'p': float(anova_p)},
        'per_seed': results,
        'time_min': (time.time()-start)/60
    }
    
    with open('results.json', 'w') as f: json.dump(final, f, indent=2)
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")
    print(f"RESULTS: {json.dumps({k:v for k,v in final.items() if k != 'per_seed'})}")

if __name__ == "__main__":
    main()
