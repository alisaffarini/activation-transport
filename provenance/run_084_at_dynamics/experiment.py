#!/usr/bin/env python3
"""Run 084: AT Training Dynamics — HOW and WHEN do architectures diverge in feature space?

Key question: Do architectures start similar and diverge? Or are they always different?
Measure AT score at checkpoints throughout training.

This gives us the "feature divergence dynamics" section of the paper.
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
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() 
                       else 'mps' if torch.backends.mps.is_available() 
                       else 'cpu')
print(f"Device: {device}")

N_SEEDS = 5  # Fewer seeds since we're measuring at many checkpoints
EPOCHS = 50
CHECKPOINT_EPOCHS = [0, 1, 2, 5, 10, 15, 20, 30, 40, 50]  # When to measure AT
LR = 0.001
BATCH_SIZE = 128
N_CLASSES = 10

# ─── Model Definitions (same as 082, compact) ───

class BasicBlock(nn.Module):
    def __init__(self, ic, oc, s=1, ds=None):
        super().__init__()
        self.c1 = nn.Conv2d(ic,oc,3,s,1,bias=False); self.b1 = nn.BatchNorm2d(oc)
        self.c2 = nn.Conv2d(oc,oc,3,1,1,bias=False); self.b2 = nn.BatchNorm2d(oc); self.ds = ds
    def forward(self, x):
        i = x; o = F.relu(self.b1(self.c1(x))); o = self.b2(self.c2(o))
        if self.ds: i = self.ds(x)
        return F.relu(o + i)

class ResNet18(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1,bias=False); self.bn1 = nn.BatchNorm2d(64)
        self.l1 = self._ml(64,64,2); self.l2 = self._ml(64,128,2,2)
        self.l3 = self._ml(128,256,2,2); self.l4 = self._ml(256,512,2,2)
        self.pool = nn.AdaptiveAvgPool2d(1); self.fc = nn.Linear(512, nc)
    def _ml(self, ic, oc, n, s=1):
        ds = nn.Sequential(nn.Conv2d(ic,oc,1,s,bias=False),nn.BatchNorm2d(oc)) if s!=1 or ic!=oc else None
        return nn.Sequential(BasicBlock(ic,oc,s,ds), *[BasicBlock(oc,oc) for _ in range(n-1)])
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for l in [self.l1,self.l2,self.l3,self.l4]: x = l(x)
        return self.fc(self.pool(x).flatten(1))
    def get_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for l in [self.l1,self.l2,self.l3,self.l4]: x = l(x)
        return self.pool(x).flatten(1)

class TBlock(nn.Module):
    def __init__(self, d, h, m):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d,h,batch_first=True)
        self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d,m),nn.GELU(),nn.Linear(m,d))
    def forward(self, x):
        x = x + self.attn(*([self.n1(x)]*3))[0]; return x + self.mlp(self.n2(x))

class SmallViT(nn.Module):
    def __init__(self, nc=10, ps=4, d=192, dep=6, h=6, m=384):
        super().__init__()
        np_ = (32//ps)**2
        self.pe = nn.Conv2d(3,d,ps,ps); self.cls = nn.Parameter(torch.randn(1,1,d)*0.02)
        self.pos = nn.Parameter(torch.randn(1,np_+1,d)*0.02)
        self.blocks = nn.ModuleList([TBlock(d,h,m) for _ in range(dep)])
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, nc)
    def forward(self, x):
        B = x.shape[0]; x = self.pe(x).flatten(2).transpose(1,2)
        x = torch.cat([self.cls.expand(B,-1,-1), x], 1) + self.pos
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x[:,0]))
    def get_features(self, x):
        B = x.shape[0]; x = self.pe(x).flatten(2).transpose(1,2)
        x = torch.cat([self.cls.expand(B,-1,-1), x], 1) + self.pos
        for b in self.blocks: x = b(x)
        return self.norm(x[:,0])


# ─── AT Metric ───

def compute_at(f1, f2, nb=25):
    d1, d2 = f1.shape[1], f2.shape[1]
    f1 = (f1 - f1.mean(0))/(f1.std(0)+1e-8); f2 = (f2 - f2.mean(0))/(f2.std(0)+1e-8)
    
    def hist(f, n):
        return np.array([np.histogram(f[:,c],bins=n,range=(-3,3))[0].astype(np.float64)+1e-10 for c in range(f.shape[1])])
    
    h1 = hist(f1,nb); h2 = hist(f2,nb)
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
    valid = [cost[i,j] for i,j in zip(ri,ci) if i<d1 and j<d2]
    return float(np.mean(valid)) if valid else 0.0


# ─── Data ───

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


def extract_features(model, loader):
    model.eval()
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            f = model.get_features(x.to(device))
            if len(f.shape)==4: f = f.mean(dim=(2,3))
            feats.append(f.cpu().numpy())
    return np.concatenate(feats)


# ─── Main: Train with checkpoints ───

def train_with_checkpoints(model, train_loader, test_loader, checkpoints):
    """Train model, saving features at specified epoch checkpoints."""
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    
    features_at_epoch = {}
    
    # Epoch 0 (before training)
    if 0 in checkpoints:
        features_at_epoch[0] = extract_features(model, test_loader)
    
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        sched.step()
        
        if ep in checkpoints:
            features_at_epoch[ep] = extract_features(model, test_loader)
            
            # Quick accuracy check
            model.eval()
            c = t = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    c += (model(x).argmax(1)==y).sum().item(); t += y.size(0)
            print(f"    Epoch {ep}: acc={c/t:.4f}, feat_shape={features_at_epoch[ep].shape}")
    
    return features_at_epoch


def run_seed(seed, train_loader, val_loader, test_loader):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
    print(f"\n{'='*50}\nSeed {seed}\n{'='*50}")
    
    # Train both architectures with checkpoints
    print("  Training ResNet-18 with checkpoints...")
    resnet = ResNet18()
    resnet_feats = train_with_checkpoints(resnet, train_loader, test_loader, CHECKPOINT_EPOCHS)
    
    print("  Training ViT with checkpoints...")
    vit = SmallViT()
    vit_feats = train_with_checkpoints(vit, train_loader, test_loader, CHECKPOINT_EPOCHS)
    
    # Train second ResNet (same-arch baseline)
    print("  Training ResNet-18 (baseline) with checkpoints...")
    torch.manual_seed(seed + 1000)
    resnet2 = ResNet18()
    resnet2_feats = train_with_checkpoints(resnet2, train_loader, test_loader, CHECKPOINT_EPOCHS)
    
    # Compute AT at each checkpoint
    dynamics = {'cross_arch': {}, 'same_arch': {}}
    
    for ep in CHECKPOINT_EPOCHS:
        if ep in resnet_feats and ep in vit_feats:
            dynamics['cross_arch'][ep] = compute_at(resnet_feats[ep], vit_feats[ep])
        if ep in resnet_feats and ep in resnet2_feats:
            dynamics['same_arch'][ep] = compute_at(resnet_feats[ep], resnet2_feats[ep])
        
        print(f"  Epoch {ep}: cross_AT={dynamics['cross_arch'].get(ep, 'N/A'):.4f}, "
              f"same_AT={dynamics['same_arch'].get(ep, 'N/A'):.4f}")
    
    return {'seed': seed, 'dynamics': dynamics}


def main():
    print("="*60 + "\nRun 084: AT Training Dynamics\n" + "="*60)
    train_loader, val_loader, test_loader = get_data()
    
    results = []; start = time.time()
    for seed in range(42, 42 + N_SEEDS):
        results.append(run_seed(seed, train_loader, val_loader, test_loader))
    
    # Aggregate dynamics
    print(f"\n{'='*60}\nAGGREGATED DYNAMICS")
    print(f"{'='*60}")
    print(f"\n{'Epoch':<8} {'Cross-Arch AT':>15} {'Same-Arch AT':>15} {'Ratio':>10}")
    print("-" * 50)
    
    for ep in CHECKPOINT_EPOCHS:
        cross = [r['dynamics']['cross_arch'].get(ep, np.nan) for r in results]
        same = [r['dynamics']['same_arch'].get(ep, np.nan) for r in results]
        cross_m, same_m = np.nanmean(cross), np.nanmean(same)
        ratio = cross_m / same_m if same_m > 0 else float('inf')
        print(f"{ep:<8} {cross_m:>10.4f} ± {np.nanstd(cross):>4.4f}  "
              f"{same_m:>10.4f} ± {np.nanstd(same):>4.4f}  {ratio:>8.2f}x")
    
    final = {
        'experiment': 'run_084_at_dynamics',
        'n_seeds': N_SEEDS,
        'checkpoint_epochs': CHECKPOINT_EPOCHS,
        'dynamics': {
            str(ep): {
                'cross_arch': {'mean': float(np.nanmean([r['dynamics']['cross_arch'].get(ep,np.nan) for r in results])),
                               'std': float(np.nanstd([r['dynamics']['cross_arch'].get(ep,np.nan) for r in results]))},
                'same_arch': {'mean': float(np.nanmean([r['dynamics']['same_arch'].get(ep,np.nan) for r in results])),
                              'std': float(np.nanstd([r['dynamics']['same_arch'].get(ep,np.nan) for r in results]))}
            } for ep in CHECKPOINT_EPOCHS
        },
        'per_seed': results,
        'time_min': (time.time()-start)/60
    }
    
    with open('results.json', 'w') as f: json.dump(final, f, indent=2)
    print(f"\nRESULTS: {json.dumps({k:v for k,v in final.items() if k != 'per_seed'})}")

if __name__ == "__main__":
    main()
