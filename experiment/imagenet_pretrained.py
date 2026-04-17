#!/usr/bin/env python3
"""ImageNet-scale Activation Transport using pretrained models from timm.

Uses accuracy-matched pretrained models:
- ResNet-50 (timm): ~80.4% top-1
- ViT-B/16 (timm): ~81.1% top-1
- Mixer-B/16 (timm): ~78.5% top-1

No training needed — just feature extraction and AT/CKA computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import numpy as np
import json
import time
import warnings
from scipy.stats import ttest_ind
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

N_SAMPLES = 5000  # ImageNet val samples for feature extraction
BATCH_SIZE = 64
AT_BINS = 25
N_SEEDS = 3  # random subsets of val set

# ─── Feature Extraction Hooks ───

class FeatureExtractor:
    def __init__(self, model, layer_names):
        self.features = {}
        self.hooks = []
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(
                    lambda m, inp, out, name=name: self._hook_fn(name, out))
                self.hooks.append(hook)

    def _hook_fn(self, name, output):
        if isinstance(output, tuple):
            output = output[0]
        self.features[name] = output.detach()

    def remove(self):
        for h in self.hooks:
            h.remove()

# ─── AT Computation (same as CIFAR experiments) ───

def compute_channel_histograms(features, n_bins=AT_BINS):
    if features.dim() == 4:
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C)
    elif features.dim() == 3:
        B, T, D = features.shape
        features = features.reshape(-1, D)
    elif features.dim() == 2:
        features = features

    features = (features - features.mean(0, keepdim=True)) / (features.std(0, keepdim=True) + 1e-8)
    n_channels = features.shape[1]
    histograms = []
    for c in range(n_channels):
        vals = features[:, c].cpu().numpy()
        hist, _ = np.histogram(vals, bins=n_bins, range=(-3, 3), density=True)
        hist = hist / (hist.sum() + 1e-10)
        histograms.append(hist)
    return np.array(histograms)

def wasserstein_1d(p, q):
    return np.sum(np.abs(np.cumsum(p) - np.cumsum(q)))

def compute_at_score(hist1, hist2):
    n1, n2 = len(hist1), len(hist2)
    n = min(n1, n2)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = wasserstein_1d(hist1[i], hist2[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].mean()

def compute_cka(X, Y):
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = torch.norm(X.T @ Y, p='fro') ** 2
    hsic_xx = torch.norm(X.T @ X, p='fro') ** 2
    hsic_yy = torch.norm(Y.T @ Y, p='fro') ** 2
    return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)).item()

# ─── Main ───

def main():
    import timm

    print("Loading pretrained models...")

    # Load models
    resnet = timm.create_model('resnet50', pretrained=True).to(device).eval()
    vit = timm.create_model('vit_base_patch16_224', pretrained=True).to(device).eval()
    mixer = timm.create_model('mixer_b16_224', pretrained=True).to(device).eval()

    # Define layer names for feature extraction
    resnet_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    vit_layers = ['blocks.2', 'blocks.5', 'blocks.8', 'blocks.11']
    mixer_layers = ['blocks.2', 'blocks.5', 'blocks.8', 'blocks.11']

    # ImageNet validation transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading ImageNet-V2 validation set...")
    val_dataset = torchvision.datasets.ImageFolder('/workspace/imagenet_v2/imagenetv2-matched-frequency-format-val', transform=transform)

    print(f"Validation set size: {len(val_dataset)}")

    all_results = []

    for seed_idx in range(N_SEEDS):
        seed = 42 + seed_idx
        print(f"\n{'='*50}\nSeed {seed} (subset {seed_idx+1}/{N_SEEDS})\n{'='*50}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        indices = np.random.permutation(len(val_dataset))[:N_SAMPLES]
        subset = Subset(val_dataset, indices)
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        # Evaluate accuracy
        print("  Evaluating accuracy...")
        for name, model in [('resnet', resnet), ('vit', vit), ('mixer', mixer)]:
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)
            print(f"    {name}: {correct/total:.4f}")

        # Extract features
        print("  Extracting features...")

        def extract_final_features(model, layer_names, loader):
            extractor = FeatureExtractor(model, layer_names)
            all_feats = {name: [] for name in layer_names}
            with torch.no_grad():
                for x, _ in loader:
                    x = x.to(device)
                    _ = model(x)
                    for name in layer_names:
                        if name in extractor.features:
                            all_feats[name].append(extractor.features[name].cpu())
            extractor.remove()
            return {name: torch.cat(feats, dim=0) for name, feats in all_feats.items() if feats}

        resnet_feats = extract_final_features(resnet, resnet_layers, loader)
        vit_feats = extract_final_features(vit, vit_layers, loader)
        mixer_feats = extract_final_features(mixer, mixer_layers, loader)

        # Compute AT at final layer
        print("  Computing AT scores...")
        r_hist = compute_channel_histograms(resnet_feats[resnet_layers[-1]])
        v_hist = compute_channel_histograms(vit_feats[vit_layers[-1]])
        m_hist = compute_channel_histograms(mixer_feats[mixer_layers[-1]])

        at_rv = compute_at_score(r_hist, v_hist)
        at_rm = compute_at_score(r_hist, m_hist)
        at_vm = compute_at_score(v_hist, m_hist)

        # CKA at final layer
        r_final = resnet_feats[resnet_layers[-1]]
        v_final = vit_feats[vit_layers[-1]]
        m_final = mixer_feats[mixer_layers[-1]]

        cka_rv = compute_cka(r_final, v_final)
        cka_rm = compute_cka(r_final, m_final)
        cka_vm = compute_cka(v_final, m_final)

        # Layer-wise AT (ResNet vs ViT)
        layer_at = {}
        for r_layer, v_layer in zip(resnet_layers, vit_layers):
            if r_layer in resnet_feats and v_layer in vit_feats:
                rh = compute_channel_histograms(resnet_feats[r_layer])
                vh = compute_channel_histograms(vit_feats[v_layer])
                layer_at[f"{r_layer}_vs_{v_layer}"] = compute_at_score(rh, vh)

        seed_result = {
            'seed': seed,
            'at': {'resnet_vs_vit': at_rv, 'resnet_vs_mixer': at_rm, 'vit_vs_mixer': at_vm},
            'cka': {'resnet_vs_vit': cka_rv, 'resnet_vs_mixer': cka_rm, 'vit_vs_mixer': cka_vm},
            'layer_wise_at': layer_at,
        }
        all_results.append(seed_result)

        print(f"    ResNet vs ViT:   AT={at_rv:.4f}, CKA={cka_rv:.4f}")
        print(f"    ResNet vs Mixer: AT={at_rm:.4f}, CKA={cka_rm:.4f}")
        print(f"    ViT vs Mixer:    AT={at_vm:.4f}, CKA={cka_vm:.4f}")

    # Aggregate
    print(f"\n{'='*50}\nAGGREGATED ({N_SEEDS} seeds)\n{'='*50}")

    at_rv_all = [r['at']['resnet_vs_vit'] for r in all_results]
    at_rm_all = [r['at']['resnet_vs_mixer'] for r in all_results]
    at_vm_all = [r['at']['vit_vs_mixer'] for r in all_results]

    final = {
        'experiment': 'imagenet_pretrained',
        'n_seeds': N_SEEDS,
        'models': {'resnet': 'resnet50', 'vit': 'vit_base_patch16_224', 'mixer': 'mixer_b16_224'},
        'n_samples': N_SAMPLES,
        'at_scores': {
            'resnet_vs_vit': {'mean': float(np.mean(at_rv_all)), 'std': float(np.std(at_rv_all, ddof=1))},
            'resnet_vs_mixer': {'mean': float(np.mean(at_rm_all)), 'std': float(np.std(at_rm_all, ddof=1))},
            'vit_vs_mixer': {'mean': float(np.mean(at_vm_all)), 'std': float(np.std(at_vm_all, ddof=1))},
        },
        'per_seed': all_results,
    }

    for pair in ['resnet_vs_vit', 'resnet_vs_mixer', 'vit_vs_mixer']:
        vals = [r['at'][pair] for r in all_results]
        print(f"  {pair}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")

    out_path = 'imagenet_pretrained_results.json'
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\nRESULTS: {json.dumps(final, default=str)}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
