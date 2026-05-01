"""Comprehensive ImageNet AT experiment.

For each (model, seed, layer): extract features, save channel histograms (for AT)
and centered kernel matrix K = X @ X.T (for CKA). Then compute pairwise AT and CKA
for all same-arch and cross-arch pairs.
"""
import torch, timm, torchvision, numpy as np, json, time, os, sys, gc, pickle
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from scipy.optimize import linear_sum_assignment
import warnings; warnings.filterwarnings("ignore")

device = torch.device("cuda")
N_SAMPLES = 5000
BATCH_SIZE = 128
AT_BINS = 25
N_SEEDS = 3
DATA_PATH = "/workspace/imagenet_v2/imagenetv2-matched-frequency-format-val"
CACHE_DIR = "/workspace/at_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# (alias, timm_name, layer_names)
# Verified: each v2 differs from v1 by >= 0.04 in mean weight magnitude
ARCHS = [
    ("resnet50_v1", "resnet50",                                 ["layer1", "layer2", "layer3", "layer4"]),
    ("resnet50_v2", "resnet50.tv_in1k",                         ["layer1", "layer2", "layer3", "layer4"]),
    ("vit_b16_v1",  "vit_base_patch16_224",                     ["blocks.2", "blocks.5", "blocks.8", "blocks.11"]),
    ("vit_b16_v2",  "deit_base_patch16_224",                    ["blocks.2", "blocks.5", "blocks.8", "blocks.11"]),
    ("mixer_b16_v1","mixer_b16_224",                            ["blocks.2", "blocks.5", "blocks.8", "blocks.11"]),
    ("mixer_b16_v2","mixer_b16_224.miil_in21k_ft_in1k",         ["blocks.2", "blocks.5", "blocks.8", "blocks.11"]),
]

# Same-arch pairs (these give us the BASELINE — what we're missing for the paper)
SAME_ARCH = [
    ("resnet50_v1", "resnet50_v2"),
    ("vit_b16_v1",  "vit_b16_v2"),
    ("mixer_b16_v1","mixer_b16_v2"),
]
# Cross-arch pairs (for each: use v1 of each)
CROSS_ARCH = [
    ("resnet50_v1", "vit_b16_v1"),
    ("resnet50_v1", "mixer_b16_v1"),
    ("vit_b16_v1",  "mixer_b16_v1"),
]
LAYER_INDEX_PAIRS = [(0, 0), (1, 1), (2, 2), (3, 3)]  # zip layers across two archs

T0 = time.time()
def dt():
    return f"{time.time()-T0:.1f}s"

def compute_at(h1, h2):
    n = min(len(h1), len(h2))
    h1 = np.asarray(h1[:n]); h2 = np.asarray(h2[:n])
    c1 = np.cumsum(h1, axis=1); c2 = np.cumsum(h2, axis=1)
    cost = np.abs(c1[:, None, :] - c2[None, :, :]).sum(axis=-1)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())

def hists_cpu(feats, n_bins=AT_BINS):
    """Optimized: vectorized bin index computation + np.bincount per channel."""
    feats = feats.cpu()
    if feats.dim() == 4:
        feats = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])
    elif feats.dim() == 3:
        feats = feats.reshape(-1, feats.shape[-1])
    feats = feats.numpy()
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-8
    feats = (feats - mean) / std
    np.clip(feats, -3.0, 3.0, out=feats)
    bin_idx = np.floor((feats + 3.0) / 6.0 * n_bins).astype(np.int64)
    np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
    n_channels = feats.shape[1]
    histograms = np.zeros((n_channels, n_bins), dtype=np.float64)
    for c in range(n_channels):
        h = np.bincount(bin_idx[:, c], minlength=n_bins)
        s = h.sum()
        if s > 0:
            histograms[c] = h / s
    return histograms

def kernel_centered(feats):
    """K = (X - X.mean) @ (X - X.mean).T where X is (N, D)."""
    X = feats.cpu().reshape(feats.shape[0], -1).float().numpy()
    X = X - X.mean(axis=0, keepdims=True)
    return X @ X.T

def cka_from_kernels(Kx, Ky):
    hxy = float((Kx * Ky).sum())
    hxx = float((Kx * Kx).sum())
    hyy = float((Ky * Ky).sum())
    return hxy / (np.sqrt(hxx * hyy) + 1e-10)

def get_subset_indices(seed, dataset_size):
    rng = np.random.RandomState(seed)
    return rng.permutation(dataset_size)[:N_SAMPLES]

def cache_path(alias, seed, layer):
    safe_layer = layer.replace(".", "_")
    return os.path.join(CACHE_DIR, f"{alias}_seed{seed}_{safe_layer}.npz")

def already_cached(alias, seed, layers):
    return all(os.path.exists(cache_path(alias, seed, l)) for l in layers)

def extract_for_model(alias, timm_name, layers, dataset, accs_dict):
    """Run all 3 seeds on this model, save histograms + kernel matrices to disk."""
    print(f"[{dt()}] >>> {alias} ({timm_name})", flush=True)

    # Skip if all cached
    all_done = all(already_cached(alias, 42 + si, layers) for si in range(N_SEEDS))
    if all_done:
        print(f"[{dt()}]   already cached, skipping", flush=True)
        return

    print(f"[{dt()}]   loading model", flush=True)
    model = timm.create_model(timm_name, pretrained=True).to(device).eval()
    cfg = timm.data.resolve_model_data_config(model)
    print(f"[{dt()}]   input cfg: {cfg.get('input_size','default')}, mean={cfg.get('mean')}", flush=True)

    # Standard 224 transform; if model wants something else, we still feed 224 (acceptable for AT, accs may shift)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=cfg.get("mean", [0.485, 0.456, 0.406]),
                             std=cfg.get("std",  [0.229, 0.224, 0.225])),
    ])
    dataset.transform = transform

    for si in range(N_SEEDS):
        seed = 42 + si
        if already_cached(alias, seed, layers):
            print(f"[{dt()}]   seed {seed} cached, skipping", flush=True)
            continue
        print(f"[{dt()}]   seed {seed}: forward pass", flush=True)
        idx = get_subset_indices(seed, len(dataset))
        loader = DataLoader(Subset(dataset, idx), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

        captures = {}
        def make_hook(key):
            def fn(m, i, o):
                captures[key] = o.detach()
            return fn
        handles = []
        for n, mod in model.named_modules():
            if n in layers:
                handles.append(mod.register_forward_hook(make_hook(n)))

        feat_buf = {l: [] for l in layers}
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
                for k, v in captures.items():
                    feat_buf[k].append(v.cpu())
                captures.clear()
        for h in handles: h.remove()
        acc = correct / total
        accs_dict.setdefault(alias, {})[seed] = acc
        print(f"[{dt()}]   seed {seed}: acc={acc:.4f}", flush=True)

        for layer in layers:
            t_l = time.time()
            feats = torch.cat(feat_buf[layer], dim=0)
            feat_buf[layer] = []  # free
            hists = hists_cpu(feats)
            kern = kernel_centered(feats)
            np.savez_compressed(cache_path(alias, seed, layer), hists=hists, kern=kern)
            del feats, hists, kern
            gc.collect()
            print(f"[{dt()}]     layer {layer} done in {time.time()-t_l:.1f}s", flush=True)
        del feat_buf
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

def load_cached(alias, seed, layer):
    d = np.load(cache_path(alias, seed, layer))
    return d["hists"], d["kern"]

def compute_pair(alias_a, layers_a, alias_b, layers_b):
    """Compute AT, CKA, layer-wise AT for a pair across all seeds."""
    seed_results = []
    for si in range(N_SEEDS):
        seed = 42 + si
        ha, ka = load_cached(alias_a, seed, layers_a[-1])
        hb, kb = load_cached(alias_b, seed, layers_b[-1])
        at_final = compute_at(ha, hb)
        cka_final = cka_from_kernels(ka, kb)
        layer_at = {}
        for la, lb in zip(layers_a, layers_b):
            ha_l, _ = load_cached(alias_a, seed, la)
            hb_l, _ = load_cached(alias_b, seed, lb)
            layer_at[f"{la}_vs_{lb}"] = compute_at(ha_l, hb_l)
        seed_results.append({
            "seed": seed, "at_final": at_final, "cka_final": cka_final,
            "layer_wise_at": layer_at,
        })
    at_vals = [r["at_final"] for r in seed_results]
    cka_vals = [r["cka_final"] for r in seed_results]
    return {
        "pair": f"{alias_a}_vs_{alias_b}",
        "per_seed": seed_results,
        "at_mean": float(np.mean(at_vals)),
        "at_std": float(np.std(at_vals, ddof=1)),
        "cka_mean": float(np.mean(cka_vals)),
        "cka_std": float(np.std(cka_vals, ddof=1)),
    }

def main():
    print(f"[{dt()}] start", flush=True)
    raw_dataset = torchvision.datasets.ImageFolder(DATA_PATH)
    raw_dataset.samples = [(p, int(os.path.basename(os.path.dirname(p)))) for p, _ in raw_dataset.samples]
    raw_dataset.targets = [t for _, t in raw_dataset.samples]
    print(f"[{dt()}] dataset: {len(raw_dataset)} samples", flush=True)

    accs = {}

    # Phase 1: per-model feature/histogram/kernel extraction
    for alias, timm_name, layers in ARCHS:
        try:
            extract_for_model(alias, timm_name, layers, raw_dataset, accs)
        except Exception as e:
            print(f"[{dt()}]   FAILED {alias}: {e}", flush=True)
            import traceback; traceback.print_exc()

    print(f"\n[{dt()}] === phase 2: pairwise comparisons ===", flush=True)
    layer_lookup = {alias: layers for alias, _, layers in ARCHS}

    same_results = []
    for a, b in SAME_ARCH:
        if a in layer_lookup and b in layer_lookup:
            try:
                r = compute_pair(a, layer_lookup[a], b, layer_lookup[b])
                same_results.append(r)
                print(f"[{dt()}] SAME {r['pair']}: AT={r['at_mean']:.4f}+-{r['at_std']:.4f} CKA={r['cka_mean']:.4f}+-{r['cka_std']:.4f}", flush=True)
            except Exception as e:
                print(f"[{dt()}] SAME {a}_vs_{b} FAILED: {e}", flush=True)

    cross_results = []
    for a, b in CROSS_ARCH:
        if a in layer_lookup and b in layer_lookup:
            try:
                r = compute_pair(a, layer_lookup[a], b, layer_lookup[b])
                cross_results.append(r)
                print(f"[{dt()}] CROSS {r['pair']}: AT={r['at_mean']:.4f}+-{r['at_std']:.4f} CKA={r['cka_mean']:.4f}+-{r['cka_std']:.4f}", flush=True)
            except Exception as e:
                print(f"[{dt()}] CROSS {a}_vs_{b} FAILED: {e}", flush=True)

    # Aggregate ratios
    summary = {"accuracies": accs, "same_arch": same_results, "cross_arch": cross_results}
    if same_results and cross_results:
        same_at = np.mean([r["at_mean"] for r in same_results])
        cross_at = np.mean([r["at_mean"] for r in cross_results])
        same_cka = np.mean([r["cka_mean"] for r in same_results])
        cross_cka = np.mean([r["cka_mean"] for r in cross_results])
        summary["headline"] = {
            "same_arch_at_mean": float(same_at),
            "cross_arch_at_mean": float(cross_at),
            "at_ratio_cross_over_same": float(cross_at / same_at) if same_at > 0 else None,
            "same_arch_cka_mean": float(same_cka),
            "cross_arch_cka_mean": float(cross_cka),
            "cka_ratio_cross_over_same": float(cross_cka / same_cka) if same_cka > 0 else None,
        }
        print(f"\n[{dt()}] === HEADLINE ===", flush=True)
        print(f"  AT  cross/same = {cross_at:.4f} / {same_at:.4f} = {cross_at/same_at:.2f}x", flush=True)
        print(f"  CKA cross/same = {cross_cka:.4f} / {same_cka:.4f} = {cross_cka/same_cka:.2f}x", flush=True)

    out = "/workspace/at_imagenet_full_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{dt()}] SAVED -> {out}", flush=True)

if __name__ == "__main__":
    main()
