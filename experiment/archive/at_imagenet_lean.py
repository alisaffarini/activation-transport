"""Lean ImageNet AT experiment with explicit timing per phase. CPU histograms for memory safety."""
import torch, timm, torchvision, numpy as np, json, time, os, sys
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from scipy.optimize import linear_sum_assignment
import warnings; warnings.filterwarnings("ignore")

device = torch.device("cuda")
N_SAMPLES = 5000
BATCH_SIZE = 128
AT_BINS = 25
N_SEEDS = 3

T0 = time.time()
def dt():
    return f"{time.time()-T0:.1f}s"

def compute_at_vectorized(h1, h2):
    n = min(len(h1), len(h2))
    h1 = np.asarray(h1[:n]); h2 = np.asarray(h2[:n])
    c1 = np.cumsum(h1, axis=1); c2 = np.cumsum(h2, axis=1)
    cost = np.abs(c1[:, None, :] - c2[None, :, :]).sum(axis=-1)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())

def hists_cpu(feats, n_bins=AT_BINS):
    """All-CPU histograms. feats: torch tensor (any device)."""
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
    n_channels = feats.shape[1]
    histograms = np.zeros((n_channels, n_bins), dtype=np.float64)
    bin_edges = np.linspace(-3.0, 3.0, n_bins + 1)
    for c in range(n_channels):
        h, _ = np.histogram(feats[:, c], bins=bin_edges, density=False)
        s = h.sum()
        if s > 0:
            histograms[c] = h / s
    return histograms

def cka_score_chunked(X, Y):
    """CKA via kernel-matrix formulation: HSIC = trace(K_X @ K_Y) on (n,n) matrices.
    Equivalent to ||X^T Y||_F^2 but avoids the (d_X, d_Y) intermediate."""
    X = X.cpu().reshape(X.shape[0], -1).float().numpy()
    Y = Y.cpu().reshape(Y.shape[0], -1).float().numpy()
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Kx = X @ X.T
    Ky = Y @ Y.T
    hxy = float((Kx * Ky).sum())
    hxx = float((Kx * Kx).sum())
    hyy = float((Ky * Ky).sum())
    return hxy / (np.sqrt(hxx * hyy) + 1e-10)

def main():
    print(f"[{dt()}] starting", flush=True)
    print(f"[{dt()}] loading models", flush=True)
    resnet = timm.create_model("resnet50", pretrained=True).to(device).eval()
    vit = timm.create_model("vit_base_patch16_224", pretrained=True).to(device).eval()
    mixer = timm.create_model("mixer_b16_224", pretrained=True).to(device).eval()
    print(f"[{dt()}] models loaded", flush=True)

    rl = ["layer1", "layer2", "layer3", "layer4"]
    vl = ["blocks.2", "blocks.5", "blocks.8", "blocks.11"]
    ml = ["blocks.2", "blocks.5", "blocks.8", "blocks.11"]

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = torchvision.datasets.ImageFolder(
        "/workspace/imagenet_v2/imagenetv2-matched-frequency-format-val", transform=transform)
    val_dataset.samples = [(p, int(os.path.basename(os.path.dirname(p)))) for p, _ in val_dataset.samples]
    val_dataset.targets = [t for _, t in val_dataset.samples]
    print(f"[{dt()}] dataset ready: {len(val_dataset)} samples", flush=True)

    all_results = []
    for si in range(N_SEEDS):
        seed = 42 + si
        print(f"\n[{dt()}] === seed {seed} ===", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        idx = np.random.permutation(len(val_dataset))[:N_SAMPLES]
        loader = DataLoader(Subset(val_dataset, idx), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

        captures = {}
        def make_hook(key):
            def fn(m, i, o):
                captures[key] = o.detach()
            return fn
        handles = []
        for n, mod in resnet.named_modules():
            if n in rl: handles.append(mod.register_forward_hook(make_hook(("resnet", n))))
        for n, mod in vit.named_modules():
            if n in vl: handles.append(mod.register_forward_hook(make_hook(("vit", n))))
        for n, mod in mixer.named_modules():
            if n in ml: handles.append(mod.register_forward_hook(make_hook(("mixer", n))))

        feat_buf = {("resnet", l): [] for l in rl}
        feat_buf.update({("vit", l): [] for l in vl})
        feat_buf.update({("mixer", l): [] for l in ml})
        correct = {"resnet": 0, "vit": 0, "mixer": 0}
        total = 0
        print(f"[{dt()}] forward passes...", flush=True)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                for name, model in [("resnet", resnet), ("vit", vit), ("mixer", mixer)]:
                    out = model(x)
                    correct[name] += (out.argmax(1) == y).sum().item()
                total += y.size(0)
                for k, v in captures.items():
                    feat_buf[k].append(v.cpu())
                captures.clear()
        for h in handles: h.remove()
        accs = {k: correct[k] / total for k in correct}
        print(f"[{dt()}] forward done, accs: {accs}", flush=True)

        cat = {k: torch.cat(v, dim=0) for k, v in feat_buf.items() if v}
        del feat_buf
        print(f"[{dt()}] concatenated", flush=True)

        # Histograms on CPU per layer to keep memory bounded
        hists = {}
        for k in list(cat.keys()):
            t_h = time.time()
            hists[k] = hists_cpu(cat[k])
            print(f"[{dt()}]   hist {k}: {time.time()-t_h:.1f}s, shape={hists[k].shape}", flush=True)
        print(f"[{dt()}] all histograms done", flush=True)

        at_rv = compute_at_vectorized(hists[("resnet", rl[-1])], hists[("vit", vl[-1])])
        at_rm = compute_at_vectorized(hists[("resnet", rl[-1])], hists[("mixer", ml[-1])])
        at_vm = compute_at_vectorized(hists[("vit", vl[-1])], hists[("mixer", ml[-1])])
        print(f"[{dt()}] final-layer AT: rv={at_rv:.4f} rm={at_rm:.4f} vm={at_vm:.4f}", flush=True)

        layer_at = {}
        for r, v in zip(rl, vl):
            layer_at[f"{r}_vs_{v}"] = compute_at_vectorized(hists[("resnet", r)], hists[("vit", v)])
        print(f"[{dt()}] layer-wise AT: {layer_at}", flush=True)

        # CKA on CPU (final layer only)
        cka_rv = cka_score_chunked(cat[("resnet", rl[-1])], cat[("vit", vl[-1])])
        cka_rm = cka_score_chunked(cat[("resnet", rl[-1])], cat[("mixer", ml[-1])])
        cka_vm = cka_score_chunked(cat[("vit", vl[-1])], cat[("mixer", ml[-1])])
        print(f"[{dt()}] CKA: rv={cka_rv:.4f} rm={cka_rm:.4f} vm={cka_vm:.4f}", flush=True)

        all_results.append({
            "seed": seed,
            "accs": accs,
            "at": {"resnet_vs_vit": at_rv, "resnet_vs_mixer": at_rm, "vit_vs_mixer": at_vm},
            "cka": {"resnet_vs_vit": cka_rv, "resnet_vs_mixer": cka_rm, "vit_vs_mixer": cka_vm},
            "layer_wise_at": layer_at,
        })
        del cat, hists; torch.cuda.empty_cache()

    print(f"\n[{dt()}] === aggregated ===", flush=True)
    final = {
        "n_seeds": N_SEEDS, "n_samples": N_SAMPLES,
        "models": {"resnet": "resnet50", "vit": "vit_base_patch16_224", "mixer": "mixer_b16_224"},
        "per_seed": all_results,
    }
    for pair in ["resnet_vs_vit", "resnet_vs_mixer", "vit_vs_mixer"]:
        v = [r["at"][pair] for r in all_results]
        c = [r["cka"][pair] for r in all_results]
        final[f"at_{pair}"] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
        final[f"cka_{pair}"] = {"mean": float(np.mean(c)), "std": float(np.std(c, ddof=1))}
        print(f"  AT {pair}: {np.mean(v):.4f} +- {np.std(v, ddof=1):.4f}", flush=True)
        print(f"  CKA {pair}: {np.mean(c):.4f} +- {np.std(c, ddof=1):.4f}", flush=True)

    out = "/workspace/at_imagenet_lean_results.json"
    with open(out, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n[{dt()}] SAVED -> {out}", flush=True)

if __name__ == "__main__":
    main()
