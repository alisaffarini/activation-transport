# Activation Transport — Next Steps to Main Conference

## Status
- Current paper: `papers/activation_transport/paper/main.tex` / `final.pdf`
- Current state: **workshop/mid-tier ready**. Core finding is solid (10× AT gap, CKA failure). Needs stronger experiments for ICML/NeurIPS main track.
- Hardware used so far: RTX 3070 Ti (8GB). Existing CIFAR-10 code is in `research/runs/run_082_at_cifar10/`.

---

## The Problem with the Current Paper

The one objection that will kill reviews:

> *"ViT achieves only 77% vs ResNet's 93% on CIFAR-10 from scratch. Maybe the high AT divergence just reflects ViT failing to learn properly on small-scale data, not a genuine architectural difference."*

The fix is **not** to train bigger models from scratch. It's to use **pretrained ImageNet weights** (already public, free to download) and just run inference. No training needed.

---

## Three Experiments Needed (in priority order)

### 1. Pretrained Feature Extraction [HIGHEST PRIORITY — ~30 min compute]

Load public pretrained weights from `timm`, extract penultimate-layer features on CIFAR-10 test set, compute AT and CKA. No training. All models at full quality.

Models to use:
- `resnet50` (pretrained on ImageNet-1K, torchvision)
- `vit_base_patch16_224` (pretrained ViT-B/16, timm)
- `deit_small_patch16_224` (DeiT-S, timm — better than raw ViT on CIFAR scale)
- `mixer_b16_224` (MLP-Mixer-B/16, timm)

What this gives you: eliminates the accuracy-gap objection entirely. All models 80%+ on ImageNet. If AT divergence persists (it will), the finding is bulletproof.

**Code to write:** A new script `research/runs/run_090_pretrained_at/experiment.py` that:
```python
import timm, torch
from scipy.optimize import linear_sum_assignment

# Load pretrained models
models = {
    'resnet50': timm.create_model('resnet50', pretrained=True, num_classes=0),
    'vit_b16': timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0),
    'deit_s': timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=0),
    'mixer_b16': timm.create_model('mixer_b16_224', pretrained=True, num_classes=0),
}
# Resize CIFAR-10 to 224x224 (torchvision transforms), extract features, compute AT + CKA
# Report pairwise AT and CKA for all pairs
```

Key detail: resize CIFAR-10 images to 224×224 (standard for these pretrained models).

### 2. More Seeds: 2 → 5 [~3 hours on 3070 Ti, or 1 hr rented]

Run existing `run_082_at_cifar10/experiment.py` with seeds 44, 45, 46 (in addition to existing 42, 43). Tightens confidence intervals. Currently 2 seeds is thin for a metric paper.

### 3. CIFAR-100 [~2 hours on 3070 Ti, or 45 min rented]

Identical code to run_082, just swap `CIFAR10` → `CIFAR100`. Second dataset in the paper. Also run pretrained models on CIFAR-100 (same resize trick).

---

## GPU-Rental-Only Experiments (require A100 40GB+, won't run on 3070 Ti)

These are out of reach locally but would push the paper from "competitive" to "strong accept."

### 4. ImageNet-1K Training from Scratch [~8–12 hrs on A100, ~$15–25 rented]

Train ResNet-50, ViT-B/16, MLP-Mixer-B, ConvNeXt-S from scratch on full ImageNet-1K (1.2M images). All models reach proper ImageNet accuracy (76–84%), completely eliminating any scale/quality objection. Then compute AT and CKA on ImageNet-1K validation set (50K images).

Why this matters: the current paper uses CIFAR-10 (32×32, 50K images). Reviewers at ICML/NeurIPS will always ask "does this hold at scale?" This answers that definitively.

Rough training times on A100 (80GB):
- ResNet-50: ~3 hrs (90 epochs)
- ViT-B/16: ~4 hrs (90 epochs with warmup)
- MLP-Mixer-B: ~3 hrs
- ConvNeXt-S: ~3 hrs
- **Total:** ~13 hrs → ~$20–$26 on Lambda A100

Models to add (beyond existing three):
- `convnext_small` (timm) — modern pure-CNN baseline, closes the "ResNet is old" objection
- `swin_small_patch4_window7_224` (Swin Transformer) — hierarchical attention, adds richer architecture diversity

### 5. Layer-Wise Analysis at Scale [~1 hr, can piggyback on Exp 4]

Once ImageNet models are trained (Exp 4), extract intermediate features at every block across the full 50K validation set and compute layer-wise AT. Produces a much smoother depth-vs-AT curve than the current 5-point version. This is essentially free once Exp 4 is done.

### 6. AT Ablation Study [~2 hrs on A100, ~$3 rented]

The reviewers will ask: how sensitive is AT to the histogram design choices? Run AT with:
- Bins: 10, 25 (current), 50, 100
- Range: [-2,2], [-3,3] (current), [-4,4]
- Metric: 1-Wasserstein (current) vs 2-Wasserstein vs KL divergence

Shows the metric is robust and not tuned. Critical for a paper introducing a new metric.

**Code to write:** `research/runs/run_091_at_ablation/experiment.py` — loops over hyperparameter grid on existing trained models (no re-training needed).

### 7. Broader Architecture Coverage [piggybacks on Exp 4, no extra training]

Once you have pretrained models (Exp 1 or Exp 4), add more architectures at zero extra training cost:
- `efficientnet_b3` — efficient CNN, different from ResNet
- `regnet_y_8gf` — RegNet, another CNN family
- `xcit_small_12_p16` — cross-covariance image transformer

Produces a richer pairwise AT matrix (6–8 architectures instead of 4) and a cleaner story about CNN vs attention families.

---

## What the Strengthened Paper Looks Like

| Experiment | Current | 3070 Ti only | With rented GPU |
|---|---|---|---|
| Training setup | Scratch, 2 seeds | Scratch, 5 seeds | Scratch, 5 seeds + ImageNet |
| Datasets | CIFAR-10 only | CIFAR-10 + CIFAR-100 | CIFAR-10 + CIFAR-100 + ImageNet-1K |
| Pretrained models | None | ResNet-50, ViT-B, DeiT-S, Mixer-B (inference only) | + ConvNeXt, Swin, EfficientNet |
| Architecture count | 4 | 4 | 8+ |
| AT ablation | None | None | Full hyperparameter sweep |
| Accuracy gap objection | Unaddressed | Eliminated (pretrained inference) | Eliminated (full training) |
| Review outcome | Workshop | ICML/NeurIPS competitive | Strong accept |

The pretrained inference experiment (Exp 1, free on your 3070 Ti) is still the single highest-leverage thing. If AT=0.25+ persists with well-trained pretrained models, the paper is already very strong before spending a dollar on rental.

---

## GPU Rental Cost Estimates

All costs as of early 2026. Cheapest providers first.

### Option A: Lambda Labs
- **GPU:** A10G (24GB VRAM)
- **Price:** ~$0.60/hr
- **Pretrained experiment (Exp 1):** 30-45 min → **~$0.40**
- **More seeds (Exp 2):** 1 hr → **~$0.60**
- **CIFAR-100 (Exp 3):** 45 min → **~$0.45**
- **Total all three:** ~2.5 hrs → **~$1.50**
- URL: lambda.ai

### Option B: RunPod
- **GPU:** A100 40GB (spot instance)
- **Price:** ~$1.50/hr spot, ~$2.00/hr on-demand
- **All three experiments combined:** ~1.5 hrs → **~$2.25–$3.00**
- URL: runpod.io

### Option C: Vast.ai
- **GPU:** A100 or 3090 (cheapest spot market)
- **Price:** ~$0.40–$0.80/hr depending on availability
- **All three experiments combined:** ~2 hrs → **~$0.80–$1.60**
- URL: vast.ai

### Option D: Google Colab Pro+
- **GPU:** A100 (when available)
- **Price:** ~$50/month flat (not per-hour)
- **Worth it if:** you need to run multiple experiments across multiple sessions

### Realistic Total Cost

| Scope | Time | Lambda Labs cost |
|---|---|---|
| Exps 1–3 only (3070 Ti feasible but faster rented) | ~2.5 hrs | ~$1.50 |
| Add ImageNet training (Exp 4) | +13 hrs | +~$22 |
| Add ablation (Exp 6) | +2 hrs | +~$3 |
| **Full suite (Exps 1–7)** | **~18 hrs** | **~$27 total** |

For a paper targeting ICML/NeurIPS, $27 in GPU rental to go from "workshop" to "strong accept" is an obvious trade.

---

## How to Tell Claude in the Next Session

**If on 3070 Ti:**
> "Look at `papers/activation_transport/NEXT_STEPS.md` and implement Experiments 1–3. Start with Experiment 1 (pretrained feature extraction). Write the code in `research/runs/run_090_pretrained_at/`."

**If on rented GPU (A100 recommended):**
> "Look at `papers/activation_transport/NEXT_STEPS.md` and implement all experiments including the GPU-rental-only ones (Exps 4–7). I'm on an A100. Start with Exp 4 (ImageNet training from scratch) and Exp 6 (ablation) in parallel."

Claude will have full context from this file.

---

## Paper Files
- LaTeX source: `papers/activation_transport/paper/main.tex`
- Compiled PDF: `papers/activation_transport/paper/final.pdf`
- References: `papers/activation_transport/paper/refs.bib`
- Existing experiment code: `research/runs/run_082_at_cifar10/experiment.py`
- Additional AT experiments: `research/runs/run_083_at_cifar100/`, `run_084_at_dynamics/`, `run_086_at_normalization/`
