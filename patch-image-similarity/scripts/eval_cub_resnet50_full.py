"""Backbone-matched comparison: train a gate + run full-scale CUB Recall@1
evaluation on the ResNet50 joint-trained checkpoint (finetune_cub_joint_resnet50.py)
-- same protocol as eval_cub_full_adaptive.py, but for a backbone in the same
tier as what Proxy-Anchor/MS/SoftTriple actually used, so this IS a valid,
backbone-controlled comparison against those published numbers (unlike the
DINOv2 comparison, which is confounded by backbone strength).

Gate is trained fresh on THIS backbone's own score distributions (the
gate-recalibration lesson from earlier this session: never reuse a gate
across different backbones/checkpoints).

Usage: python eval_cub_resnet50_full.py --cub-dir /path/CUB_200_2011 \
    --checkpoint checkpoints_cub_joint_resnet50/model_final.pt
"""
import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

IMAGE_SIZE = 224
PATCH_K = 4
SHORTLIST_K = 100
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ResNet50JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = tvm.resnet50(weights=None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                   base.layer1, base.layer2, base.layer3)
        self.layer4 = base.layer4

    def forward(self, pixel_values):
        feat = self.stem(pixel_values)
        feat = self.layer4(feat)
        B, C, H, W = feat.shape
        patches = F.normalize(feat.permute(0, 2, 3, 1).reshape(B, H * W, C), dim=-1)
        global_emb = F.normalize(feat.mean(dim=(2, 3)), dim=-1)
        return global_emb, patches


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def to_pixel_values(images, device):
    batch = torch.stack([TF.to_tensor(img) for img in images])
    return (batch.to(device) - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


def load_cub_index(cub_dir):
    by_class = defaultdict(list)
    image_paths = {}
    with open(os.path.join(cub_dir, "images.txt")) as f:
        for line in f:
            image_id, path = line.split()
            image_paths[image_id] = path
    with open(os.path.join(cub_dir, "image_class_labels.txt")) as f:
        for line in f:
            image_id, class_id = line.split()
            by_class[class_id].append(image_paths[image_id])
    return by_class


def build_cub_pairs(cub_dir, classes, n, seed):
    rng = random.Random(seed)
    by_class = load_cub_index(cub_dir)
    images_root = os.path.join(cub_dir, "images")
    pairs = []
    for _ in range(n):
        pos_class = rng.choice(classes)
        neg_class = rng.choice([c for c in classes if c != pos_class])
        if len(by_class[pos_class]) < 2:
            continue
        anchor, positive = rng.sample(by_class[pos_class], 2)
        negative = rng.choice(by_class[neg_class])
        a = os.path.join(images_root, anchor)
        pairs.append((a, os.path.join(images_root, positive), 1))
        pairs.append((a, os.path.join(images_root, negative), 0))
    return pairs


def mean_patch_similarity(patches):
    sim = patches @ patches.T
    n = sim.shape[0]
    return ((sim.sum() - torch.diagonal(sim).sum()) / (n * (n - 1))).item()


def topk_maxsim(a, b, k):
    sim = a @ b.T
    max_a, max_b = sim.max(dim=1).values, sim.max(dim=0).values
    ka, kb = min(k, max_a.shape[0]), min(k, max_b.shape[0])
    return 0.5 * (max_a.topk(ka).values.mean().item() + max_b.topk(kb).values.mean().item())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-gate-train", type=int, default=300)
    ap.add_argument("--gate-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet50JointModel().to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    def encode_pil(images):
        pv = to_pixel_values(images, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            g, p = model(pv)
        return g.float(), p.float()

    def encode_path(path):
        img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        g, p = encode_pil([img])
        return g[0], p[0]

    # ---- Train gate on CUB classes 1-100 ----
    print("Building + featurizing gate training pairs (CUB classes 1-100)...")
    train_pairs = build_cub_pairs(args.cub_dir, [str(i) for i in range(1, 101)], args.n_gate_train, 0)
    cache = {}
    X, y = [], []
    for i, (a, b, label) in enumerate(train_pairs):
        if a not in cache:
            cache[a] = encode_path(a)
        if b not in cache:
            cache[b] = encode_path(b)
        ga, pa = cache[a]
        gb, pb = cache[b]
        g = (ga @ gb).item()
        div_a = mean_patch_similarity(pa)
        v = topk_maxsim(pa, pb, PATCH_K)
        X.append([g, v, div_a]); y.append(label)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(train_pairs)}")
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    print(f"Feature mean={mean} std={std}")

    gate = Gate().to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=1e-2, weight_decay=1e-4)
    Xn = (X - mean) / std
    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    for step in range(args.gate_steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(gate(Xt), yt)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"  gate step {step + 1}/{args.gate_steps} loss={loss.item():.4f}")
    gate.eval()
    gate_mean = torch.tensor(mean, device=device, dtype=torch.float32)
    gate_std = torch.tensor(std, device=device, dtype=torch.float32)

    # ---- Full-scale eval on CUB test gallery (classes 101-200) ----
    print("\nLoading CUB test set (classes 101-200)...")
    by_class = load_cub_index(args.cub_dir)
    test_classes = [str(i) for i in range(101, 201)]
    images_root = os.path.join(args.cub_dir, "images")
    paths, classes = [], []
    for c in test_classes:
        for p in by_class[c]:
            paths.append(p); classes.append(c)
    N = len(paths)
    print(f"Found {N} test images.")
    classes_t = torch.tensor([hash(c) for c in classes], device=device)

    print("Encoding all test images...")
    all_global = torch.zeros((N, 2048), dtype=torch.float32)
    all_patches = torch.zeros((N, 49, 2048), dtype=torch.bfloat16)
    for i in tqdm(range(0, N, args.batch_size)):
        batch_paths = paths[i:i + args.batch_size]
        imgs = [Image.open(os.path.join(images_root, p)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                for p in batch_paths]
        g, p = encode_pil(imgs)
        all_global[i:i + len(batch_paths)] = g.cpu()
        all_patches[i:i + len(batch_paths)] = p.to(torch.bfloat16).cpu()

    all_global_gpu = all_global.to(device)
    print("Computing per-image patch diversity...")
    diversity = torch.zeros(N, device=device)
    for c0 in range(0, N, 512):
        c1 = min(N, c0 + 512)
        p = all_patches[c0:c1].to(device).float()
        sim = torch.bmm(p, p.transpose(1, 2))
        n = sim.shape[-1]
        off_diag = sim.sum(dim=(1, 2)) - torch.diagonal(sim, dim1=1, dim2=2).sum(dim=1)
        diversity[c0:c1] = off_diag / (n * (n - 1))

    print("Stage 1: global shortlist...")
    global_sims = all_global_gpu @ all_global_gpu.T
    global_sims.fill_diagonal_(-1e9)
    shortlist_vals, shortlist_idx = global_sims.topk(SHORTLIST_K, dim=1)

    print("Stage 2: patch top-K rerank...")
    patch_scores = torch.zeros((N, SHORTLIST_K), device=device)
    query_chunk = 16
    P = all_patches.shape[1]
    for q0 in tqdm(range(0, N, query_chunk)):
        q1 = min(N, q0 + query_chunk)
        Qb = q1 - q0
        q_patches = all_patches[q0:q1].to(device).float()
        cand_idx = shortlist_idx[q0:q1].cpu()
        cand_patches = all_patches[cand_idx].to(device).float()
        cand_flat = cand_patches.reshape(Qb, -1, cand_patches.shape[-1])
        sim_flat = torch.bmm(q_patches, cand_flat.transpose(1, 2))
        sim2 = sim_flat.view(Qb, P, SHORTLIST_K, P).permute(0, 2, 1, 3)
        k_eff = min(PATCH_K, P)
        a_to_b = sim2.max(dim=3).values.topk(k_eff, dim=2).values.mean(dim=2)
        b_to_a = sim2.max(dim=2).values.topk(k_eff, dim=2).values.mean(dim=2)
        patch_scores[q0:q1] = 0.5 * (a_to_b + b_to_a)

    print("Combining via learned gate...")
    div_expand = diversity.unsqueeze(1).expand(-1, SHORTLIST_K)
    features = torch.stack([shortlist_vals, patch_scores, div_expand], dim=-1)
    features_n = (features - gate_mean) / gate_std
    adaptive_scores = torch.sigmoid(gate(features_n.reshape(-1, 3).float()).reshape(N, SHORTLIST_K))

    def compute_metrics(scores, cand_idx):
        cand_classes = classes_t[cand_idx]
        query_classes = classes_t.unsqueeze(1)
        ranked_order = scores.argsort(dim=1, descending=True)
        ranked_classes = torch.gather(cand_classes, 1, ranked_order)
        relevant = (ranked_classes == query_classes).float()
        recall_at = {r: relevant[:, :r].any(dim=1).float().mean().item() for r in (1, 2, 4, 8)}
        class_counts = defaultdict(int)
        for c in classes:
            class_counts[c] += 1
        R_per_query = torch.tensor([class_counts[c] - 1 for c in classes], device=device).clamp(min=1, max=SHORTLIST_K)
        cum_relevant = relevant.cumsum(dim=1)
        ranks = torch.arange(1, SHORTLIST_K + 1, device=device, dtype=torch.float32).unsqueeze(0)
        precision_at_rank = cum_relevant / ranks
        rank_mask = (torch.arange(1, SHORTLIST_K + 1, device=device).unsqueeze(0) <= R_per_query.unsqueeze(1)).float()
        ap = (precision_at_rank * relevant * rank_mask).sum(dim=1) / R_per_query.float()
        return recall_at, ap.mean().item()

    global_recall, global_map = compute_metrics(shortlist_vals, shortlist_idx)
    patch_recall, patch_map = compute_metrics(patch_scores, shortlist_idx)
    adaptive_recall, adaptive_map = compute_metrics(adaptive_scores, shortlist_idx)

    print("\n" + "=" * 70)
    print(f"FULL CUB TEST GALLERY (N={N}), ResNet50 backbone -- SAME TIER AS PUBLISHED SOTA")
    print("=" * 70)
    print("Published (BN-Inception): SoftTriple R@1=65.4%, MS R@1=65.7%, Proxy-Anchor R@1=71.1%")
    for name, recall, map_r in [("global-only", global_recall, global_map),
                                 ("patch-only (top-K rerank)", patch_recall, patch_map),
                                 ("adaptive (learned gate)", adaptive_recall, adaptive_map)]:
        print(f"\n  {name}:")
        for r in (1, 2, 4, 8):
            print(f"    R@{r}: {recall[r]:.4f}")
        print(f"    MAP@R: {map_r:.4f}")

    out = {"test_size": N,
           "global": {"recall": global_recall, "map_at_r": global_map},
           "patch": {"recall": patch_recall, "map_at_r": patch_map},
           "adaptive": {"recall": adaptive_recall, "map_at_r": adaptive_map}}
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "cub_resnet50_full_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
