"""Full-scale CUB Recall@1/2/4/8 + MAP@R on the standard test gallery (5,924
images, classes 101-200) -- the same protocol published metric-learning
papers use, so this is directly comparable to literature numbers (unlike the
sampled-triplet/pairwise-AUC metrics used throughout the adaptive-scorer
exploration this session).

Uses the validated two-stage approach for the patch component (global
shortlist top-K_SHORTLIST candidates, then exact top-K_PATCH MaxSim rerank --
matches brute force within noise, ~6x faster, already proven this session)
so this is tractable at full gallery scale. The learned gate combines
global + patch scores for the adaptive ranking.

Usage: python eval_cub_full_adaptive.py --cub-dir /path/CUB_200_2011 \
    --checkpoint checkpoints_cub_joint/backbone_final.pt --gate-path data/learned_gate_joint_recalibrated.pt
"""
import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

MODEL_NAME = "facebook/dinov2-base"
IMAGE_SIZE = 224
PATCH_K = 16
SHORTLIST_K = 100


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_cub_test(cub_dir):
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
    test_classes = [str(i) for i in range(101, 201)]
    images = []
    for c in test_classes:
        for p in by_class[c]:
            images.append({"class_id": c, "path": p})
    return images


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gate-path", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    gate_data = torch.load(args.gate_path, map_location=device)
    gate = Gate().to(device)
    gate.load_state_dict(gate_data["state_dict"])
    gate.eval()
    gate_mean = torch.tensor(gate_data["mean"], device=device, dtype=torch.float32)
    gate_std = torch.tensor(gate_data["std"], device=device, dtype=torch.float32)

    print("Loading CUB test set...")
    test_images = load_cub_test(args.cub_dir)
    N = len(test_images)
    print(f"Found {N} test images.")
    images_root = os.path.join(args.cub_dir, "images")
    paths = [img["path"] for img in test_images]
    classes = [img["class_id"] for img in test_images]
    classes_t = torch.tensor([hash(c) for c in classes], device=device)

    print("Encoding all test images (global + patches)...")
    all_global = torch.zeros((N, 768), dtype=torch.float32)
    all_patches = torch.zeros((N, 256, 768), dtype=torch.bfloat16)
    for i in tqdm(range(0, N, args.batch_size)):
        batch_paths = paths[i:i + args.batch_size]
        imgs = [Image.open(os.path.join(images_root, p)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                for p in batch_paths]
        pixel_values = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(pixel_values=pixel_values)
        last_hidden = out.last_hidden_state
        cls = F.normalize(last_hidden[:, 0, :], dim=-1)
        patches = F.normalize(last_hidden[:, 1:, :], dim=-1)
        all_global[i:i + len(batch_paths)] = cls.float().cpu()
        all_patches[i:i + len(batch_paths)] = patches.cpu()

    all_global_gpu = all_global.to(device)
    print("Computing per-image patch diversity (batched)...")
    diversity = torch.zeros(N, device=device)
    div_chunk = 256
    for c0 in range(0, N, div_chunk):
        c1 = min(N, c0 + div_chunk)
        p = all_patches[c0:c1].to(device).float()  # (B, 256, D)
        sim = torch.bmm(p, p.transpose(1, 2))  # (B, 256, 256)
        n = sim.shape[-1]
        off_diag_sum = sim.sum(dim=(1, 2)) - torch.diagonal(sim, dim1=1, dim2=2).sum(dim=1)
        diversity[c0:c1] = off_diag_sum / (n * (n - 1))

    print("Stage 1: global similarity + top-K shortlist...")
    global_sims_full = all_global_gpu @ all_global_gpu.T  # (N,N), cheap
    global_sims_full.fill_diagonal_(-1e9)
    shortlist_vals, shortlist_idx = global_sims_full.topk(SHORTLIST_K, dim=1)  # (N, SHORTLIST_K)

    print("Stage 2: exact top-K patch MaxSim rerank on shortlist...")
    patch_scores = torch.zeros((N, SHORTLIST_K), device=device)
    query_chunk = 8
    for q0 in tqdm(range(0, N, query_chunk)):
        q1 = min(N, q0 + query_chunk)
        q_patches = all_patches[q0:q1].to(device).float()  # (Qb, P, D)
        cand_idx = shortlist_idx[q0:q1].cpu()  # (Qb, K)
        cand_patches = all_patches[cand_idx].to(device).float()  # (Qb, K, P, D)
        Qb = q1 - q0
        P = q_patches.shape[1]
        cand_flat = cand_patches.reshape(Qb, -1, cand_patches.shape[-1])
        sim_flat = torch.bmm(q_patches, cand_flat.transpose(1, 2))  # (Qb, P, K*P)
        sim2 = sim_flat.view(Qb, P, SHORTLIST_K, P).permute(0, 2, 1, 3)  # (Qb,K,P,P)
        k_eff = min(PATCH_K, P)
        a_to_b = sim2.max(dim=3).values.topk(k_eff, dim=2).values.mean(dim=2)
        b_to_a = sim2.max(dim=2).values.topk(k_eff, dim=2).values.mean(dim=2)
        patch_scores[q0:q1] = 0.5 * (a_to_b + b_to_a)

    print("Combining via learned gate...")
    div_expand = diversity.unsqueeze(1).expand(-1, SHORTLIST_K)  # (N, K)
    features = torch.stack([shortlist_vals, patch_scores, div_expand], dim=-1)  # (N,K,3)
    features_n = (features - gate_mean) / gate_std
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        gate_logits = gate(features_n.reshape(-1, 3).float()).reshape(N, SHORTLIST_K)
    adaptive_scores = torch.sigmoid(gate_logits)

    def compute_metrics(scores, cand_idx):
        cand_classes = classes_t[cand_idx]  # (N,K)
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
        map_at_r = ap.mean().item()
        return recall_at, map_at_r

    global_recall, global_map = compute_metrics(shortlist_vals, shortlist_idx)
    patch_recall, patch_map = compute_metrics(patch_scores, shortlist_idx)
    adaptive_recall, adaptive_map = compute_metrics(adaptive_scores, shortlist_idx)

    print("\n" + "=" * 70)
    print(f"FULL CUB TEST GALLERY (N={N}), standard protocol -- comparable to literature")
    print("=" * 70)
    print(f"Published SOTA reference: Proxy-Anchor R@1=71.1%, MS R@1=65.7%, SoftTriple R@1=65.4%")
    for name, recall, map_r in [("global-only", global_recall, global_map),
                                 ("patch-only (top-K rerank)", patch_recall, patch_map),
                                 ("adaptive (learned gate)", adaptive_recall, adaptive_map)]:
        print(f"\n  {name}:")
        for r in (1, 2, 4, 8):
            print(f"    R@{r}: {recall[r]:.4f}")
        print(f"    MAP@R: {map_r:.4f}")

    out = {
        "test_size": N,
        "global": {"recall": global_recall, "map_at_r": global_map},
        "patch": {"recall": patch_recall, "map_at_r": patch_map},
        "adaptive": {"recall": adaptive_recall, "map_at_r": adaptive_map},
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "cub_full_adaptive_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
