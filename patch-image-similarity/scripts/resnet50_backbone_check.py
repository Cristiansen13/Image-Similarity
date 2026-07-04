"""Short test: does patch-level MaxSim close (or reverse) the gap to the
standard global-embedding recipe on a WEAKER backbone than DINOv2?

Motivation: the backbone-controlled comparison on DINOv2 found the standard
global recipe beats our patch-proxy architecture by ~3.6pt on CUB. Hypothesis:
DINOv2's own pretraining already optimizes dense patch-consistency alongside
its global CLS objective, making its CLS token unusually strong -- an older,
purely globally-supervised backbone (ResNet50/ImageNet, closer to what the
original metric-learning papers used) might not have that advantage baked in,
so patch-level matching could recover more relative value there.

Both variants share the same frozen stem (conv1..layer3), fine-tune only
layer4 (the ResNet analogue of "unfreeze last N transformer blocks"), same
augment+cosine-schedule, same proxy-loss family (M=8 prototype-token proxies
for patch; single proxy vector for global) as the DINOv2 comparison, so only
the backbone+representation differs. This is an exploratory, short test (1-2
seeds), not the full multi-seed rigor of the main suite -- meant to check
DIRECTION of the effect, not pin down precise numbers.

Patch representation: ResNet50's layer4 output (B, 2048, 7, 7) treated as 49
patch tokens of dim 2048 (the conv-net analogue of ViT patch tokens).
Global representation: standard global-average-pooled 2048-dim vector.

Usage:
  python resnet50_backbone_check.py --cub-dir /path/CUB_200_2011 --mode patch --seed 0
  python resnet50_backbone_check.py --cub-dir /path/CUB_200_2011 --mode global --seed 0
"""
import argparse
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms.functional as TF
from PIL import Image

IMAGE_SIZE = 224
AUG_SIZE = 256
HIDDEN_D = 2048
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ResNet50Backbone(nn.Module):
    """Shared stem, only layer4 trainable -- the ResNet analogue of
    "unfreeze last N transformer blocks" used throughout this project."""
    def __init__(self, mode):
        super().__init__()
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                   base.layer1, base.layer2, base.layer3)
        self.layer4 = base.layer4
        for p in self.stem.parameters():
            p.requires_grad = False
        self.mode = mode

    def forward(self, pixel_values):
        feat = self.stem(pixel_values)
        feat = self.layer4(feat)  # (B, 2048, 7, 7)
        if self.mode == "patch":
            B, C, H, W = feat.shape
            patches = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 49, 2048)
            return F.normalize(patches, dim=-1)
        else:
            pooled = feat.mean(dim=(2, 3))  # (B, 2048), standard global-avg-pool
            return F.normalize(pooled, dim=-1)


def image_to_proxy_maxsim(patches, proxies):
    sim = torch.einsum("bnd,cmd->bcnm", patches, proxies)
    img_to_proxy = sim.max(dim=3).values.mean(dim=2)
    proxy_to_img = sim.max(dim=2).values.mean(dim=2)
    return 0.5 * (img_to_proxy + proxy_to_img)


def proxy_anchor_loss(scores, label_idx, alpha, delta):
    B, C = scores.shape
    device = scores.device
    pos_mask = torch.zeros(B, C, device=device, dtype=torch.bool)
    pos_mask[torch.arange(B, device=device), label_idx] = True
    neg_mask = ~pos_mask
    with_pos = pos_mask.any(dim=0)
    num_pos = with_pos.sum().clamp(min=1)

    def masked_lse_with_zero(z, mask):
        z = z.masked_fill(~mask, float("-inf"))
        zeros = torch.zeros(1, z.shape[1], device=z.device, dtype=z.dtype)
        return torch.logsumexp(torch.cat([z, zeros], dim=0), dim=0)

    pos_term = masked_lse_with_zero(-alpha * (scores - delta), pos_mask)
    pos_loss = (pos_term * with_pos).sum() / num_pos
    neg_term = masked_lse_with_zero(alpha * (scores + delta), neg_mask)
    neg_loss = neg_term.sum() / C
    return pos_loss + neg_loss


def load_cub_index(cub_dir):
    import os
    by_class = defaultdict(list)
    image_paths = {}
    with open(f"{cub_dir}/images.txt") as f:
        for line in f:
            image_id, path = line.split()
            image_paths[image_id] = path
    with open(f"{cub_dir}/image_class_labels.txt") as f:
        for line in f:
            image_id, class_id = line.split()
            by_class[class_id].append(image_paths[image_id])
    return by_class


def load_image(path, images_root, rng, augment):
    import os
    img = Image.open(os.path.join(images_root, path)).convert("RGB")
    if augment:
        img = img.resize((AUG_SIZE, AUG_SIZE))
        off = AUG_SIZE - IMAGE_SIZE
        x, y = rng.randint(0, off), rng.randint(0, off)
        img = img.crop((x, y, x + IMAGE_SIZE, y + IMAGE_SIZE))
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    return img


def to_pixel_values(images, device):
    """ResNet50 needs standard ImageNet normalization (no HF processor here)."""
    batch = torch.stack([TF.to_tensor(img) for img in images])  # (B,3,H,W) in [0,1]
    return (batch.to(device) - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


def sample_batch(by_class, classes, images_root, P, K, rng, augment, device):
    chosen = rng.sample(classes, P)
    paths, labels = [], []
    for c in chosen:
        pool = by_class[c]
        imgs = rng.sample(pool, min(K, len(pool)))
        paths.extend(imgs)
        labels.extend([c] * len(imgs))
    images = [load_image(p, images_root, rng, augment) for p in paths]
    return to_pixel_values(images, device), labels


@torch.no_grad()
def evaluate_full(model, cub_dir, device, mode, batch_size=64):
    import os
    by_class = load_cub_index(cub_dir)
    test_classes = [str(i) for i in range(101, 201)]
    images_root = f"{cub_dir}/images"
    paths, classes = [], []
    for c in test_classes:
        for p in by_class[c]:
            paths.append(p)
            classes.append(c)
    N = len(paths)

    model.eval()
    embeds_list = []
    for i in range(0, N, batch_size):
        batch_paths = paths[i:i + batch_size]
        imgs = [Image.open(os.path.join(images_root, p)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                for p in batch_paths]
        pixel_values = to_pixel_values(imgs, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(pixel_values)
        embeds_list.append(out.float().cpu())
    model.train()
    all_embeds = torch.cat(embeds_list, dim=0).to(device)  # (N, D) or (N, 49, D)

    classes_t = torch.tensor([hash(c) for c in classes], device=device)
    class_counts = defaultdict(int)
    for c in classes:
        class_counts[c] += 1
    R_per_query = torch.tensor([class_counts[c] - 1 for c in classes], device=device).clamp(min=1, max=N - 1)
    K = min(100, N - 1)

    if mode == "global":
        sims = all_embeds @ all_embeds.T
        sims.fill_diagonal_(-1e9)
        _, ranked_idx = sims.topk(K, dim=1)
    else:
        # chunked patch MaxSim (N~5924, 49 patches -- much cheaper than DINOv2's 256)
        query_chunk, cand_chunk = 16, 2048
        best = torch.full((N, K), -1, dtype=torch.long, device=device)
        best_scores = torch.full((N, K), -1e9, device=device)
        for c0 in range(0, N, cand_chunk):
            c1 = min(N, c0 + cand_chunk)
            c_embs = all_embeds[c0:c1]
            for q0 in range(0, N, query_chunk):
                q1 = min(N, q0 + query_chunk)
                q_embs = all_embeds[q0:q1]
                sim = torch.einsum("qpd,crd->qcpr", q_embs, c_embs)
                a_to_b = sim.max(dim=3).values.mean(dim=2)
                b_to_a = sim.max(dim=2).values.mean(dim=2)
                scores = 0.5 * (a_to_b + b_to_a)
                for i in range(q1 - q0):
                    gq = q0 + i
                    if c0 <= gq < c1:
                        scores[i, gq - c0] = -1e9
                cur_k = min(K, scores.shape[1])
                top_vals, top_idx = scores.topk(cur_k, dim=1)
                merged_vals = torch.cat([best_scores[q0:q1], top_vals], dim=1)
                merged_idx = torch.cat([best[q0:q1], top_idx + c0], dim=1)
                sel_vals, sel_pos = merged_vals.topk(K, dim=1)
                best_scores[q0:q1] = sel_vals
                best[q0:q1] = torch.gather(merged_idx, 1, sel_pos)
        ranked_idx = best

    ranked_classes = classes_t[ranked_idx]
    query_classes = classes_t.unsqueeze(1)
    relevant = (ranked_classes == query_classes).float()

    recall_at = {}
    for r in (1, 2, 4, 8):
        recall_at[r] = relevant[:, :r].any(dim=1).float().mean().item()

    cum_relevant = relevant.cumsum(dim=1)
    ranks = torch.arange(1, K + 1, device=device, dtype=torch.float32).unsqueeze(0)
    precision_at_rank = cum_relevant / ranks
    R_batch = R_per_query.clamp(max=K)
    rank_mask = (torch.arange(1, K + 1, device=device).unsqueeze(0) <= R_batch.unsqueeze(1)).float()
    ap = (precision_at_rank * relevant * rank_mask).sum(dim=1) / R_batch.float()
    map_at_r = ap.mean().item()

    return {"recall_at_1": recall_at[1], "recall_at_2": recall_at[2],
            "recall_at_4": recall_at[4], "recall_at_8": recall_at[8], "map_at_r": map_at_r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-dir", required=True)
    ap.add_argument("--mode", choices=["patch", "global"], required=True)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--proxy-lr", type=float, default=5e-2)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    by_class = load_cub_index(args.cub_dir)
    train_classes = [str(i) for i in range(1, 101)]
    class_to_idx = {c: i for i, c in enumerate(train_classes)}
    num_classes = len(train_classes)
    images_root = f"{args.cub_dir}/images"

    model = ResNet50Backbone(args.mode).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Mode={args.mode}  trainable params={sum(p.numel() for p in trainable):,}")

    if args.mode == "patch":
        proxies = nn.Parameter(torch.randn(num_classes, args.M, HIDDEN_D, device=device))
    else:
        proxies = nn.Parameter(torch.randn(num_classes, HIDDEN_D, device=device))

    opt = torch.optim.AdamW([
        {"params": trainable, "lr": args.lr, "weight_decay": 1e-4},
        {"params": [proxies], "lr": args.proxy_lr, "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    print(f"Training resnet50-{args.mode}: P={args.P} K={args.K} steps={args.steps} seed={args.seed}")
    t0 = time.time()
    running_loss = 0.0
    for step in range(1, args.steps + 1):
        pixel_values, labels = sample_batch(by_class, train_classes, images_root, args.P, args.K,
                                             rng, True, device)
        label_idx = torch.tensor([class_to_idx[c] for c in labels], device=device)

        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeds = model(pixel_values)
            proxies_norm = F.normalize(proxies, dim=-1).to(embeds.dtype)
        if args.mode == "patch":
            scores = image_to_proxy_maxsim(embeds.float(), proxies_norm.float())
        else:
            scores = embeds.float() @ proxies_norm.float().T
        loss = proxy_anchor_loss(scores, label_idx, args.alpha, args.delta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable + [proxies], 1.0)
        opt.step()
        scheduler.step()

        running_loss += loss.item()
        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{args.steps}  loss={running_loss / 50:.4f}  ({elapsed:.0f}s)")
            running_loss = 0.0

    print("Evaluating...")
    metrics = evaluate_full(model, args.cub_dir, device, args.mode)
    print(f"RESULT mode={args.mode} seed={args.seed}: {metrics}")


if __name__ == "__main__":
    main()
