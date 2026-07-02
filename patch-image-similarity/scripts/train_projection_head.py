"""Train a lightweight linear projection on top of frozen DINOv2 patch
embeddings, via a differentiable MaxSim triplet loss, then evaluate on a
genuinely held-out test split (disjoint product classes, never touched
during training or threshold selection).

Everything here operates on the patch embeddings cached by
make_sop_train_test_embeddings.py -- no DINOv2 forward passes during
training, so iteration is fast even on CPU.
"""
import json
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from maxsim import symmetric_maxsim, threshold_hit_rate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(ROOT, "data", os.environ.get("SOP_SPLIT_NAME", "sop_split"))
IN_DIM = 768
OUT_DIM = 128
EPOCHS = int(os.environ.get("HEAD_EPOCHS", "300"))
BATCH_SIZE = int(os.environ.get("HEAD_BATCH", "0"))  # 0 = full batch (original behavior)
HEAD_TYPE = os.environ.get("HEAD_TYPE", "linear")  # linear | mlp
LR = 2e-3
WEIGHT_DECAY = 1e-3
MARGIN = 0.15
VAL_FRACTION = 0.2
SEED = 0
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(12)]  # 0.30..0.85


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=IN_DIM, out_dim=OUT_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), dim=-1)


class MLPHead(nn.Module):
    """Two-layer head with residual-style bottleneck + dropout, for the
    scaled-up training runs where a linear probe may underfit."""

    def __init__(self, in_dim=IN_DIM, out_dim=OUT_DIM, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim, bias=False),
        )

    def forward(self, x):
        return torch.nn.functional.normalize(self.net(x), dim=-1)


def make_head():
    return MLPHead() if HEAD_TYPE == "mlp" else ProjectionHead()


def differentiable_score(a, b):
    """Same math as symmetric_maxsim, kept inline so autograd tracks it."""
    sim = a @ b.T
    return 0.5 * (sim.max(dim=1).values.mean() + sim.max(dim=0).values.mean())


def project_all(head, embeddings, paths):
    if head is None:
        return {p: embeddings[p] for p in paths}
    return {p: head(embeddings[p]) for p in paths}


def score_triplets(triplets, embeddings, scorer):
    correct = 0
    for t in triplets:
        pos = scorer(embeddings[t["anchor"]], embeddings[t["positive"]])
        neg = scorer(embeddings[t["anchor"]], embeddings[t["negative"]])
        if pos > neg:
            correct += 1
    return correct / len(triplets)


def meanmax_scorer(a, b):
    return symmetric_maxsim(a, b).similarity


def best_threshold_loo(triplets, embeddings):
    """Pick the threshold that maximizes leave-one-out accuracy, using only
    the given triplets (call with TRAIN triplets only)."""
    n = len(triplets)
    correct = torch.zeros(len(THRESHOLDS), n, dtype=torch.bool)
    for i, t in enumerate(triplets):
        a, p, neg = embeddings[t["anchor"]], embeddings[t["positive"]], embeddings[t["negative"]]
        for ti, thresh in enumerate(THRESHOLDS):
            correct[ti, i] = threshold_hit_rate(a, p, thresh) > threshold_hit_rate(a, neg, thresh)

    chosen = []
    for i in range(n):
        mask = torch.ones(n, dtype=torch.bool)
        mask[i] = False
        best_ti = int(correct[:, mask].float().mean(dim=1).argmax())
        chosen.append(THRESHOLDS[best_ti])
    return max(set(chosen), key=chosen.count), chosen


def _triplet_batch_loss(head, batch, raw_embeddings):
    # project each unique image in the batch once, reuse across triplets
    unique_paths = sorted({t[k] for t in batch for k in ("anchor", "positive", "negative")})
    projected = {p: head(raw_embeddings[p]) for p in unique_paths}
    losses = []
    for t in batch:
        pos_score = differentiable_score(projected[t["anchor"]], projected[t["positive"]])
        neg_score = differentiable_score(projected[t["anchor"]], projected[t["negative"]])
        losses.append(torch.relu(torch.tensor(MARGIN) + neg_score - pos_score))
    return torch.stack(losses).mean()


def train(head, train_triplets, val_triplets, raw_embeddings):
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val_acc, best_state = -1.0, None
    rng = random.Random(SEED)
    batch_size = BATCH_SIZE if BATCH_SIZE > 0 else len(train_triplets)

    for epoch in range(EPOCHS):
        order = train_triplets[:]
        rng.shuffle(order)
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            opt.zero_grad()
            loss = _triplet_batch_loss(head, batch, raw_embeddings)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        mean_loss = sum(epoch_losses) / len(epoch_losses)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            head.eval()
            with torch.no_grad():
                proj_val = {p: head(raw_embeddings[p]) for t in val_triplets for p in
                            (t["anchor"], t["positive"], t["negative"])}
            val_acc = score_triplets(val_triplets, proj_val, meanmax_scorer)
            head.train()
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
            if epoch % 50 == 0 or (EPOCHS <= 60 and epoch % 10 == 0):
                print(f"  epoch {epoch:4d}  loss={mean_loss:.4f}  val_acc={val_acc:.3f}")

    head.load_state_dict(best_state)
    head.eval()
    return head, best_val_acc


def main():
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    with open(os.path.join(SPLIT_DIR, "train_triplets.json")) as f:
        all_train_triplets = json.load(f)
    with open(os.path.join(SPLIT_DIR, "test_triplets.json")) as f:
        test_triplets = json.load(f)
    raw_embeddings = torch.load(os.path.join(SPLIT_DIR, "patch_embeddings.pt"))

    rng.shuffle(all_train_triplets)
    n_val = max(1, int(len(all_train_triplets) * VAL_FRACTION))
    val_triplets, train_triplets = all_train_triplets[:n_val], all_train_triplets[n_val:]
    print(f"{len(train_triplets)} train / {len(val_triplets)} internal-val / {len(test_triplets)} held-out test triplets")

    print("\n--- Zero-shot baseline (raw DINOv2, no projection) ---")
    zs_meanmax_train = score_triplets(all_train_triplets, raw_embeddings, meanmax_scorer)
    zs_meanmax_test = score_triplets(test_triplets, raw_embeddings, meanmax_scorer)
    zs_thresh, _ = best_threshold_loo(all_train_triplets, raw_embeddings)
    zs_thresh_test = score_triplets(test_triplets, raw_embeddings,
                                     lambda a, b: threshold_hit_rate(a, b, zs_thresh))
    print(f"zero-shot meanmax:            train={zs_meanmax_train:.3f}  test={zs_meanmax_test:.3f}")
    print(f"zero-shot threshold(t={zs_thresh}): train->test  test={zs_thresh_test:.3f}")

    print(f"\n--- Training projection head (type={HEAD_TYPE}, epochs={EPOCHS}, batch={BATCH_SIZE or 'full'}) ---")
    head = make_head()
    head, best_val_acc = train(head, train_triplets, val_triplets, raw_embeddings)
    print(f"Best internal val_acc during training: {best_val_acc:.3f}")

    with torch.no_grad():
        all_paths = sorted({t[k] for triplets in (all_train_triplets, test_triplets)
                             for t in triplets for k in ("anchor", "positive", "negative")})
        projected = {p: head(raw_embeddings[p]) for p in all_paths}

    print("\n--- Trained projection, evaluated on held-out test ---")
    tr_meanmax_train = score_triplets(all_train_triplets, projected, meanmax_scorer)
    tr_meanmax_test = score_triplets(test_triplets, projected, meanmax_scorer)
    tr_thresh, _ = best_threshold_loo(all_train_triplets, projected)
    tr_thresh_test = score_triplets(test_triplets, projected,
                                     lambda a, b: threshold_hit_rate(a, b, tr_thresh))
    print(f"trained meanmax:               train={tr_meanmax_train:.3f}  test={tr_meanmax_test:.3f}")
    print(f"trained threshold(t={tr_thresh}):     train->test  test={tr_thresh_test:.3f}")

    print("\n=== SUMMARY (all numbers on the same held-out test set, never used for training or threshold selection) ===")
    print(f"{'method':40s}{'test accuracy':>15s}")
    print("-" * 55)
    print(f"{'zero-shot mean-of-max':40s}{zs_meanmax_test:15.3f}")
    print(f"{'zero-shot threshold hit-rate':40s}{zs_thresh_test:15.3f}")
    print(f"{'trained-head mean-of-max':40s}{tr_meanmax_test:15.3f}")
    print(f"{'trained-head threshold hit-rate':40s}{tr_thresh_test:15.3f}")

    head_path = os.path.join(SPLIT_DIR, "projection_head.pt")
    torch.save({"state_dict": head.state_dict(), "in_dim": IN_DIM, "out_dim": OUT_DIM,
                "head_type": HEAD_TYPE}, head_path)
    print(f"Saved trained projection head to {head_path}")

    out_path = os.path.join(SPLIT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_train": len(train_triplets), "n_val": len(val_triplets), "n_test": len(test_triplets),
            "zero_shot_meanmax_test": zs_meanmax_test,
            "zero_shot_threshold_test": zs_thresh_test, "zero_shot_threshold": zs_thresh,
            "trained_meanmax_test": tr_meanmax_test,
            "trained_threshold_test": tr_thresh_test, "trained_threshold": tr_thresh,
        }, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
