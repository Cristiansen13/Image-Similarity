import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

MODEL_NAME = "facebook/dinov2-base"
IMAGE_SIZE = 224

class FineTuneModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        return F.normalize(out.last_hidden_state[:, 1:, :], dim=-1)

def load_ebay_test(ebay_test_path):
    # Returns a list of dicts: {"class_id": ..., "path": ...}
    images = []
    with open(ebay_test_path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                _, class_id, _, path = parts
                images.append({"class_id": class_id, "path": path})
    return images

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebay-test", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading SOP test set from {args.ebay_test}...")
    test_images = load_ebay_test(args.ebay_test)
    print(f"Found {len(test_images)} test images.")
    
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    
    model = FineTuneModel().to(device)
    if args.checkpoint != "zero-shot":
        print(f"Loading checkpoint {args.checkpoint}...")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.backbone.load_state_dict(state_dict)
    else:
        print("Evaluating ZERO-SHOT model.")
    model.eval()

    N = len(test_images)
    paths = [img["path"] for img in test_images]
    classes = [img["class_id"] for img in test_images]

    emb_cache_path = os.path.join(os.path.dirname(args.checkpoint) if args.checkpoint != "zero-shot" else ".", "all_embeddings_cache.pt")
    if os.path.exists(emb_cache_path):
        print("Loading cached embeddings...")
        all_embeddings = torch.load(emb_cache_path)
    else:
        print("Encoding all test images...")
        all_embeddings = torch.zeros((N, 256, 768), dtype=torch.bfloat16)
        
        
        
        for i in tqdm(range(0, N, args.batch_size)):
            batch_paths = paths[i:i + args.batch_size]
            images = [Image.open(os.path.join(args.images_root, p)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)) for p in batch_paths]
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                patches = model(pixel_values)
            all_embeddings[i:i+len(batch_paths)] = patches.cpu()
        print("Saving embeddings to cache...")
        torch.save(all_embeddings, emb_cache_path)
        
    print("Computing Recall@1 using chunked MaxSim...")
    # To find Recall@1, for each query we need the class of the closest *other* image.
    hits_1 = 0
    query_chunk_size = 8
    cand_chunk_size = 4096
    
    # We will swap loops: outer loop is candidates (transferred 15 times), inner is queries.
    best_scores = torch.full((N,), -1000.0, device=device, dtype=torch.bfloat16)
    best_idx = torch.zeros((N,), dtype=torch.long, device=device)
    
    for c_start in tqdm(range(0, N, cand_chunk_size), desc="Candidate chunks"):
        c_end = min(N, c_start + cand_chunk_size)
        c_embs = all_embeddings[c_start:c_end].to(device)  # (C, 256, 768)
        
        for q_start in range(0, N, query_chunk_size):
            q_end = min(N, q_start + query_chunk_size)
            q_embs = all_embeddings[q_start:q_end].to(device)  # (Q, 256, 768)
            Q = q_end - q_start
            
            # Compute MaxSim: q_embs x c_embs
            # sim = (Q, C, 256, 256)
            sim = torch.einsum("qpd,crd->qcpr", q_embs, c_embs)
            a_to_b = sim.max(dim=3).values.mean(dim=2)  # (Q, C)
            b_to_a = sim.max(dim=2).values.mean(dim=2)  # (Q, C)
            scores = 0.5 * (a_to_b + b_to_a)  # (Q, C)
            
            # Mask out self-matches
            for i in range(Q):
                global_q_idx = q_start + i
                if c_start <= global_q_idx < c_end:
                    local_c_idx = global_q_idx - c_start
                    scores[i, local_c_idx] = -1000.0
                    
            chunk_best_scores, chunk_best_idx = scores.max(dim=1)
            
            # Update global bests
            current_best_scores = best_scores[q_start:q_end]
            update_mask = chunk_best_scores > current_best_scores
            
            # We must use clone or index assignment carefully
            best_scores[q_start:q_end] = torch.where(update_mask, chunk_best_scores, current_best_scores)
            best_idx[q_start:q_end] = torch.where(update_mask, chunk_best_idx + c_start, best_idx[q_start:q_end])
            
    # Check hits
    best_idx = best_idx.cpu().tolist()
    for i in range(N):
        query_class = classes[i]
            pred_class = classes[best_idx[i]]
            if query_class == pred_class:
                hits_1 += 1

    recall_1 = hits_1 / N
    print(f"\nFinal Full-Set Recall@1: {recall_1:.4f} ({hits_1}/{N})")
    
    out_path = os.path.join(os.path.dirname(args.checkpoint) if args.checkpoint != "zero-shot" else ".", "full_test_recall.json")
    with open(out_path, "w") as f:
        json.dump({
            "test_size": N,
            "recall_1": recall_1,
            "hits": hits_1
        }, f, indent=2)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
