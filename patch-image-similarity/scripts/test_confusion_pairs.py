"""Does raw visual similarity overstate similarity for images that share
pose/composition but differ semantically? Test on a real COCO pair: a young
child casually riding a bike on pavement vs. a young man doing a wheelie
stunt on coastal rocks -- same "person + bike + helmet + diagonal riding
pose" structure, very different actual content/context.

Compares CLIP and zero-shot DINOv2 (both general-purpose, domain-agnostic)
against a genuinely different pair from the same set for contrast, then
checks whether NIM captions of the two images make the semantic difference
explicit where the raw vision score doesn't.

Note: our trained projection head was trained ONLY on SOP product photos --
applying it here is a domain-mismatch sanity check, not a fair test of
"does training help here" (it was never trained for this domain).
"""
import os
import sys

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from encoders import PatchEncoder
from maxsim import symmetric_maxsim, global_cosine
from nim_caption_regions import NimRegionCaptioner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "data", "coco_confusion_images")
CAPTION_CACHE = os.path.join(ROOT, "data", "coco_confusion_captions_cache.json")

CHILD_BIKE = os.path.join(IMG_DIR, "210299.jpg")
MAN_STUNT_BIKE = os.path.join(IMG_DIR, "472623.jpg")
WOMAN_BENCH = os.path.join(IMG_DIR, "517069.jpg")
GIRL_BENCH = os.path.join(IMG_DIR, "391375.jpg")


def main():
    print("Loading models...")
    encoder = PatchEncoder()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    captioner = NimRegionCaptioner(cache_path=CAPTION_CACHE)

    def clip_embed(path):
        image = Image.open(path).convert("RGB")
        inputs = clip_proc(images=image, return_tensors="pt")
        with torch.no_grad():
            feat = clip_model.get_image_features(**inputs)
        return torch.nn.functional.normalize(feat[0], dim=0)

    pairs = [
        ("child-bike vs man-stunt-bike (structurally similar, semantically different)",
         CHILD_BIKE, MAN_STUNT_BIKE),
        ("woman-bench vs girl-bench (structurally similar, semantically different)",
         WOMAN_BENCH, GIRL_BENCH),
    ]

    for label, path_a, path_b in pairs:
        print(f"\n=== {label} ===")
        enc_a, enc_b = encoder.encode(path_a), encoder.encode(path_b)

        clip_sim = global_cosine(clip_embed(path_a), clip_embed(path_b))
        dino_global = global_cosine(
            torch.nn.functional.normalize(enc_a.patch_embeddings.mean(dim=0), dim=0),
            torch.nn.functional.normalize(enc_b.patch_embeddings.mean(dim=0), dim=0),
        )
        dino_meanmax = symmetric_maxsim(enc_a.patch_embeddings, enc_b.patch_embeddings).similarity

        print(f"  CLIP global cosine:       {clip_sim:.3f}")
        print(f"  DINOv2 global cosine:     {dino_global:.3f}")
        print(f"  DINOv2 patch MaxSim:      {dino_meanmax:.3f}")

        regions_a = captioner.caption_regions(path_a, grid_size=3)
        regions_b = captioner.caption_regions(path_b, grid_size=3)
        print(f"  Captions A: {[r.description for r in regions_a]}")
        print(f"  Captions B: {[r.description for r in regions_b]}")


if __name__ == "__main__":
    main()
