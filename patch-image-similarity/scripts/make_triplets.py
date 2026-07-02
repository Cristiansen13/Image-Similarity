"""Generate triplet_accuracy test set per spec section 4: (anchor, positive,
negative) triples where positive should score more similar to anchor than
negative does.

- Synthetic triplets (cheap, many): anchor + an augmented/cropped/rotated
  version of itself as positive, + a different base image as negative. Tests
  robustness to spatial transforms.
- Hand-picked semantic triplets (fewer, manual judgment calls): cases where
  "similar" means same object/scene/style rather than same pixels -- reuses
  the Phase 0 test pairs (same object/different background, same scene
  shifted/cropped, near-duplicate) plus a few cross-category groupings
  (e.g. two abstract graphic patterns vs. a real photo).

Real-photo variety here is limited to scikit-image's locally bundled sample
images (no downloads) -- fewer hand-picked semantic triplets than the spec's
20-50 suggestion as a result; compensated with more synthetic ones.
"""
import json
import os

import numpy as np
import skimage.data as skd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMAGES_DIR = os.path.join(ROOT, "data", "test_images")
TRIPLET_IMAGES_DIR = os.path.join(ROOT, "data", "triplet_images")
os.makedirs(TRIPLET_IMAGES_DIR, exist_ok=True)


def to_pil(arr):
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr).convert("RGB")


def save(img, name, subdir=TRIPLET_IMAGES_DIR):
    path = os.path.join(subdir, name)
    img.save(path)
    return path


def rotate_flip_jitter(img, rng):
    out = img.rotate(rng.uniform(-25, 25), expand=False, fillcolor=(128, 128, 128))
    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.array(out).astype(np.int16)
    brightness = rng.uniform(-30, 30)
    arr = np.clip(arr + brightness, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def crop_zoom(img, rng):
    w, h = img.size
    frac = rng.uniform(0.5, 0.75)
    cw, ch = int(w * frac), int(h * frac)
    x0 = rng.integers(0, w - cw + 1)
    y0 = rng.integers(0, h - ch + 1)
    return img.crop((x0, y0, x0 + cw, y0 + ch))


def main():
    rng = np.random.default_rng(42)

    bases = {
        "astronaut": to_pil(skd.astronaut()),
        "coffee": to_pil(skd.coffee()),
        "rocket": to_pil(skd.rocket()),
        "cat": to_pil(skd.chelsea()),
        "coins": to_pil(skd.coins()),
        "camera": to_pil(skd.camera()),
        "moon": to_pil(skd.moon()),
        "clock": to_pil(skd.clock()),
        "checkerboard": to_pil(skd.checkerboard()),
        "colorwheel": to_pil(skd.colorwheel()),
        "page": to_pil(skd.page()),
        "text": to_pil(skd.text()),
        "logo": to_pil(skd.logo()),
        "brick": to_pil(skd.brick()),
        "grass": to_pil(skd.grass()),
        "gravel": to_pil(skd.gravel()),
    }
    base_paths = {name: save(img, f"{name}.png") for name, img in bases.items()}

    triplets = []

    # --- synthetic triplets: anchor + augmented positive + random-other negative ---
    names = list(bases.keys())
    augment_fns = [("crop_zoom", crop_zoom), ("rotate_flip_jitter", rotate_flip_jitter)]
    for name in names:
        anchor_img = bases[name]
        for aug_name, aug_fn in augment_fns:
            positive_img = aug_fn(anchor_img, rng)
            positive_path = save(positive_img, f"{name}_{aug_name}.jpg")
            negative_name = rng.choice([n for n in names if n != name])
            triplets.append({
                "anchor": base_paths[name],
                "positive": positive_path,
                "negative": base_paths[negative_name],
                "type": "synthetic",
                "note": f"{name} vs {aug_name}({name}) vs {negative_name}",
            })

    # --- hand-picked semantic triplets ---
    ti = TEST_IMAGES_DIR

    def ti_path(fname):
        return os.path.join(ti, fname)

    semantic = [
        (base_paths["rocket"], ti_path("rocket_on_grass.png"), base_paths["cat"],
         "same object (rocket), different background, vs unrelated"),
        (base_paths["rocket"], ti_path("rocket_on_gravel.png"), base_paths["coffee"],
         "same object (rocket), different background, vs unrelated"),
        (ti_path("motorcycle_left.png"), ti_path("motorcycle_right.png"), base_paths["astronaut"],
         "same scene, camera-shifted (real stereo pair), vs unrelated"),
        (base_paths["astronaut"], ti_path("astronaut_crop.png"), base_paths["rocket"],
         "same scene, cropped, vs unrelated"),
        (base_paths["coffee"], ti_path("coffee_crop.png"), base_paths["cat"],
         "same scene, cropped (zoomed to subject), vs unrelated"),
        (base_paths["astronaut"], ti_path("astronaut_dup.jpg"), base_paths["coffee"],
         "near-duplicate (resize/noise/jpeg) vs unrelated"),
        (base_paths["coffee"], ti_path("coffee_dup.jpg"), base_paths["astronaut"],
         "near-duplicate (resize/noise/jpeg) vs unrelated"),
        (base_paths["grass"], base_paths["gravel"], base_paths["brick"],
         "loose ground-cover textures (grass/gravel) vs a built structure (brick wall)"),
        (base_paths["checkerboard"], base_paths["colorwheel"], base_paths["cat"],
         "abstract graphic/geometric patterns vs a real photo"),
        (base_paths["cat"], base_paths["camera"], base_paths["moon"],
         "man-made object on a plain background (cat photo has a similar plain-background studio feel to camera) vs a distant celestial scene"),
        (base_paths["page"], base_paths["text"], base_paths["cat"],
         "scanned text/document imagery vs a real photo"),
        (base_paths["clock"], base_paths["camera"], base_paths["grass"],
         "small manufactured object photographed on plain background vs a ground texture"),
    ]
    for anchor, positive, negative, note in semantic:
        triplets.append({
            "anchor": anchor, "positive": positive, "negative": negative,
            "type": "semantic", "note": note,
        })

    manifest_path = os.path.join(ROOT, "data", "triplets.json")
    with open(manifest_path, "w") as f:
        json.dump(triplets, f, indent=2)

    n_synthetic = sum(1 for t in triplets if t["type"] == "synthetic")
    n_semantic = sum(1 for t in triplets if t["type"] == "semantic")
    unique_images = {t[k] for t in triplets for k in ("anchor", "positive", "negative")}
    print(f"Wrote {len(triplets)} triplets ({n_synthetic} synthetic, {n_semantic} semantic) "
          f"to {manifest_path}")
    print(f"Unique images referenced: {len(unique_images)}")


if __name__ == "__main__":
    main()
