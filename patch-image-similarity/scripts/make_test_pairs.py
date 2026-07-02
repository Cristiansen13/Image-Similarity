"""Generate ~10 test image pairs for Phase 0 sanity checking.

Uses scikit-image's bundled sample photos (no downloads) plus PIL transforms
to build near-duplicate, cropped, shifted, composited, and unrelated pairs.
Writes images to data/test_images/ and a manifest to data/test_images/pairs.json.
"""
import io
import json
import os

import numpy as np
from PIL import Image, ImageFilter
import skimage.data as skd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "test_images")
os.makedirs(OUT_DIR, exist_ok=True)


def to_pil(arr):
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr).convert("RGB")


def save(img, name):
    path = os.path.join(OUT_DIR, name)
    img.save(path)
    return path


def jpeg_recompress(img, quality=15):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def add_noise(img, sigma=8):
    arr = np.array(img).astype(np.int16)
    noise = np.random.default_rng(0).normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_near_duplicate(img):
    small = img.resize((img.width // 3, img.height // 3), Image.BILINEAR)
    small = small.resize(img.size, Image.BILINEAR)
    small = add_noise(small, sigma=6)
    return jpeg_recompress(small, quality=20)


def composite_on_background(fg_crop, bg_arr, canvas=(400, 400)):
    bg = to_pil(bg_arr).resize(canvas)
    fg = fg_crop.copy()
    fg.thumbnail((canvas[0] - 40, canvas[1] - 40))
    x = (canvas[0] - fg.width) // 2
    y = (canvas[1] - fg.height) // 2
    bg.paste(fg, (x, y))
    return bg


def main():
    pairs = []

    astronaut = to_pil(skd.astronaut())
    coffee = to_pil(skd.coffee())
    rocket = to_pil(skd.rocket())
    cat = to_pil(skd.chelsea())
    coins = to_pil(skd.coins())
    brick = skd.brick()
    grass = skd.grass()
    gravel = skd.gravel()
    left, right, _ = skd.stereo_motorcycle()
    left, right = to_pil(left), to_pil(right)

    # 1. near-duplicate: astronaut vs downsampled/noisy/jpeg version
    a_path = save(astronaut, "astronaut.png")
    a_dup_path = save(make_near_duplicate(astronaut), "astronaut_dup.jpg")
    pairs.append({"name": "near_duplicate_astronaut", "a": a_path, "b": a_dup_path,
                  "expected": "high", "category": "near_duplicate"})

    # 2. near-duplicate: coffee vs downsampled/noisy/jpeg version
    c_path = save(coffee, "coffee.png")
    c_dup_path = save(make_near_duplicate(coffee), "coffee_dup.jpg")
    pairs.append({"name": "near_duplicate_coffee", "a": c_path, "b": c_dup_path,
                  "expected": "high", "category": "near_duplicate"})

    # 3. same scene, camera-shifted: stereo pair (real parallax shift)
    left_path = save(left, "motorcycle_left.png")
    right_path = save(right, "motorcycle_right.png")
    pairs.append({"name": "same_scene_shifted_motorcycle", "a": left_path, "b": right_path,
                  "expected": "high", "category": "same_scene_shifted"})

    # 4. same scene, cropped: astronaut vs center 60% crop
    w, h = astronaut.size
    cx0, cy0 = int(w * 0.2), int(h * 0.2)
    cx1, cy1 = int(w * 0.8), int(h * 0.8)
    astronaut_crop = astronaut.crop((cx0, cy0, cx1, cy1))
    ac_path = save(astronaut_crop, "astronaut_crop.png")
    pairs.append({"name": "same_scene_cropped_astronaut", "a": a_path, "b": ac_path,
                  "expected": "high", "category": "same_scene_cropped"})

    # 5. same scene, cropped: coffee vs zoomed-in cup-only crop
    w, h = coffee.size
    coffee_crop = coffee.crop((int(w * 0.15), int(h * 0.05), int(w * 0.75), int(h * 0.75)))
    cc_path = save(coffee_crop, "coffee_crop.png")
    pairs.append({"name": "same_scene_cropped_coffee", "a": c_path, "b": cc_path,
                  "expected": "medium-high", "category": "same_scene_cropped"})

    # 6/7. same object, different background: rocket body composited onto textures
    rocket_body = rocket.crop((270, 40, 370, 400))
    r_path = save(rocket, "rocket.png")
    r_grass_path = save(composite_on_background(rocket_body, grass), "rocket_on_grass.png")
    r_gravel_path = save(composite_on_background(rocket_body, gravel), "rocket_on_gravel.png")
    pairs.append({"name": "same_object_diff_background_rocket_grass", "a": r_path, "b": r_grass_path,
                  "expected": "medium", "category": "same_object_diff_background"})
    pairs.append({"name": "same_object_diff_background_rocket_gravel", "a": r_path, "b": r_gravel_path,
                  "expected": "medium", "category": "same_object_diff_background"})

    # 8-11. unrelated pairs
    cat_path = save(cat, "cat.png")
    coins_path = save(coins, "coins.png")
    pairs.append({"name": "unrelated_astronaut_coffee", "a": a_path, "b": c_path,
                  "expected": "low", "category": "unrelated"})
    pairs.append({"name": "unrelated_cat_coins", "a": cat_path, "b": coins_path,
                  "expected": "low", "category": "unrelated"})
    pairs.append({"name": "unrelated_rocket_cat", "a": r_path, "b": cat_path,
                  "expected": "low", "category": "unrelated"})
    brick_path = save(to_pil(brick), "brick.png")
    grass_path = save(to_pil(grass), "grass.png")
    pairs.append({"name": "unrelated_texture_brick_grass", "a": brick_path, "b": grass_path,
                  "expected": "low", "category": "unrelated"})

    manifest_path = os.path.join(OUT_DIR, "pairs.json")
    with open(manifest_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"Wrote {len(pairs)} pairs to {manifest_path}")
    for p in pairs:
        print(f"  {p['name']:45s} [{p['category']:25s}] expected={p['expected']}")


if __name__ == "__main__":
    main()
