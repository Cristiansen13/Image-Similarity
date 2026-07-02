"""Region captioning for Phase 1.

The spec's recipe is a single instruction-following LLM call per image: overlay
a numbered grid and ask for one structured JSON caption per cell. That needs a
vision-capable instruction-tuned LLM (e.g. Claude with an API key). No API key
is configured in this environment, so this module substitutes a local model
(BLIP image captioning) instead:

- one captioning call per grid cell (not per raw ViT patch, so still far
  fewer calls than the 256-patch blowup the spec warns about: 16-36 vs 256)
- each cell is captioned from a *padded* crop (crop expanded by `padding_frac`
  before clipping to image bounds) rather than the bare cell, so the model
  still sees surrounding context instead of an isolated, ambiguous fragment
- BLIP is not instruction-following, so it can't return structured multi-cell
  JSON from one whole-image call the way the spec's target LLM would

If an ANTHROPIC_API_KEY (or similar) becomes available, replace this module's
`_caption_local` with a single grid-overlay + structured-JSON call and keep the
same `caption_regions()` interface -- callers don't need to change.
"""
import hashlib
import json
import os
from dataclasses import dataclass, asdict

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"


@dataclass
class Region:
    row: int
    col: int
    description: str


class RegionCaptioner:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu",
                 cache_path: str | None = None):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device).eval()
        self.device = device
        self.cache_path = cache_path
        self._cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self._cache = json.load(f)

    def _save_cache(self):
        if self.cache_path:
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)

    def _cache_key(self, image_path: str, grid_size: int, padding_frac: float) -> str:
        stat = os.stat(image_path)
        raw = f"{image_path}|{stat.st_mtime}|{grid_size}|{padding_frac}"
        return hashlib.sha1(raw.encode()).hexdigest()

    @torch.no_grad()
    def _caption_crop(self, crop: Image.Image) -> str:
        inputs = self.processor(images=crop, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=25)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def caption_regions(self, image_path: str, grid_size: int = 4,
                         padding_frac: float = 0.4) -> list[Region]:
        key = self._cache_key(image_path, grid_size, padding_frac)
        if key in self._cache:
            return [Region(**r) for r in self._cache[key]]

        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        cell_w, cell_h = w / grid_size, h / grid_size

        regions = []
        for row in range(grid_size):
            for col in range(grid_size):
                x0, y0 = col * cell_w, row * cell_h
                x1, y1 = x0 + cell_w, y0 + cell_h
                pad_x, pad_y = cell_w * padding_frac, cell_h * padding_frac
                crop_box = (
                    max(0, x0 - pad_x), max(0, y0 - pad_y),
                    min(w, x1 + pad_x), min(h, y1 + pad_y),
                )
                crop = image.crop(crop_box)
                description = self._caption_crop(crop)
                regions.append(Region(row=row, col=col, description=description))

        self._cache[key] = [asdict(r) for r in regions]
        self._save_cache()
        return regions
