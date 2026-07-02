"""DINOv2 patch-token encoder.

Wraps a frozen DINOv2 ViT and exposes both the per-patch token grid (for
late-interaction / MaxSim scoring) and a global embedding (CLS token) used
as the sanity-check baseline sitting next to it.
"""
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DEFAULT_MODEL = "facebook/dinov2-base"
DEFAULT_IMAGE_SIZE = 224  # divisible by patch_size=14 -> clean 16x16 patch grid


@dataclass
class EncodedImage:
    patch_embeddings: torch.Tensor  # (num_patches, hidden_dim), L2-normalized
    global_embedding: torch.Tensor  # (hidden_dim,), L2-normalized (CLS token)
    grid_size: int  # patches per side (grid_size x grid_size == num_patches)


class PatchEncoder:
    """Loads a frozen ViT once, encodes image paths into patch + global embeddings."""

    def __init__(self, model_name: str = DEFAULT_MODEL, image_size: int = DEFAULT_IMAGE_SIZE,
                 device: str = "cpu"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.processor.size = {"height": image_size, "width": image_size}
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device).eval()
        self.patch_size = self.model.config.patch_size
        self.image_size = image_size
        if image_size % self.patch_size != 0:
            raise ValueError(f"image_size {image_size} not divisible by patch_size {self.patch_size}")
        self.grid_size = image_size // self.patch_size

    @torch.no_grad()
    def encode(self, image_path: str) -> EncodedImage:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state[0]  # (1 + num_patches, hidden_dim)

        num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        cls_token = last_hidden[0]
        patch_tokens = last_hidden[1 + num_register_tokens:]

        expected_patches = self.grid_size * self.grid_size
        if patch_tokens.shape[0] != expected_patches:
            raise RuntimeError(
                f"expected {expected_patches} patch tokens, got {patch_tokens.shape[0]}"
            )

        patch_embeddings = torch.nn.functional.normalize(patch_tokens, dim=-1)
        global_embedding = torch.nn.functional.normalize(cls_token, dim=-1)

        return EncodedImage(
            patch_embeddings=patch_embeddings,
            global_embedding=global_embedding,
            grid_size=self.grid_size,
        )
