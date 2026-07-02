"""Sentence-embedding wrapper for region captions."""
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> torch.Tensor:
        """Returns (len(texts), dim) L2-normalized embeddings."""
        embeddings = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        return embeddings
