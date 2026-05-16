"""
src/embeddings/clip_extractor.py

CLIP ViT-B/32 semantic embedding extractor.

Embeddings are L2-normalized so cosine similarity = dot product.
The embedding drift D_embed is the primary semantic signal.

D_embed = 1 - cosine_similarity(E_t, E_{t-1})
         = 1 - (E_t · E_{t-1})  [since both are unit vectors]
"""

import time
from typing import Optional

import numpy as np
import torch


class CLIPExtractor:
    """
    Extracts CLIP ViT-B/32 semantic embeddings from frames.

    Embeddings are always L2-normalized (unit vectors).
    Cosine distance is then simply: 1 - dot(E_t, E_{t-1}).

    Args:
        model_name: CLIP model variant (default ViT-B/32)
        device: 'cuda' or 'cpu'
        batch_size: inference batch size (keep 1 for edge)
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
    ):
        import clip

        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        self.embedding_dim = 512  # ViT-B/32 output dim

    @torch.no_grad()
    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract L2-normalized CLIP embedding from a BGR frame.

        Args:
            frame: BGR uint8 numpy array (H, W, 3)

        Returns:
            np.ndarray of shape (512,) — unit vector
        """
        from PIL import Image

        # Convert BGR → RGB → PIL
        rgb = frame[:, :, ::-1]
        pil_image = Image.fromarray(rgb)

        # CLIP preprocessing (resize + normalize)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        # Extract embedding
        embedding = self.model.encode_image(tensor)

        # L2 normalize
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().squeeze(0)  # (512,)

    def embedding_drift(self, e1: np.ndarray, e2: np.ndarray) -> float:
        """
        Compute cosine distance between two unit-norm embeddings.
        D_embed = 1 - dot(e1, e2)
        Range: [0, 2] but practically [0, 1] for semantically related frames.

        Args:
            e1, e2: unit-norm embeddings of shape (512,)

        Returns:
            float in [0, 2]
        """
        return float(1.0 - np.dot(e1, e2))

    def timed_extract(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Extract embedding and return (embedding, inference_ms).
        """
        t0 = time.perf_counter()
        embedding = self.extract(frame)
        ms = (time.perf_counter() - t0) * 1000
        return embedding, ms
