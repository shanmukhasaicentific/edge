"""
src/semantic_memory/memory.py

Temporal Semantic Memory.

Maintains a decaying memory of past semantic states.
Memory update rule:
    M_t = λ · M_{t-1} + (1 - λ) · Z_t

Where:
    M_t  = memory state at time t
    λ    = decay factor (0.8 – 0.95)
    Z_t  = current semantic state embedding

This gives recent frames higher influence while preserving
historical context — critical for detecting slow semantic drift.

The memory acts as a reference point: if the current embedding
drifts far from memory, a VLM call is warranted.

This persistent memory serves a similar conceptual role to QueST's
persistent semantic queries — both maintain a stable semantic reference
across time to detect drift from the scene's established identity.
The key architectural difference: QueST uses learnable query embeddings
that attend globally via cross-attention; we use an exponentially decayed
running average of CLIP embeddings. Our approach is far lighter,
requiring no training, making it suitable for edge deployment.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class MemoryState:
    """Snapshot of memory at a given frame."""
    frame_id: int
    memory_embedding: np.ndarray        # M_t — decayed memory vector
    last_vlm_embedding: np.ndarray      # embedding at last VLM call
    frames_since_vlm: int               # how many frames since last VLM call
    vlm_call_count: int                 # total VLM calls so far


class TemporalSemanticMemory:
    """
    Exponentially decaying temporal memory over CLIP embeddings.

    Args:
        decay_lambda: λ in [0, 1]. Higher = slower decay = longer memory.
            Recommended: 0.9 for 30fps video (≈ 10-frame effective window)
        embedding_dim: CLIP ViT-B/32 = 512
    """

    def __init__(
        self,
        decay_lambda: float = 0.9,
        embedding_dim: int = 512,
    ):
        assert 0.0 < decay_lambda < 1.0, "decay_lambda must be in (0, 1)"
        self.decay_lambda = decay_lambda
        self.embedding_dim = embedding_dim

        # State
        self._memory: Optional[np.ndarray] = None           # M_t
        self._last_vlm_embedding: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._frames_since_vlm: int = 0
        self._vlm_call_count: int = 0

    def update(self, embedding: np.ndarray) -> np.ndarray:
        """
        Update memory with new embedding.
        M_t = λ · M_{t-1} + (1 - λ) · E_t

        Args:
            embedding: L2-normalized CLIP embedding (512,)

        Returns:
            Updated memory state M_t (512,)
        """
        if self._memory is None:
            # Cold start: initialize memory to first embedding
            self._memory = embedding.copy()
        else:
            self._memory = (
                self.decay_lambda * self._memory
                + (1 - self.decay_lambda) * embedding
            )
            # Re-normalize memory to keep it as a unit vector
            norm = np.linalg.norm(self._memory)
            if norm > 1e-8:
                self._memory = self._memory / norm

        self._frame_id += 1
        self._frames_since_vlm += 1
        return self._memory.copy()

    def register_vlm_call(self, embedding: np.ndarray):
        """
        Call this when a VLM invocation occurs.
        Resets the frames_since_vlm counter and saves the reference embedding.
        """
        self._last_vlm_embedding = embedding.copy()
        self._frames_since_vlm = 0
        self._vlm_call_count += 1

    def drift_from_last_vlm(self, current_embedding: np.ndarray) -> float:
        """
        Cosine distance from the embedding at the last VLM call.
        Used as an alternative drift signal.

        Returns 1.0 if no VLM call has occurred yet (forcing first call).
        """
        if self._last_vlm_embedding is None:
            return 1.0
        return float(1.0 - np.dot(current_embedding, self._last_vlm_embedding))

    def drift_from_memory(self, current_embedding: np.ndarray) -> float:
        """
        Cosine distance from the current memory state M_t.
        """
        if self._memory is None:
            return 1.0
        return float(1.0 - np.dot(current_embedding, self._memory))

    @property
    def state(self) -> MemoryState:
        return MemoryState(
            frame_id=self._frame_id,
            memory_embedding=self._memory.copy() if self._memory is not None else np.zeros(self.embedding_dim),
            last_vlm_embedding=self._last_vlm_embedding.copy() if self._last_vlm_embedding is not None else np.zeros(self.embedding_dim),
            frames_since_vlm=self._frames_since_vlm,
            vlm_call_count=self._vlm_call_count,
        )

    def reset(self):
        self._memory = None
        self._last_vlm_embedding = None
        self._frame_id = 0
        self._frames_since_vlm = 0
        self._vlm_call_count = 0
