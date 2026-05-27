"""
src/scheduler/semantic_cache.py

Semantic Cache for VLM Output Reuse.

When semantic drift D_t < τ_low, the scene is semantically stable
and we can reuse the last VLM output instead of re-invoking inference.

Cache entry structure:
    - vlm_output: the caption/description string
    - embedding:  CLIP embedding at time of VLM call
    - timestamp:  frame_id when cached
    - hit_count:  number of times this entry was reused

Reuse condition:
    D_t < τ_low  →  cache hit, skip VLM
    τ_low ≤ D_t < τ_high  →  partial reasoning (Tier 2)
    D_t ≥ τ_high  →  full VLM call (Tier 3), update cache
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CacheEntry:
    vlm_output: str
    embedding: np.ndarray       # CLIP embedding at time of this VLM call
    frame_id: int
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0


@dataclass
class CacheResult:
    hit: bool
    output: Optional[str]       # VLM output string if hit, else None
    entry: Optional[CacheEntry]
    drift_at_query: float


class SemanticCache:
    """
    Single-entry semantic cache (last VLM output).

    Design decision: We use a single-entry cache (not LRU) because
    for video streams, the most recent semantic state is the only
    relevant reference. Older entries would only cause stale hits.

    Args:
        tau_low:  D_t below this → cache hit (reuse output)
        tau_high: D_t above this → full VLM call (Tier 3)
        max_frames_cached: force VLM call if cache is this many frames old
    """

    def __init__(
        self,
        tau_low: float = 0.15,
        tau_high: float = 0.40,
        max_frames_cached: int = 150,   # ~5 sec at 30fps
    ):
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.max_frames_cached = max_frames_cached

        self._entry: Optional[CacheEntry] = None
        self._total_hits = 0
        self._total_misses = 0

    def query(self, drift: float, frame_id: int) -> CacheResult:
        """
        Check if cache can serve this frame.

        Args:
            drift: D_t semantic drift score
            frame_id: current frame number

        Returns:
            CacheResult — hit=True means reuse is valid
        """
        if self._entry is None:
            self._total_misses += 1
            return CacheResult(hit=False, output=None, entry=None, drift_at_query=drift)

        # Force miss if cache is too stale
        frames_age = frame_id - self._entry.frame_id
        if frames_age >= self.max_frames_cached:
            self._total_misses += 1
            return CacheResult(hit=False, output=None, entry=self._entry, drift_at_query=drift)

        # Drift-based decision
        if drift < self.tau_low:
            self._entry.hit_count += 1
            self._total_hits += 1
            return CacheResult(hit=True, output=self._entry.vlm_output, entry=self._entry, drift_at_query=drift)

        self._total_misses += 1
        return CacheResult(hit=False, output=None, entry=self._entry, drift_at_query=drift)

    def update(self, vlm_output: str, embedding: np.ndarray, frame_id: int):
        """
        Store new VLM output in cache after a Tier 3 invocation.

        Args:
            vlm_output: caption/description from VLM
            embedding:  CLIP embedding at this frame
            frame_id:   current frame number
        """
        self._entry = CacheEntry(
            vlm_output=vlm_output,
            embedding=embedding,
            frame_id=frame_id,
        )

    @property
    def hit_rate(self) -> float:
        total = self._total_hits + self._total_misses
        return self._total_hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        return {
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": self.hit_rate,
            "current_entry_age": None if self._entry is None else self._entry.hit_count,
        }

    def reset(self):
        self._entry = None
        self._total_hits = 0
        self._total_misses = 0
