"""
src/scheduler/vlm_scheduler.py

VLM Invocation Scheduler.

Maps semantic drift D_t to a compute tier decision, then either:
    - Tier 1: skip VLM entirely (cache hit or low drift)
    - Tier 2: invoke lightweight captioning (mid drift)
    - Tier 3: invoke full VLM reasoning (high drift)

Scheduling rule:
    D_t < τ_low   →  Tier 1 (cache reuse)
    τ_low ≤ D_t < τ_high  →  Tier 2 (lightweight)
    D_t ≥ τ_high  →  Tier 3 (full VLM)

Also supports:
    - forced VLM calls every N frames (safety net)
    - minimum interval between VLM calls (rate limiting)
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

from src.scheduler.semantic_cache import CacheResult, SemanticCache


class ComputeTier(IntEnum):
    TIER_1 = 1   # CPU only — cache reuse, no VLM
    TIER_2 = 2   # GPU light — lightweight captioning
    TIER_3 = 3   # GPU heavy — full VLM reasoning


@dataclass
class SchedulerDecision:
    tier: ComputeTier
    invoke_vlm: bool
    cache_result: Optional[CacheResult]
    reason: str              # human-readable explanation
    drift: float
    frame_id: int


class VLMScheduler:
    """
    Adaptive VLM invocation scheduler.

    Args:
        tau_low:       drift threshold for cache reuse (Tier 1)
        tau_high:      drift threshold for full VLM (Tier 3)
        force_every_n: force a VLM call every N frames regardless of drift
        min_interval:  minimum frames between consecutive VLM calls
        cache:         SemanticCache instance (injected)
    """

    def __init__(
        self,
        tau_low: float = 0.15,
        tau_high: float = 0.40,
        force_every_n: int = 300,        # ~10 sec at 30fps
        min_interval: int = 5,           # never call VLM more than once per 5 frames
        cache: Optional[SemanticCache] = None,
    ):
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.force_every_n = force_every_n
        self.min_interval = min_interval
        self.cache = cache or SemanticCache(tau_low=tau_low, tau_high=tau_high)

        self._last_vlm_frame: int = -999
        self._frame_count: int = 0
        self._tier_counts = {1: 0, 2: 0, 3: 0}

    def decide(self, drift: float, frame_id: int) -> SchedulerDecision:
        """
        Make a scheduling decision for the current frame.

        Args:
            drift:    D_t semantic drift score [0, 1]
            frame_id: current frame number

        Returns:
            SchedulerDecision with tier and VLM invocation flag
        """
        self._frame_count += 1
        frames_since_vlm = frame_id - self._last_vlm_frame

        # --- Forced VLM call (safety net for very long cache hits) ---
        if frames_since_vlm >= self.force_every_n:
            decision = SchedulerDecision(
                tier=ComputeTier.TIER_3,
                invoke_vlm=True,
                cache_result=None,
                reason="forced_periodic",
                drift=drift,
                frame_id=frame_id,
            )
            self._record(decision)
            return decision

        # --- Rate limiting: don't call VLM too frequently ---
        if frames_since_vlm < self.min_interval:
            cache_result = self.cache.query(drift=0.0, frame_id=frame_id)  # force cache check
            decision = SchedulerDecision(
                tier=ComputeTier.TIER_1,
                invoke_vlm=False,
                cache_result=cache_result,
                reason="rate_limited",
                drift=drift,
                frame_id=frame_id,
            )
            self._record(decision)
            return decision

        # --- Cache query ---
        cache_result = self.cache.query(drift=drift, frame_id=frame_id)

        # Tier 1: cache hit
        if cache_result.hit:
            decision = SchedulerDecision(
                tier=ComputeTier.TIER_1,
                invoke_vlm=False,
                cache_result=cache_result,
                reason="cache_hit",
                drift=drift,
                frame_id=frame_id,
            )
            self._record(decision)
            return decision

        # Tier 2: mid-range drift
        if drift < self.tau_high:
            decision = SchedulerDecision(
                tier=ComputeTier.TIER_2,
                invoke_vlm=True,
                cache_result=cache_result,
                reason="mid_drift_tier2",
                drift=drift,
                frame_id=frame_id,
            )
            self._record(decision)
            return decision

        # Tier 3: high drift — full VLM
        decision = SchedulerDecision(
            tier=ComputeTier.TIER_3,
            invoke_vlm=True,
            cache_result=cache_result,
            reason="high_drift_tier3",
            drift=drift,
            frame_id=frame_id,
        )
        self._record(decision)
        return decision

    def register_vlm_call(self, frame_id: int, vlm_output: str, embedding: np.ndarray):
        """
        Must be called after every successful VLM invocation
        to update the cache and last-call tracker.
        """
        self._last_vlm_frame = frame_id
        self.cache.update(vlm_output=vlm_output, embedding=embedding, frame_id=frame_id)

    def _record(self, decision: SchedulerDecision):
        self._tier_counts[int(decision.tier)] += 1

    @property
    def tier_distribution(self) -> dict:
        total = sum(self._tier_counts.values())
        return {
            f"tier_{k}": v for k, v in self._tier_counts.items()
        } | {"total_frames": total}

    @property
    def vlm_call_rate(self) -> float:
        """Fraction of frames that triggered a VLM call (Tier 2 or 3)."""
        total = sum(self._tier_counts.values())
        vlm_frames = self._tier_counts[2] + self._tier_counts[3]
        return vlm_frames / total if total > 0 else 0.0

    def reset(self):
        self._last_vlm_frame = -999
        self._frame_count = 0
        self._tier_counts = {1: 0, 2: 0, 3: 0}
        self.cache.reset()
