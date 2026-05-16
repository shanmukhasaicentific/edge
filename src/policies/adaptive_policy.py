"""
src/policies/adaptive_policy.py

Adaptive Compute Allocation Policy.

This module formalizes the compute allocation decision as a policy:
    π(D_t) → Tier ∈ {1, 2, 3}

Three policy types are implemented:

1. ThresholdPolicy (proposed):
    π(D_t) = Tier 1 if D_t < τ_low
    π(D_t) = Tier 2 if τ_low ≤ D_t < τ_high
    π(D_t) = Tier 3 if D_t ≥ τ_high

2. FixedIntervalPolicy (baseline):
    Invoke VLM every K frames regardless of drift.

3. EveryFramePolicy (baseline):
    Always invoke VLM at Tier 3. Maximum compute, maximum semantic retention.

4. MotionGatingPolicy (baseline):
    Invoke VLM only on detected motion events.

All policies share the same interface → easy ablation comparison.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PolicyDecision:
    tier: int                 # 1, 2, or 3
    invoke_vlm: bool
    reason: str


class BasePolicy(ABC):
    """Abstract base class for all compute allocation policies."""

    @abstractmethod
    def decide(self, drift: float, frame_id: int, motion_score: float = 0.0) -> PolicyDecision:
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ─────────────────────────────────────────────────────────────
# PROPOSED: Semantic Threshold Policy
# ─────────────────────────────────────────────────────────────

class ThresholdPolicy(BasePolicy):
    """
    Proposed adaptive policy based on semantic drift thresholds.
    
    Args:
        tau_low:  D_t below this → Tier 1 (cache reuse)
        tau_high: D_t above this → Tier 3 (full VLM)
        [between] → Tier 2 (lightweight)
    """

    def __init__(self, tau_low: float = 0.15, tau_high: float = 0.40):
        self.tau_low = tau_low
        self.tau_high = tau_high

    def decide(self, drift: float, frame_id: int, motion_score: float = 0.0) -> PolicyDecision:
        if drift < self.tau_low:
            return PolicyDecision(tier=1, invoke_vlm=False, reason="low_drift_cache")
        elif drift < self.tau_high:
            return PolicyDecision(tier=2, invoke_vlm=True, reason="mid_drift_tier2")
        else:
            return PolicyDecision(tier=3, invoke_vlm=True, reason="high_drift_tier3")

    def reset(self):
        pass  # stateless


# ─────────────────────────────────────────────────────────────
# BASELINE 1: Every-Frame Policy
# ─────────────────────────────────────────────────────────────

class EveryFramePolicy(BasePolicy):
    """Baseline: Invoke full VLM on every frame. Maximum compute cost."""

    def decide(self, drift: float, frame_id: int, motion_score: float = 0.0) -> PolicyDecision:
        return PolicyDecision(tier=3, invoke_vlm=True, reason="every_frame")

    def reset(self):
        pass


# ─────────────────────────────────────────────────────────────
# BASELINE 2: Fixed Interval Policy
# ─────────────────────────────────────────────────────────────

class FixedIntervalPolicy(BasePolicy):
    """
    Baseline: Invoke VLM every K frames.
    K is a hyperparameter to sweep (K=5, 10, 30, 60).
    """

    def __init__(self, interval: int = 30):
        self.interval = interval
        self._frame_count = 0

    def decide(self, drift: float, frame_id: int, motion_score: float = 0.0) -> PolicyDecision:
        self._frame_count += 1
        if self._frame_count % self.interval == 0:
            return PolicyDecision(tier=3, invoke_vlm=True, reason="fixed_interval")
        return PolicyDecision(tier=1, invoke_vlm=False, reason="interval_skip")

    def reset(self):
        self._frame_count = 0


# ─────────────────────────────────────────────────────────────
# BASELINE 3: Motion Gating Policy
# ─────────────────────────────────────────────────────────────

class MotionGatingPolicy(BasePolicy):
    """
    Baseline: Invoke VLM only when frame-level motion exceeds threshold.
    Motion score from FrameFilter (mean absolute difference).
    """

    def __init__(self, motion_threshold: float = 10.0):
        self.motion_threshold = motion_threshold

    def decide(self, drift: float, frame_id: int, motion_score: float = 0.0) -> PolicyDecision:
        if motion_score >= self.motion_threshold:
            return PolicyDecision(tier=3, invoke_vlm=True, reason="motion_triggered")
        return PolicyDecision(tier=1, invoke_vlm=False, reason="low_motion_skip")

    def reset(self):
        pass


# ─────────────────────────────────────────────────────────────
# BASELINE 4: Embedding Threshold Gating Policy
# ─────────────────────────────────────────────────────────────

class EmbeddingThresholdPolicy(BasePolicy):
    """
    Baseline: Invoke VLM only on CLIP embedding drift.
    Uses only D_embed, ignoring D_objects and D_track.
    This isolates the contribution of the full drift formulation.
    """

    def __init__(self, tau: float = 0.30):
        self.tau = tau

    def decide(self, drift: float, frame_id: int, motion_score: float = 0.0) -> PolicyDecision:
        # drift here is D_embed only (caller must pass embed-only drift)
        if drift >= self.tau:
            return PolicyDecision(tier=3, invoke_vlm=True, reason="embed_threshold")
        return PolicyDecision(tier=1, invoke_vlm=False, reason="embed_stable")

    def reset(self):
        pass


# ─────────────────────────────────────────────────────────────
# Policy Registry — for experiment configs
# ─────────────────────────────────────────────────────────────

POLICY_REGISTRY = {
    "threshold":          ThresholdPolicy,
    "every_frame":        EveryFramePolicy,
    "fixed_interval":     FixedIntervalPolicy,
    "motion_gating":      MotionGatingPolicy,
    "embedding_threshold": EmbeddingThresholdPolicy,
}


def build_policy(name: str, **kwargs) -> BasePolicy:
    """
    Instantiate a policy by name with keyword arguments.
    Used by experiment configs to avoid hardcoding.

    Example:
        policy = build_policy("threshold", tau_low=0.2, tau_high=0.5)
        policy = build_policy("fixed_interval", interval=30)
    """
    if name not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy: {name}. Available: {list(POLICY_REGISTRY.keys())}")
    return POLICY_REGISTRY[name](**kwargs)
