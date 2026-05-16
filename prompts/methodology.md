# Methodology Prompt

## Purpose

Use this prompt when: writing methodology sections, justifying design choices, formalizing math, or reasoning about system architecture decisions.

## Context

Paste the relevant section of code or draft, then ask:

> "Review and strengthen this methodology. Justify every design choice mathematically. Flag any weak points. Suggest improvements that remain feasible with Kaggle T4 GPUs and a 3-month timeline."

## Methodology Pillars

### 1. Semantic Drift Modeling

Justify why cosine distance in CLIP embedding space captures semantic change better than pixel-level metrics.

Key argument: CLIP embeddings encode semantic content, not visual texture. Two frames with identical objects but different lighting have low cosine drift but high pixel drift.

### 2. Temporal Semantic Memory

Justify memory design: exponential decay ensures old states fade, preventing stale cache hits from suppressing necessary VLM calls.

```python
M_t = λ · M_{t-1} + (1-λ) · Z_t
```

Justify λ range: 0.8–0.95 balances recency vs. stability.

### 3. Adaptive Compute Allocation

Justify three-tier design:
- Tier 1 always runs (no VLM cost)
- Tier 2 as intermediate option (reduces false negatives)
- Tier 3 only for high-drift frames (controls peak cost)

### 4. Threshold Policy

Justify τ selection via Pareto analysis. Do not hardcode τ — sweep [0.1, 0.9] and report Pareto frontier.

### 5. Semantic Cache Reuse

Justify cache reuse: if D_t < τ_low, reuse last VLM output. Reduces redundant inference without semantic fidelity loss.

## Checklist Before Submitting Any Methodology Section

- [ ] Every module has a mathematical formulation
- [ ] Every design choice is justified (not just described)
- [ ] All hyperparameters are swept, not hardcoded
- [ ] Semantic retention metric is formally defined
- [ ] Compute cost metric is formally defined
- [ ] SCE (Semantic Compute Efficiency) is computed and reported
- [ ] Baselines are fairly implemented
- [ ] Ablation study covers each module independently
