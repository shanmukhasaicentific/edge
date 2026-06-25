# Semantic Monitoring for Adaptive Compute Allocation in Edge Robotic Vision-Language Systems

Autonomous research program for M.Tech thesis on semantic-aware VLM scheduling.

## Setup

Work with the user to establish the research baseline:

1. **Run tag**: Use the date (e.g. `jun25`). Branch: `autoresearch/<tag>`
2. **Read in-scope files**:
   - `README.md` — thesis context (../README.md in parent directory)
   - `prepare.py` — fixed pipeline, video loading, evaluation metrics. Do not modify.
   - `train.py` — the file you modify. Scheduler parameters and drift formulation.
3. **Verify test video exists**: Point to the video file (provided in ../experiments/test/)
4. **Initialize results.tsv**: Create with header row. Baseline recorded after first run.
5. **Confirm and start**.

## Experimentation

**Three-phase research program:**

### Phase 1: Threshold Sweep (Find Pareto Frontier)

Test all combinations of τ_low and τ_high thresholds:
- τ_low ∈ [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
- τ_high ∈ [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
- For each: α=0.5, β=0.3, γ=0.2 (drift weights FIXED)

**Goal**: Identify Pareto-optimal (τ_low, τ_high) that maximizes SCE (Semantic Compute Efficiency).

**Key metric**: SCE = semantic_retention / vlm_call_rate

### Phase 2: Ablation Studies (Understand Component Importance)

Test which drift components matter most:
- **Baseline**: α=0.5, β=0.3, γ=0.2
- **No embedding**: α=0.0, β=0.5, γ=0.5
- **No objects**: α=1.0, β=0.0, γ=0.0
- **No tracking**: α=0.6, β=0.4, γ=0.0
- **Uniform**: α=0.33, β=0.33, γ=0.33

Use best (τ_low, τ_high) from Phase 1 for all runs.

**Goal**: Rank which component (embedding, objects, tracking) is most important.

### Phase 3: Failure Analysis (Test Edge Cases)

Test robustness on challenging scenarios:
- Stable scene (negative control)
- Rapid semantic transitions (positive control)
- Camera motion (confounding variable)
- Lighting changes (confounding variable)
- Object persistence (tracking robustness)

Use best config from Phase 1.

**Goal**: Identify failure modes and false positive/negative rates.

## Output Format

Each run prints a summary:

```
---
sce:                  0.8200
semantic_retention:   0.9400
vlm_call_rate:        0.0420
latency_ms:           45.3
gpu_utilization:      0.42
elapsed_seconds:      127.5
false_positives:      2
false_negatives:      0
```

Extract from log:
```
grep "^sce:\|^semantic_retention:\|^vlm_call_rate:" run.log
```

## Logging Results

Log to `results.tsv` (tab-separated):

```
commit	sce	semantic_retention	vlm_call_rate	false_pos	false_neg	status	description
a1b2c3d	0.8200	0.9400	0.0420	2	0	keep	tau_low=0.15, tau_high=0.40
b2c3d4e	0.7900	0.9200	0.0680	3	1	discard	tau_low=0.10, tau_high=0.35 (worse SCE)
c3d4e5f	0.8150	0.9300	0.0450	1	0	keep	ablation: no_tracking (alpha=0.6)
```

Columns:
1. commit (short 7-char hash)
2. sce (primary metric — higher is better)
3. semantic_retention (% of frames with valid state)
4. vlm_call_rate (calls per second)
5. false_positives (unnecessary VLM calls)
6. false_negatives (missed semantic changes)
7. status: `keep`, `discard`, or `crash`
8. description of what was tested

## The Experiment Loop

LOOP FOREVER:

1. **Read git state** — which branch/commit
2. **Modify `train.py`** — change one parameter or test one phase
3. **git commit** — with clear message
4. **Run**: `uv run train.py > run.log 2>&1`
5. **Extract results**: `grep "^sce:\|^semantic_retention:\|^vlm_call_rate:" run.log`
6. **If crashed**: Check `tail -n 50 run.log` for error. Fix if trivial, skip if fundamental.
7. **Record in results.tsv** — do NOT commit this file
8. **If SCE improved** (higher): Keep the commit, advance branch
9. **If SCE worse or equal**: git reset back

**What you CAN modify:**
- `train.py` — scheduler logic, drift formulation, threshold values, component weights
- Experiment design — which phase to run next

**What you CANNOT modify:**
- `prepare.py` — fixed pipeline, video loading, metric calculation
- Input videos or test data
- External dependencies (only what's in requirements)

**What's the goal?**
- **Phase 1**: Find (τ_low, τ_high) with highest SCE
- **Phase 2**: Understand which drift components matter
- **Phase 3**: Identify failure modes and edge cases
- **Success**: All three phases complete with actionable insights

**Simplicity criterion**: A 0.01 improvement in SCE that adds 10 lines of hacky code? Probably not worth it. A 0.01 improvement that simplifies code? Definitely keep.

**Timeout**: Each experiment ~10-15 minutes (video processing varies). If exceeds 30 minutes, kill it.

**NEVER STOP**: Run experiments autonomously until manually interrupted. If ideas run out, try:
- Re-read the program.md for missed angles
- Combine previous near-misses
- Try more extreme threshold values
- Analyze failure patterns from earlier runs
- Question assumptions in the drift formulation

The user expects you to run 50-100 experiments while they sleep, discovering the optimal configuration automatically.
