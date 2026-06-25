# Semantic Monitoring for Adaptive Compute Allocation in Edge Robotic Vision-Language Systems

**Autonomous Research Program for M.Tech Thesis**

This document defines the research protocol for discovering optimal semantic drift thresholds and component weights for adaptive VLM scheduling in edge robotics.

---

## Phase 0: Observe (Baseline Metrics)

**Goal:** Understand the semantic drift landscape on test videos.

**Scientific Question:** How does semantic drift behave over time on our test videos?

**Method:**
1. Run pipeline on test video with default config:
   - τ_low = 0.15, τ_high = 0.40 (from experiments/test/config.json)
   - α = 0.5, β = 0.3, γ = 0.2
   - skip_vlm = True (fast mode)

2. Record metrics:
   - `semantic_drift_mean`: Average D_t across video
   - `semantic_drift_std`: Standard deviation
   - `drift_max`: Peak D_t value
   - `vlm_call_rate`: Calls per second
   - `semantic_retention`: % of frames where state is usable
   - `sce`: Semantic Compute Efficiency (primary metric)
   - `latency_mean_ms`: Average per-frame latency
   - `gpu_utilization_%`: GPU load

3. Visualize:
   - D_t over time (semantic drift curve)
   - Tier decisions (when VLM is triggered)
   - Object tracking stability

**Success Criterion:** Baseline metrics recorded, video runs without errors.

**Log Entry Format:**
```
### Phase 0 — Baseline Observation
**Hypothesis:** Default config should show τ_high triggers at semantic transitions
**Parameters:** τ_low=0.15, τ_high=0.40, α=0.5, β=0.3, γ=0.2
**Video:** [name], [duration] seconds
**Observations:** [what the drift curve shows]
**Metrics:**
- vlm_call_rate: X calls/sec
- semantic_retention: Y%
- sce: Z
**Conclusion:** Baseline established. Ready for Phase 1.
```

---

## Phase 1: Threshold Sweep (Grid Search)

**Goal:** Find the Pareto frontier of semantic retention vs. compute efficiency.

**Scientific Question:** What are the optimal τ_low and τ_high values for maximizing SCE while maintaining semantic retention?

**Method:**

Test all combinations:
```
τ_low ∈ [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
τ_high ∈ [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
WHERE τ_low < τ_high
```

This creates ~30 valid configurations. Run each **once** on the test video with:
- α = 0.5, β = 0.3, γ = 0.2 (FIXED)
- skip_vlm = True (fast mode)
- Same video for all runs

**For each config:**
1. Run pipeline and record:
   - `vlm_call_rate` (calls per second)
   - `semantic_retention` (% of usable frames)
   - `sce` = semantic_retention / vlm_call_rate (efficiency metric)
   - `latency_mean_ms`
   - `gpu_utilization_%`

2. Log in results.csv:
   ```
   exp_id,phase,tau_low,tau_high,vlm_call_rate,semantic_retention,sce,latency_ms,gpu_util
   1,threshold_sweep,0.05,0.25,...
   2,threshold_sweep,0.05,0.30,...
   ...
   ```

**Analysis After Phase 1:**

1. **Plot Pareto frontier:**
   - X-axis: vlm_call_rate (lower is better = more efficient)
   - Y-axis: semantic_retention (higher is better = more fidelity)
   - Color: SCE (brighter = better)
   - Mark Pareto-optimal points (red stars)

2. **Identify optimal configuration:**
   ```
   best_sce_config = argmax(sce) over all (τ_low, τ_high)
   ```

3. **Answer these questions:**
   - **Q1:** Which (τ_low, τ_high) pair gives highest SCE?
   - **Q2:** Is the Pareto frontier linear or convex?
   - **Q3:** What is the trade-off slope (Δ retention / Δ efficiency)?
   - **Q4:** Are there "elbow" points where slope changes sharply?
   - **Q5:** How sensitive is SCE to τ_low vs. τ_high?

**Success Criterion:** Pareto frontier plot generated, optimal thresholds identified, questions answered.

**Log Entry Format:**
```
### Phase 1 — Threshold Sweep
**Status:** [Complete]
**Experiments:** 28 configurations tested
**Video:** [name]
**Observations:**
- Pareto frontier shows [linear/convex/complex] shape
- Highest SCE: 0.82 at (τ_low=0.15, τ_high=0.40)
- Trade-off slope: [retention/efficiency ratio]
**Optimal Thresholds Found:**
- τ_low = 0.15 (or best from results)
- τ_high = 0.40 (or best from results)
- SCE = 0.82
- Semantic retention = 94%
- VLM call rate = 0.042 calls/sec
**Interpretation:** [why this makes sense]
**Weaknesses:**
- Single video, may not generalize
- VLM skipped (using heuristic retention estimate)
- Camera motion effects not analyzed
**Next Phase:** Ablation studies to understand which drift components matter most
```

---

## Phase 2: Ablation Studies

**Goal:** Understand which semantic drift components (embedding, objects, tracking) are most important.

**Scientific Question:** How much does each component of D_t = α·D_embed + β·D_objects + γ·D_track contribute to scheduling effectiveness?

**Method:**

Run **5 configurations**:

1. **Baseline:** α=0.5, β=0.3, γ=0.2 (from Phase 0 + Phase 1 optimal)
2. **No Embedding:** α=0.0, β=0.5, γ=0.5 (remove CLIP drift)
3. **No Objects:** α=1.0, β=0.0, γ=0.0 (remove object novelty)
4. **No Tracking:** α=0.6, β=0.4, γ=0.0 (remove tracking drift)
5. **Uniform:** α=0.33, β=0.33, γ=0.33 (equal contribution)

**Use thresholds from Phase 1 (best τ_low, τ_high)** for all 5 runs.

**For each ablation:**
1. Run pipeline and record same metrics as Phase 1
2. Log in ablation_results.csv

**Analysis After Phase 2:**

1. **Compute importance for each component:**
   ```
   baseline_sce = 0.82
   
   importance_alpha = (baseline_sce - sce_no_alpha) / baseline_sce * 100
   importance_beta = (baseline_sce - sce_no_beta) / baseline_sce * 100
   importance_gamma = (baseline_sce - sce_no_gamma) / baseline_sce * 100
   ```

2. **Rank components by importance (which removal hurts most?)**

3. **Answer these questions:**
   - **Q1:** Which component is most important (highest SCE drop)?
   - **Q2:** Can we remove any component without major loss?
   - **Q3:** Does uniform weighting work better than baseline?
   - **Q4:** Are components additive or interactive?
   - **Q5:** Should we rebalance weights based on sensitivity?

**Success Criterion:** Ablation results show clear ranking of component importance.

**Log Entry Format:**
```
### Phase 2 — Ablation Studies
**Status:** [Complete]
**Experiments:** 5 configurations (baseline + 4 ablations)
**Video:** [same as Phase 1]
**Observations:**
- Baseline (α=0.5, β=0.3, γ=0.2) → SCE=0.82
- No embedding (α=0.0, ...) → SCE=0.65 (Δ = -0.17, -20.7%)
- No objects (α=1.0, ...) → SCE=0.71 (Δ = -0.11, -13.4%)
- No tracking (α=0.6, ...) → SCE=0.79 (Δ = -0.03, -3.7%)
- Uniform (α=0.33, ...) → SCE=0.78 (Δ = -0.04, -4.9%)

**Ranking of Component Importance:**
1. Embedding (α): -20.7% when removed (CRITICAL)
2. Objects (β): -13.4% when removed (IMPORTANT)
3. Tracking (γ): -3.7% when removed (MINOR)

**Interpretation:**
- CLIP semantic drift is the dominant signal
- Object novelty adds meaningful contribution
- Tracking is supporting signal, not primary
**Weaknesses:**
- Single video analysis only
- No statistical significance testing
- Interactive effects not measured
**Next Phase:** Failure analysis on diverse scenarios
```

---

## Phase 3: Failure Analysis (Edge Cases)

**Goal:** Test robustness and identify failure modes of the optimal scheduler.

**Scientific Question:** When does the semantic scheduler fail? What are the edge cases and confounding factors?

**Method:**

Test the **best configuration from Phase 1** (optimal τ_low, τ_high) on challenging scenarios:

**Scenario 1: Stable Scene (Negative Control)**
- Use a video segment with minimal semantic change
- Expected: Very low VLM call rate, high semantic retention
- Check: Does scheduler avoid false triggers?

**Scenario 2: Rapid Transitions (Positive Control)**
- Use a segment with abrupt semantic changes
- Expected: VLM called at transitions, captures all changes
- Check: Does scheduler catch real changes?

**Scenario 3: Camera Motion (Confounding Variable)**
- Use a segment with camera panning/zooming but stable semantics
- Expected: VLM should NOT be called (no semantic change)
- Check: Does optical flow contaminate D_embed?

**Scenario 4: Lighting Changes (Confounding Variable)**
- Use a segment with shadows, brightness changes, but stable semantics
- Expected: VLM should NOT be called
- Check: Is CLIP drift sensitive to lighting?

**Scenario 5: Object Persistence (Tracking Robustness)**
- Use a segment where objects enter/exit frequently
- Expected: Birth/death don't trigger unnecessary VLM calls
- Check: Does τ_low prevent false triggers from object motion?

**For each scenario:**
1. Identify or create test segment
2. Run with optimal config from Phase 1
3. Record:
   - `false_positives`: VLM called when semantics didn't change
   - `false_negatives`: VLM skipped when semantics DID change
   - `scenario_name`: Name of scenario
   - Other standard metrics

4. Visualize:
   - Drift curve with VLM calls marked
   - Optical flow overlay (to see camera motion)
   - Object tracks overlay

**Analysis After Phase 3:**

Answer the 10-question failure analysis framework:

1. **What happened?** (Describe the scenario)
2. **Why did it happen?** (Root cause)
3. **Which semantic pattern caused it?** (D_t behavior)
4. **Did VLM invocation make sense given the scene?** (Was decision correct?)
5. **Could VLM have been skipped?** (False positive check)
6. **Was semantic information lost?** (False negative check)
7. **Were there false triggers?** (Unnecessary VLM calls)
8. **Were there missed triggers?** (Missed changes)
9. **Did camera motion affect semantic drift?** (Confounding analysis)
10. **What evidence supports this conclusion?** (Metrics + visualization)

**Success Criterion:** All scenarios tested, false positives/negatives quantified, failure modes documented.

**Log Entry Format:**
```
### Phase 3 — Failure Analysis
**Status:** [Complete]
**Scenarios Tested:** 5 (stable, transitions, camera, lighting, tracking)

**Scenario 1: Stable Scene**
- False positives: 2 (unnecessary VLM calls)
- False negatives: 0
- Interpretation: τ_low=0.15 triggers on minor noise

**Scenario 2: Rapid Transitions**
- False positives: 0
- False negatives: 1 (missed 1 transition)
- Interpretation: τ_high=0.40 may be too strict for abrupt changes

**Scenario 3: Camera Motion**
- False positives: 3 (camera motion mistaken for semantic change)
- False negatives: 0
- Interpretation: OPTICAL FLOW CONTAMINATION — need optical flow subtraction

**Scenario 4: Lighting Changes**
- False positives: 1 (shadow caused CLIP drift)
- False negatives: 0
- Interpretation: CLIP sensitive to lighting, may need preprocessing

**Scenario 5: Object Tracking**
- False positives: 0
- False negatives: 0
- Interpretation: Object-based signals working well

**Summary Failure Modes:**
1. Camera motion → false positives (need to subtract optical flow from D_embed)
2. Lighting → false positives (need lighting-invariant preprocessing)
3. Stable scenes → borderline true positives (τ_low might be lowered)
4. Rapid transitions → borderline false negatives (τ_high might be raised)

**Recommendations for Next Iteration:**
- Add optical flow subtraction: D_embed_corrected = D_embed - optical_flow_magnitude
- Test CLIP with lighting-invariant preprocessing
- Re-sweep τ_low, τ_high with new D_t formulation
- Validate on multi-scenario video

**Weaknesses:**
- Scenarios are synthetic/curated (may not reflect real deployment)
- Optical flow assumption may not hold for occlusions
- Single camera, single environment
```

---

## Overall Success Criteria

✅ **Phase 0 complete:** Baseline metrics recorded
✅ **Phase 1 complete:** Pareto frontier plotted, optimal thresholds found
✅ **Phase 2 complete:** Component importance ranked
✅ **Phase 3 complete:** Failure modes identified, robustness understood

**Thesis Ready For Section 4 (Experiments) when:**
1. Results CSV has ≥25 experiments across all phases
2. Pareto frontier plot shows clear trade-off
3. Ablation results rank components by importance
4. Failure analysis identifies 3+ specific confounding factors
5. All results logged in research_log.md with evidence
6. Recommendations for next iteration documented

---

## Key Metrics to Track

| Metric | Definition | Why It Matters |
|--------|-----------|-----------------|
| `vlm_call_rate` | VLM calls per second | Measures compute efficiency |
| `semantic_retention` | % of frames where scheduler decision was valid | Measures fidelity |
| `sce` | semantic_retention / vlm_call_rate | Primary optimization metric |
| `false_positives` | VLM called when semantics stable | Indicates over-triggering |
| `false_negatives` | VLM skipped when semantics changed | Indicates under-triggering |
| `latency_mean_ms` | Average per-frame latency | Deployment constraint |
| `gpu_utilization_%` | GPU load | Resource efficiency |

---

## How to Run This Program

On **Kaggle** or **local machine**:

```python
# Phase 0: Baseline
python scripts/run_pipeline.py \
  --video test_video.mp4 \
  --tau_low 0.15 --tau_high 0.40 \
  --alpha 0.5 --beta 0.3 --gamma 0.2 \
  --skip_vlm \
  --output_dir results/phase0/

# Phase 1: Threshold Sweep (use threshold_sweep.py or loop through configs)
python scripts/sweep_thresholds.py \
  --video test_video.mp4 \
  --output_dir results/phase1/

# Phase 2: Ablations (run pipeline 5 times with different alpha, beta, gamma)
python scripts/run_pipeline.py --alpha 0.0 --beta 0.5 --gamma 0.5 ...
python scripts/run_pipeline.py --alpha 1.0 --beta 0.0 --gamma 0.0 ...
# ... (repeat for other ablations)

# Phase 3: Failure Analysis (run on multiple scenario videos)
python scripts/run_pipeline.py --video scenario_stable.mp4 ...
python scripts/run_pipeline.py --video scenario_motion.mp4 ...
# ... (repeat for other scenarios)

# Analyze & Export
python scripts/analyze_results.py results/ --output thesis_section_4.md
```

---

## Final Output for Thesis

After completing all phases, you will have:

📊 **results.csv** — 25+ experiments with all metrics
📈 **pareto_frontier.png** — Visualization of semantic retention vs efficiency trade-off
📋 **research_log.md** — Full experiment log with hypotheses, observations, conclusions
🎯 **optimal_config.json** — Best configuration discovered (τ_low, τ_high, α, β, γ)
🔍 **failure_analysis.md** — Edge cases, confounding factors, recommendations
📄 **thesis_section_4.md** — Results ready to paste into thesis

---

**This program enforces the scientific method. Observe → Hypothesize → Test (one thing) → Evaluate → Record.**

**Every claim in your thesis must be backed by an experiment logged in research_log.md.**
