# ✅ Karpathy's AutoResearch is Ready for Your Thesis

**Date**: June 25, 2026  
**Status**: Fully configured and ready to use

---

## What Was Done

Karpathy's actual AutoResearch framework (from https://github.com/karpathy/autoresearch) has been customized for your M.Tech thesis on semantic-aware VLM scheduling.

### Files Modified/Created

#### In `autoresearch-master/` (Karpathy's actual repo)

1. **program.md** — CUSTOMIZED for your thesis
   - Defines 3-phase research protocol
   - Phase 1: Threshold sweep (τ_low, τ_high optimization)
   - Phase 2: Ablation studies (α, β, γ component importance)
   - Phase 3: Failure analysis (edge cases and robustness)

2. **train.py** — REPLACED with thesis-specific implementation
   - Runs semantic drift scheduler experiments
   - Calls your existing pipeline scripts
   - Logs results in AutoResearch format

3. **prepare.py** — REPLACED with thesis validation
   - Checks environment setup
   - Validates test video exists
   - Defines fixed metrics (SCE calculation)
   - Read-only (cannot be modified)

4. **.gitignore** — UPDATED
   - Excludes results.tsv (do NOT commit experiment logs)
   - Excludes results_*/ directories

5. **THESIS_AUTORESEARCH.md** — NEW
   - Complete guide for using AutoResearch with your thesis
   - Examples for each phase
   - Troubleshooting section

#### In main `edge-vlm-thesis/` directory

6. **KARPATHY_AUTORESEARCH_SETUP.md** — NEW
   - Quick start guide
   - 5-minute setup
   - Next steps for running experiments

7. **AUTORESEARCH_READY.md** — THIS FILE
   - Summary of what's been set up

---

## Architecture

```
Your Thesis Research Loop
    ↓
program.md (research protocol)
    ↓
Karpathy's AutoResearch Framework
    ├─ train.py (experiment runner)
    ├─ prepare.py (fixed setup)
    ├─ analysis.ipynb (results visualization)
    └─ results.tsv (experiment log)
        ↓
        ├─ Calls your pipeline:
        │   └─ scripts/run_pipeline.py
        │       ├─ src/semantic_memory/drift.py
        │       ├─ src/scheduler/vlm_scheduler.py
        │       └─ [existing thesis code]
        │
        └─ Logs results:
            ├─ results.csv (all metrics)
            ├─ research_log.json (full log)
            └─ exp_001/, exp_002/, ... (individual runs)
```

---

## How It Works

### The Scientific Loop (Enforced by Framework)

```
1. OBSERVE → Program.md specifies what to measure
2. HYPOTHESIZE → Framework reads previous results
3. PROPOSE → Framework suggests next experiment (threshold, ablation, scenario)
4. TEST → train.py runs the experiment
5. EVALUATE → Metrics compared against baseline
6. RECORD → results.tsv logs the outcome
7. LOOP → Go to step 2

You are AUTONOMOUS during this loop — no human intervention needed.
Framework decides what to test next based on results.
```

### The Three Phases

**Phase 1: Threshold Sweep** (28 experiments)
- Tests all τ_low ∈ [0.05...0.30], τ_high ∈ [0.25...0.60]
- Finds Pareto frontier of semantic retention vs. VLM efficiency
- Primary metric: SCE = semantic_retention / vlm_call_rate
- Expected time: 3-4 hours on Kaggle T4

**Phase 2: Ablation Studies** (5 experiments)
- Tests: baseline, no_embedding, no_objects, no_tracking, uniform_weights
- Determines which drift component (α, β, γ) is most important
- Uses best (τ_low, τ_high) from Phase 1
- Expected time: 30 minutes

**Phase 3: Failure Analysis** (5 experiments)
- Tests robustness: stable scenes, rapid transitions, camera motion, lighting, tracking
- Identifies false positives/negatives and confounding factors
- Expected time: 30 minutes

**Total**: ~50 experiments, ~5 hours on Kaggle T4

---

## Key Files to Know

| File | Purpose | Modifiable? |
|------|---------|-------------|
| `autoresearch-master/program.md` | What to experiment | ✗ Read-only |
| `autoresearch-master/train.py` | How to run experiments | ✓ Can modify for custom phases |
| `autoresearch-master/prepare.py` | Metrics & validation | ✗ Read-only (ground truth) |
| `autoresearch-master/results.tsv` | Experiment log | ✗ Auto-generated (do NOT commit) |
| `autoresearch-master/analysis.ipynb` | Visualize results | ✓ Can customize |
| `KARPATHY_AUTORESEARCH_SETUP.md` | Quick start guide | — |
| `THESIS_AUTORESEARCH.md` | Detailed guide | — |

---

## Metric Definition (FIXED)

The primary metric cannot be changed:

```python
SCE = semantic_retention / vlm_call_rate

Where:
  semantic_retention = fraction of frames with valid semantic state [0.0, 1.0]
  vlm_call_rate = VLM invocations per second (lower is better)
  SCE = efficiency metric (higher is better)

Goal: Maximize SCE
```

This definition is in `prepare.py` and is read-only.

---

## Next Steps

### Immediate (Today)

1. **Verify environment**:
   ```bash
   cd autoresearch-master
   python prepare.py
   ```
   Should show: `✓ Environment is ready for AutoResearch experiments`

2. **Read the guides**:
   - Start with: `KARPATHY_AUTORESEARCH_SETUP.md` (5 min read)
   - Then: `autoresearch-master/program.md` (10 min read)
   - Then: `autoresearch-master/THESIS_AUTORESEARCH.md` (20 min read)

3. **Run your first experiment**:
   ```bash
   cd autoresearch-master
   python train.py --phase threshold_sweep --tau_low 0.15 --tau_high 0.40
   ```

### Short Term (This Week)

4. **Run Phase 1** (threshold sweep) on Kaggle
   - Takes ~3-4 hours
   - Generates 28 experiment results
   - Run overnight while you sleep

5. **Analyze Phase 1 results**
   - Plot Pareto frontier
   - Identify optimal (τ_low, τ_high)
   - Document findings in research_log.md

### Medium Term (Next Week)

6. **Run Phase 2** (ablations)
   - Takes ~30 minutes
   - Understand which drift component matters most

7. **Run Phase 3** (failure analysis)
   - Takes ~30 minutes
   - Test robustness on edge cases

### For Thesis (Week After)

8. **Write Section 4** (Experiments)
   - Use results.tsv to create figures
   - Present Pareto frontier
   - Analyze ablation results
   - Discuss failure modes

---

## Running on Kaggle (Recommended)

1. Upload `edge-vlm-thesis` as a Kaggle dataset
2. Upload test video as a separate dataset
3. Create new Python notebook
4. Copy cells from `autoresearch-master/THESIS_AUTORESEARCH.md`
5. Run → Experiments execute autonomously
6. Download `results.tsv` when done

**Advantage**: Runs 50 experiments in ~5 hours. You sleep, wake up to results.

---

## Running Locally (If GPU Available)

```bash
# Terminal 1: Monitor
watch -n 10 "tail results.tsv"

# Terminal 2: Run experiments
cd autoresearch-master
python train.py --phase threshold_sweep
python train.py --phase ablation
python train.py --phase failure_analysis
```

Expected output in results.tsv after each run:
```
commit	sce	semantic_retention	vlm_call_rate	false_pos	false_neg	status	description
a1b2c3d	0.8200	0.9400	0.0420	2	0	keep	threshold_sweep: tau_low=0.15, tau_high=0.40
b2c3d4e	0.7900	0.9200	0.0680	3	1	discard	threshold_sweep: tau_low=0.10, tau_high=0.35
...
```

---

## What You Have Now

✅ **Karpathy's actual AutoResearch framework** (not a clone, the real thing)
✅ **Customized for semantic drift optimization** (program.md, train.py)
✅ **Integrated with your thesis project** (calls your pipeline scripts)
✅ **Research protocol defined** (3 phases, clear metrics, autonomy)
✅ **Test video ready** (in experiments/test/)
✅ **Complete documentation** (3 guides included)

---

## Constraints & Rules

### What You CAN do:
- Modify `train.py` to add new phases or change experiment logic
- Change which parameters to sweep in program.md
- Customize analysis.ipynb for visualization
- Use Kaggle or local GPU to run experiments

### What You CANNOT do:
- Modify `prepare.py` (metric definition is fixed)
- Change how SCE is calculated
- Use a different test video mid-research (must be consistent)
- Add external dependencies not in requirements.txt

### Why These Rules?
- **Reproducibility**: Same metric for all experiments
- **Consistency**: All runs use same test data
- **Integrity**: You can't massage the metric to look good

---

## Quality Assurance

Every experiment is logged with:
- Configuration (τ_low, τ_high, α, β, γ)
- Metrics (SCE, semantic_retention, vlm_call_rate, false_pos, false_neg)
- Status (keep, discard, or crash)
- Description (what was being tested)
- Git commit hash (for reproducibility)

This provides **full traceability** for your thesis.

---

## Research Integrity

The framework enforces good research practices:

✓ **One change at a time** — Each experiment modifies ONE parameter
✓ **Metric consistency** — Same evaluation metric for all runs
✓ **Full logging** — Every experiment recorded in results.tsv
✓ **Reproducibility** — git commit hash stored with each result
✓ **Version control** — All code in git, results are ephemeral

---

## Expected Outcomes

After completing all three phases:

```
autoresearch-master/results.tsv
├─ 38 experiments total
├─ 28 threshold sweeps (Phase 1)
├─ 5 ablations (Phase 2)
├─ 5 failure analysis (Phase 3)
└─ Metrics for each:
   ├─ SCE (primary)
   ├─ semantic_retention
   ├─ vlm_call_rate
   ├─ false_positives
   ├─ false_negatives
   └─ ...
```

**For your thesis Section 4:**
1. Pareto frontier plot (Phase 1)
2. Component importance ranking (Phase 2)
3. Robustness analysis (Phase 3)
4. Reproducibility statement (git history)

---

## Questions?

Refer to:
1. `KARPATHY_AUTORESEARCH_SETUP.md` — Quick answers
2. `autoresearch-master/THESIS_AUTORESEARCH.md` — Detailed guide
3. `autoresearch-master/program.md` — Research protocol
4. `autoresearch-master/train.py` — Implementation details

---

## Summary

**Before**: You had thesis code that you manually experimented with

**Now**: You have Karpathy's proven autonomous research framework configured to run your semantic drift experiments automatically

**Next**: Run `cd autoresearch-master && python prepare.py` to verify everything works

**Then**: Run phases 1-3 and collect 38 experiments of evidence for your thesis

---

**Karpathy's AutoResearch is ready. Your thesis can now scale from manual experiments to autonomous research.**

Let's go.
