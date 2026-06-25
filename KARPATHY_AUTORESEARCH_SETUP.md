# Karpathy's AutoResearch Setup for Your Thesis

Your edge-vlm-thesis project now includes **Karpathy's actual AutoResearch framework** customized for semantic drift optimization.

---

## What You Have

```
edge-vlm-thesis/
├── autoresearch-master/          ← Karpathy's AutoResearch (CUSTOMIZED FOR YOUR THESIS)
│   ├── program.md                ← Research protocol (3 phases)
│   ├── train.py                  ← Experiment runner (modified for scheduler)
│   ├── prepare.py                ← Fixed setup & metrics (read-only)
│   ├── results.tsv               ← Auto-generated experiment log
│   ├── analysis.ipynb            ← Results analysis
│   └── THESIS_AUTORESEARCH.md    ← Detailed guide
│
├── src/                          ← Your thesis code (drift, scheduler, etc.)
├── scripts/                      ← Your pipeline scripts (run_pipeline.py, etc.)
├── experiments/                  ← Test videos
└── README.md                     ← Project overview
```

---

## Quick Start (5 minutes)

### Step 1: Verify Everything Works

```bash
cd autoresearch-master
python prepare.py
```

Expected output:
```
✓ Environment is ready for AutoResearch experiments
```

If you see errors, fix the issues (missing video, missing packages, etc.)

### Step 2: Understand the Research Plan

Read these in order:

1. **program.md** — Your 3-phase research protocol
   - Phase 1: Threshold sweep (find optimal τ_low, τ_high)
   - Phase 2: Ablation studies (understand component importance)
   - Phase 3: Failure analysis (test edge cases)

2. **THESIS_AUTORESEARCH.md** — Detailed how-to guide

3. **train.py** — The experiment runner (what actually runs)

### Step 3: Run Your First Experiment

```bash
cd autoresearch-master

# Establish baseline (Phase 1, first config)
python train.py --phase threshold_sweep --tau_low 0.15 --tau_high 0.40

# See output:
# ---
# sce:                  0.8200
# semantic_retention:   0.9400
# vlm_call_rate:        0.0420
# ...

# Check results
tail results.tsv
```

### Step 4: Run All Threshold Sweeps

The framework generates all combinations automatically:

```bash
python train.py --phase threshold_sweep
# Runs 28 experiments (all tau_low x tau_high combinations)
# Takes ~3-4 hours
```

### Step 5: Analyze Phase 1 Results

```bash
# Plot Pareto frontier from results.tsv
python analysis.ipynb  # Open in Jupyter
```

### Step 6: Run Ablations (Phase 2)

```bash
python train.py --phase ablation
# Runs 5 ablation studies
# Takes ~30 minutes
```

### Step 7: Run Failure Analysis (Phase 3)

```bash
python train.py --phase failure_analysis --scenario stable
python train.py --phase failure_analysis --scenario motion
# ... (more scenarios)
# Takes ~30 minutes
```

---

## The Autonomous Research Loop

This is the powerful part. Instead of manually running experiments, let the framework do it:

### Option A: Kaggle Notebook (Recommended)

1. Upload edge-vlm-thesis to Kaggle datasets
2. Create new Python notebook
3. Copy cells from KAGGLE_NOTEBOOK_TEMPLATE.md (in autoresearch-master/)
4. Run → AutoResearch executes all 50+ experiments automatically while you sleep

### Option B: Local Machine with Git

```bash
# Create research branch
git checkout -b autoresearch/jun25

cd autoresearch-master

# Run Phase 1 autonomously
python train.py --phase threshold_sweep

# Run Phase 2 autonomously
python train.py --phase ablation

# Run Phase 3 autonomously
python train.py --phase failure_analysis

# Commit results (not results.tsv — that's untracked)
git commit -am "Phase 1-3 complete: SCE=0.82"
git push origin autoresearch/jun25
```

---

## Understanding the Files

| File | Purpose | Modify? |
|------|---------|---------|
| **program.md** | What to experiment (3 phases) | ✗ Read-only |
| **train.py** | How to run experiments | ✓ Can modify |
| **prepare.py** | Fixed setup & metrics | ✗ Read-only |
| **results.tsv** | Experiment log (auto-generated) | ✗ Do NOT commit |
| **THESIS_AUTORESEARCH.md** | Detailed guide | — |

### Key Constraint

**The metric SCE is FIXED and cannot be changed:**

```python
SCE = semantic_retention / vlm_call_rate
```

You can only improve it by finding better thresholds (τ_low, τ_high) and weights (α, β, γ).

---

## Expected Results

After completing all three phases, you'll have:

```
autoresearch-master/results.tsv
├─ 28 threshold sweep experiments (Phase 1)
├─ 5 ablation studies (Phase 2)
├─ 5 failure analysis experiments (Phase 3)
└─ Total: 38 experiments logged

Metrics recorded for each:
- sce (primary metric)
- semantic_retention
- vlm_call_rate
- false_positives
- false_negatives
- [and more]
```

**For your thesis:**

1. **Section 4 Figures**: Plot Pareto frontier from results.tsv
2. **Component Analysis**: Which drift component matters most (Phase 2)
3. **Robustness**: False positive/negative rates by scenario (Phase 3)
4. **Reproducibility**: results.tsv shows every experiment

---

## Examples

### Running Phase 1 (Threshold Sweep)

```bash
# Automatic: generates all 28 combinations
python train.py --phase threshold_sweep

# Or manual: run specific combo
python train.py --phase threshold_sweep --tau_low 0.15 --tau_high 0.40
```

### Running Phase 2 (Ablations)

```bash
# After Phase 1, find best tau_low, tau_high from results.tsv

# Then test ablations:
python train.py --phase ablation --alpha 0.0 --beta 0.5 --gamma 0.5  # No embedding
python train.py --phase ablation --alpha 1.0 --beta 0.0 --gamma 0.0  # No objects
python train.py --phase ablation --alpha 0.6 --beta 0.4 --gamma 0.0  # No tracking
python train.py --phase ablation --alpha 0.33 --beta 0.33 --gamma 0.33  # Uniform
```

### Running Phase 3 (Failure Analysis)

```bash
# Test robustness on challenging scenarios
python train.py --phase failure_analysis --scenario stable
python train.py --phase failure_analysis --scenario motion
python train.py --phase failure_analysis --scenario lighting
```

---

## Troubleshooting

### Error: "No test video found"

```bash
# Add a test video to experiments/test/
mkdir -p experiments/test/
# Copy your video here as test_video.mp4
```

### Error: "prepare.py validation failed"

```bash
# Install missing packages
pip install -r ../requirements.txt

# Or check that src/ exists
ls ../src/semantic_memory/
ls ../src/scheduler/
```

### Error: "train.py times out"

```bash
# Check that pipeline script exists
ls ../scripts/run_pipeline.py

# Or reduce the time budget in prepare.py
# TIME_BUDGET = 600  # Currently 10 minutes
```

---

## Next Steps

1. ✅ Read this file (you are here)
2. → Read `autoresearch-master/THESIS_AUTORESEARCH.md` for detailed guide
3. → Read `autoresearch-master/program.md` to understand the 3 phases
4. → Run `python autoresearch-master/prepare.py` to verify setup
5. → Run Phase 1: `python autoresearch-master/train.py --phase threshold_sweep`
6. → Analyze results and write thesis Section 4

---

## Key Principle

**You're not manually coding the scheduler optimization. You're defining a research protocol (program.md) and letting the framework autonomously run it.**

This is Karpathy's insight: the AI can be the researcher, and you're the advisor who designs the protocol.

---

## References

- **Original AutoResearch**: https://github.com/karpathy/autoresearch
- **Detailed Guide**: `autoresearch-master/THESIS_AUTORESEARCH.md`
- **Research Protocol**: `autoresearch-master/program.md`
- **Experiment Code**: `autoresearch-master/train.py`
- **Project Context**: `README.md`

---

**Ready to run semantic drift research autonomously.**

The framework is configured, the protocol is defined, and the test video is ready.

Start with: `cd autoresearch-master && python prepare.py`
